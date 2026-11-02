"""Lexicon rescoring with the recognizer's own CTC posteriors.

For every out-of-lexicon letter run in the greedy output, candidate words close in edit distance are
scored by their CTC log-likelihood over the frames between the run's neighbouring characters:
    score(c) = log P_ctc(c | frames) + alpha * [c in lexicon]
The greedy word competes with alpha = 0. Posteriors are stored as the top-k classes per frame
(remaining mass floored), which keeps caches small and makes the loss an approximation.
"""
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .context_lm import NgramLM, decode_line
from .edit_costs import EditCosts
from .postprocess import FoldIndex, fold, is_known, nearest

FLOOR = 1e-6
_WORDS = re.compile(r'[^\W\d_]+')


@dataclass(frozen=True)
class LineLogits:
    prediction: str
    top_idx: np.ndarray  # (T, k) int16 class ids
    top_val: np.ndarray  # (T, k) float16 probabilities


def empty_line(k: int = 20) -> LineLogits:
    return LineLogits('', np.zeros((0, k), np.int16), np.zeros((0, k), np.float16))


def compress(probs: torch.Tensor, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    values, indices = probs.float().topk(k, dim=0)  # probs: (C, T)
    return indices.T.cpu().numpy().astype(np.int16), values.T.cpu().numpy().astype(np.float16)  # cuda/mps -> host


def merge_lines(lines: list[LineLogits], n_classes: int, l2c: dict[int, str], k: int = 20) -> LineLogits:
    """Average the posteriors of one line read from several versions of the image (test-time augmentation);
    the versions share the polygons, so frames line up. Lines whose frame count differs are left out."""
    usable = [line for line in lines if line.top_idx.shape == lines[0].top_idx.shape]
    if len(usable) < 2 or not usable[0].top_idx.shape[0]:
        return lines[0]
    total = torch.zeros((usable[0].top_idx.shape[0], n_classes))
    for line in usable:
        dense = torch.zeros_like(total)
        dense.scatter_(1, torch.from_numpy(line.top_idx.astype(np.int64)), torch.from_numpy(line.top_val.astype(np.float32)))
        total += dense
    idx, val = compress((total / len(usable)).T, k)
    best, prev, out = idx[:, 0].astype(int), 0, []
    for label in best:
        if label != prev and label != 0:
            out.append(l2c.get(label, ''))
        prev = label
    return LineLogits(unicodedata.normalize('NFC', ''.join(out)), idx, val)


def _log_probs(line: LineLogits, n_classes: int, start: int, end: int) -> torch.Tensor:
    idx = torch.from_numpy(line.top_idx[start:end].astype(np.int64))
    val = torch.from_numpy(line.top_val[start:end].astype(np.float32))
    dense = torch.full((end - start, n_classes), FLOOR)
    dense.scatter_(1, idx, val.clamp_min(FLOOR))
    return (dense / dense.sum(1, keepdim=True)).log()


def _greedy_runs(line: LineLogits, l2c: dict[int, str]) -> list[tuple[str, int, int]]:
    """(char, first_frame, last_frame) for each emitted label of the best path."""
    best = line.top_idx[:, 0].astype(int)
    runs, prev = [], 0
    for t, label in enumerate(best):
        if label != prev and label != 0:
            runs.append([l2c.get(label, ''), t, t])
        elif label == prev and label != 0:
            runs[-1][2] = t
        prev = label
    return [tuple(r) for r in runs]


def _is_letter(char: str) -> bool:
    return bool(char) and unicodedata.category(char)[0] in 'LM'


def _letter_words(runs: list[tuple[str, int, int]], n_frames: int) -> list[tuple[int, int, int, int]]:
    """(first_run, last_run, start_frame, end_frame) for maximal letter runs; frames span to neighbours."""
    words, i = [], 0
    while i < len(runs):
        if not _is_letter(runs[i][0]):
            i += 1
            continue
        j = i
        while j + 1 < len(runs) and _is_letter(runs[j + 1][0]):
            j += 1
        start = runs[i - 1][2] + 1 if i > 0 else 0
        end = runs[j + 1][1] if j + 1 < len(runs) else n_frames
        words.append((i, j, start, end))
        i = j + 1
    return words


def _match_case(candidate: str, original: str) -> str:
    if original.isupper() and len(original) > 1:
        return candidate.upper()
    if original[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate


def _encode(word: str, c2l: dict[str, list[int]]) -> list[int] | None:
    labels = []
    for char in unicodedata.normalize('NFD', word):
        if char not in c2l:
            return None
        labels.extend(c2l[char])
    return labels


def _ctc_scores(log_probs: torch.Tensor, targets: list[list[int]]) -> torch.Tensor:
    lp = log_probs.unsqueeze(1).expand(-1, len(targets), -1)
    flat = torch.tensor([label for t in targets for label in t], dtype=torch.long)
    in_lens = torch.full((len(targets),), log_probs.shape[0], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    loss = F.ctc_loss(lp, flat, in_lens, tgt_lens, blank=0, reduction='none', zero_infinity=False)
    return -loss


@dataclass(frozen=True)
class RescoreParams:
    alpha: float = 4.0
    max_ratio: float = 0.5
    candidates: int = 10
    min_len: int = 3
    min_edits: int = 1  # candidates may always differ by at least this many edits, however short the word
    pool: int = 0  # >0: score this many nearest lexicon words with CTC and keep the best `candidates` of them
    nbest: int = 0  # >0: also take lexicon neighbours of the word's nbest most likely readings (prefix beam search)
    protect: float = 0.0  # >0: an in-lexicon word whose every letter has at least this posterior stays as read
    coherent: bool = False  # an in-lexicon word whose pairs with both read neighbours occur in the LM corpus stays
    costs: EditCosts | None = None  # learned edit costs: also lexicon words reached by undoing learned confusions
    cost_max: float = 0.0  # ... at this total cost at most


def _confident(line: LineLogits, runs: list[tuple[str, int, int]], span: tuple[int, int, int, int],
               threshold: float) -> bool:
    """Every letter of the read word peaks at a posterior of at least threshold (0 = never)."""
    if threshold <= 0:
        return False
    first, last = span[0], span[1]
    return all(float(line.top_val[a:b + 1, 0].max()) >= threshold for _, a, b in runs[first:last + 1])


def _nbest_readings(line: LineLogits, start: int, end: int, l2c: dict[int, str], n: int,
                    beam: int = 16, min_prob: float = 1e-3) -> list[str]:
    """The n most likely label sequences over frames [start, end), by CTC prefix beam search on the stored top-k
    posteriors (probabilities renormalized over the beam at every frame)."""
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (1.0, 0.0)}  # prefix -> (p ends in blank, p not)
    for t in range(start, end):
        nxt: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])
        for prefix, (p_blank, p_label) in beams.items():
            for label, p in zip(line.top_idx[t].tolist(), line.top_val[t].astype(float).tolist()):
                if p < min_prob:
                    continue
                if label == 0:
                    nxt[prefix][0] += (p_blank + p_label) * p
                elif prefix and label == prefix[-1]:
                    nxt[prefix + (label,)][1] += p_blank * p
                    nxt[prefix][1] += p_label * p
                else:
                    nxt[prefix + (label,)][1] += (p_blank + p_label) * p
        ranked = sorted(nxt.items(), key=lambda item: -sum(item[1]))[:beam]
        total = sum(sum(v) for _, v in ranked) or 1.0
        beams = {k: (v[0] / total, v[1] / total) for k, v in ranked}
    best = sorted(beams, key=lambda k: -sum(beams[k]))[:n]
    return [unicodedata.normalize('NFC', ''.join(l2c.get(label, '') for label in prefix)) for prefix in best]


def _word_slot(line: LineLogits, runs: list[tuple[str, int, int]], span: tuple[int, int, int, int],
               c2l: dict[str, list[int]], n_classes: int, known: frozenset[str], choices: tuple[str, ...],
               params: RescoreParams, open_known: bool, folded: FoldIndex | None = None,
               short: tuple[str, ...] = (), l2c: dict[int, str] | None = None,
               extra: tuple[str, ...] = ()) -> list[tuple[str, float]]:
    """(candidate, CTC log-likelihood + alpha if in lexicon) for one word; the read word first.
    A single entry means the word is left as read (too short, in the lexicon and not opened, or no candidate).
    With folded, candidates are found ignoring accents and each one scores as the better of its accented and
    accent-free spellings, so não can win over a read 'nao'/'nar' when the image shows no tilde.
    Words shorter than min_len are left as read unless short is given: then its words within one edit compete.
    extra: further lexicon words to score (the beam search's reading of this word)."""
    first, last, start, end = span
    original = unicodedata.normalize('NFC', ''.join(r[0] for r in runs[first:last + 1]))
    lower = original.lower()
    if (len(original) < params.min_len and not short) or end <= start or (
            is_known(lower, known, folded) and (not open_known or _confident(line, runs, span, params.protect))):
        return [(original, 0.0)]
    if len(original) < params.min_len:
        cands = [_match_case(c, original) for c in dict.fromkeys([*short, *extra])
                 if c != lower and (c in extra or Levenshtein.distance(fold(c), fold(lower)) <= 1)]
    else:
        cutoff = max(params.min_edits, int(len(original) * params.max_ratio))
        found = nearest(lower, choices, folded, cutoff, max(params.candidates, params.pool) + 1, params.costs,
                        params.cost_max)
        found = [c for c in found if c != lower][:max(params.candidates, params.pool)]
        if params.nbest and l2c:
            for reading in _nbest_readings(line, start, end, l2c, params.nbest):
                reading = reading.lower()
                if reading != lower and reading.isalpha():
                    found += nearest(reading, choices, folded, max(params.min_edits, int(len(reading) * params.max_ratio)),
                                     params.candidates)
        cands = [_match_case(c, original) for c in dict.fromkeys(found + list(extra)) if c != lower]
    encoded = [(c, e) for c, e in [(c, _encode(c, c2l)) for c in [original] + cands] if e and len(e) <= end - start]
    if len(encoded) < 2 or encoded[0][0] != original:
        return [(original, 0.0)]
    log_probs = _log_probs(line, n_classes, start, end)
    scores = _ctc_scores(log_probs, [e for _, e in encoded])
    if folded is not None:
        plain = [_encode(fold(c), c2l) for c, _ in encoded]
        plain = [p if p and len(p) <= end - start else e for p, (_, e) in zip(plain, encoded)]
        scores = torch.maximum(scores, _ctc_scores(log_probs, plain))
    slot = [(c, float(s) + (params.alpha if c.lower() in known else 0.0)) for (c, _), s in zip(encoded, scores)]
    return slot[:1] + sorted(slot[1:], key=lambda item: -item[1])[:params.candidates] if params.pool else slot


def _assemble(runs: list[tuple[str, int, int]], replacements: dict[int, tuple[int, str]]) -> str:
    out, i = [], 0
    while i < len(runs):
        if i in replacements:
            last, word = replacements[i]
            out.append(word)
            i = last + 1
        else:
            out.append(runs[i][0])
            i += 1
    return unicodedata.normalize('NFC', ''.join(out))


def rescore_line(line: LineLogits, c2l: dict[str, list[int]], l2c: dict[int, str], n_classes: int,
                 known: frozenset[str], choices: tuple[str, ...], params: RescoreParams,
                 folded: FoldIndex | None = None) -> str:
    """Each out-of-lexicon word on its own: the best of (read word, lexicon candidates) by CTC score + alpha."""
    runs = _greedy_runs(line, l2c)
    replacements: dict[int, tuple[int, str]] = {}
    for span in _letter_words(runs, line.top_idx.shape[0]):
        slot = _word_slot(line, runs, span, c2l, n_classes, known, choices, params, False, folded)
        best = max(range(len(slot)), key=lambda k: slot[k][1])
        if best:
            replacements[span[0]] = (span[1], slot[best][0])
    return _assemble(runs, replacements)


def _aligned_words(read: list[str], alternative: str, known: frozenset[str]) -> list[tuple[str, ...]]:
    """Per read word, the alternative reading's word replacing it one-to-one (if in the lexicon)."""
    other = [w.lower() for w in _WORDS.findall(alternative)]
    vocab: dict[str, int] = {}
    a = [vocab.setdefault(w.lower(), len(vocab)) for w in read]
    b = [vocab.setdefault(w, len(vocab)) for w in other]
    extras: list[tuple[str, ...]] = [()] * len(read)
    for op in Levenshtein.opcodes(a, b):
        if op.tag == 'replace' and op.src_end - op.src_start == op.dest_end - op.dest_start:
            for i, j in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
                if other[j] in known:
                    extras[i] = (other[j],)
    return extras


def _coherent(read: list[str], i: int, known: frozenset[str], lm: NgramLM) -> bool:
    """read[i] is in the lexicon and forms corpus-attested pairs with the read words around it."""
    word = read[i].lower()
    if word not in known:
        return False
    before = i == 0 or lm.seen(read[i - 1].lower(), word)
    return before and (i == len(read) - 1 or lm.seen(word, read[i + 1].lower()))


def _gaps(runs: list[tuple[str, int, int]], spans: list[tuple[int, int, int, int]]) -> list[str]:
    """Text of the runs before the first word, between consecutive words and after the last (len(spans) + 1)."""
    bounds = [-1] + [b for span in spans for b in (span[0], span[1])] + [len(runs)]
    return [''.join(r[0] for r in runs[bounds[k] + 1:bounds[k + 1]]) for k in range(0, len(bounds), 2)]


def _context_lm(words: list[str], lm: NgramLM) -> float:
    return sum(lm.logp(w.lower(), p.lower() if p else None) for p, w in zip([None] + words[:-1], words) if w)


@dataclass(frozen=True)
class RegionParams:
    margin: float = 0.0  # the alternative must beat the rescored text by this much (CTC log-lik + beta * LM)
    max_words: int = 3  # regions with more words on either side are left alone


def _region_score(line: LineLogits, c2l: dict[str, list[int]], n_classes: int, frames: tuple[int, int],
                  texts: list[str]) -> list[float] | None:
    encoded = [_encode(t, c2l) for t in texts]
    start, end = frames
    if any(e is None or len(e) > end - start for e in encoded) or end <= start:
        return None
    return [float(x) for x in _ctc_scores(_log_probs(line, n_classes, start, end), encoded)]


def arbitrate_regions(line: LineLogits, runs: list[tuple[str, int, int]], spans: list[tuple[int, int, int, int]],
                      words: list[str], alternative: str, known: frozenset[str], folded: FoldIndex | None,
                      lm: NgramLM, beta: float, c2l: dict[str, list[int]], n_classes: int,
                      params: RegionParams) -> str:
    """The rescored line, except where the alternative reading (beam search) disagrees on a short stretch,
    including stretches with a different number of words (inserted, split or merged words; Granell et al. 2015,
    Lund et al. 2011): there both versions are scored by CTC over the stretch's frames plus beta * LM with the
    neighbouring words, and the alternative is taken if every one of its words is in the lexicon and it wins by
    params.margin."""
    gaps = _gaps(runs, spans)
    other = _WORDS.findall(alternative)
    vocab: dict[str, int] = {}
    a = [vocab.setdefault(w.lower(), len(vocab)) for w in words]
    b = [vocab.setdefault(w.lower(), len(vocab)) for w in other]
    out_words, out_gaps = list(words), list(gaps)
    for op in Levenshtein.opcodes(a, b):
        i1, i2, mine, theirs = op.src_start, op.src_end, words[op.src_start:op.src_end], other[op.dest_start:op.dest_end]
        if op.tag == 'equal' or len(mine) > params.max_words or len(theirs) > params.max_words:
            continue
        if not all(is_known(w.lower(), known, folded) or w.lower() in known for w in theirs) or any(
                len(w) == 1 and fold(w.lower()) not in 'aeiou' for w in theirs):  # lone consonants: abbreviations
            continue
        if folded is not None:  # the lexicon's accented spelling (the beam matches words without accents)
            theirs = [_match_case(folded.surface.get(fold(w.lower()), w), w) for w in theirs]
        if i1 < i2:
            frames = (spans[i1][2], spans[i2 - 1][3])
            ours = ''.join(w + g for w, g in zip(mine, gaps[i1 + 1:i2])) + mine[-1]
            texts = [ours, ' '.join(theirs)]
        elif 0 < i1 < len(words) and not gaps[i1].strip():
            frames = (spans[i1 - 1][3], spans[i1][2])
            texts = [gaps[i1], f"{gaps[i1]}{' '.join(theirs)} "]
        else:
            continue
        scores = _region_score(line, c2l, n_classes, frames, texts)
        if scores is None:
            continue
        before, after = words[i1 - 1:i1], words[i2:i2 + 1]
        lm_ours = _context_lm(before + mine + after, lm) - _context_lm(before, lm)
        lm_theirs = _context_lm(before + theirs + after, lm) - _context_lm(before, lm)
        if scores[1] + beta * lm_theirs - scores[0] - beta * lm_ours <= params.margin:
            continue
        if i1 < i2:
            out_words[i1] = ' '.join(_match_case(w, mine[0]) if k == 0 else w for k, w in enumerate(theirs))
            for k in range(i1 + 1, i2):
                out_words[k], out_gaps[k] = '', ''
        else:
            out_gaps[i1] = texts[1]
    text = out_gaps[0] + ''.join(w + g for w, g in zip(out_words, out_gaps[1:]))
    return unicodedata.normalize('NFC', re.sub(r' {2,}', ' ', text))


def rescore_line_context(line: LineLogits, c2l: dict[str, list[int]], l2c: dict[int, str], n_classes: int,
                         known: frozenset[str], choices: tuple[str, ...], params: RescoreParams, lm: NgramLM,
                         beta: float, open_known: bool, continues_word: bool = False,
                         folded: FoldIndex | None = None, order: int = 2, short: tuple[str, ...] = (),
                         alternative: str | None = None, regions: RegionParams | None = None) -> str:
    """The whole line at once: candidates as in rescore_line (plus in-lexicon words when open_known, and short
    words when short is given), chosen jointly with an n-gram language model of the given order weighted by beta
    (context_lm.decode_line). continues_word: the previous line ends with a hyphen, so the first word is the tail
    of a split word and is left as read. alternative: another reading of the line (beam search); its lexicon
    words aligned one-to-one with read words join those words' candidates."""
    runs = _greedy_runs(line, l2c)
    spans = _letter_words(runs, line.top_idx.shape[0])
    read = [''.join(r[0] for r in runs[span[0]:span[1] + 1]) for span in spans]
    extras = _aligned_words(read, alternative, known) if alternative else [()] * len(spans)
    slots = [_word_slot(line, runs, span, c2l, n_classes, known, choices, params, open_known, folded, short,
                        l2c, extra) for span, extra in zip(spans, extras)]
    if continues_word and slots:
        slots[0] = slots[0][:1]
    if params.coherent:
        slots = [slot[:1] if _coherent(read, i, known, lm) else slot for i, slot in enumerate(slots)]
    words = decode_line(slots, lm, beta, order)
    if regions is not None and alternative is not None:
        return arbitrate_regions(line, runs, spans, words, alternative, known, folded, lm, beta, c2l, n_classes,
                                 regions)
    replacements = {span[0]: (span[1], word) for span, slot, word in zip(spans, slots, words) if word != slot[0][0]}
    return _assemble(runs, replacements)
