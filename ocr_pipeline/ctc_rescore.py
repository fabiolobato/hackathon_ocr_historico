"""Lexicon rescoring with the recognizer's own CTC posteriors.

For every out-of-lexicon letter run in the greedy output, candidate words close in edit distance are
scored by their CTC log-likelihood over the frames between the run's neighbouring characters:
    score(c) = log P_ctc(c | frames) + alpha * [c in lexicon]
The greedy word competes with alpha = 0. Posteriors are stored as the top-k classes per frame
(remaining mass floored), which keeps caches small and makes the loss an approximation.
"""
import unicodedata
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .context_lm import BigramLM, decode_line
from .postprocess import FoldIndex, fold, is_known, nearest

FLOOR = 1e-6


@dataclass(frozen=True)
class LineLogits:
    prediction: str
    top_idx: np.ndarray  # (T, k) int16 class ids
    top_val: np.ndarray  # (T, k) float16 probabilities


def empty_line(k: int = 20) -> LineLogits:
    return LineLogits('', np.zeros((0, k), np.int16), np.zeros((0, k), np.float16))


def compress(probs: torch.Tensor, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    values, indices = probs.float().topk(k, dim=0)  # probs: (C, T)
    return indices.T.numpy().astype(np.int16), values.T.numpy().astype(np.float16)


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


def _word_slot(line: LineLogits, runs: list[tuple[str, int, int]], span: tuple[int, int, int, int],
               c2l: dict[str, list[int]], n_classes: int, known: frozenset[str], choices: tuple[str, ...],
               params: RescoreParams, open_known: bool,
               folded: FoldIndex | None = None) -> list[tuple[str, float]]:
    """(candidate, CTC log-likelihood + alpha if in lexicon) for one word; the read word first.
    A single entry means the word is left as read (too short, in the lexicon and not opened, or no candidate).
    With folded, candidates are found ignoring accents and each one scores as the better of its accented and
    accent-free spellings, so não can win over a read 'nao'/'nar' when the image shows no tilde."""
    first, last, start, end = span
    original = unicodedata.normalize('NFC', ''.join(r[0] for r in runs[first:last + 1]))
    lower = original.lower()
    if len(original) < params.min_len or end <= start or (is_known(lower, known, folded) and not open_known):
        return [(original, 0.0)]
    cutoff = max(params.min_edits, int(len(original) * params.max_ratio))
    found = nearest(lower, choices, folded, cutoff, max(params.candidates, params.pool) + 1)
    cands = [_match_case(c, original) for c in found if c != lower][:max(params.candidates, params.pool)]
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


def rescore_line_context(line: LineLogits, c2l: dict[str, list[int]], l2c: dict[int, str], n_classes: int,
                         known: frozenset[str], choices: tuple[str, ...], params: RescoreParams, lm: BigramLM,
                         beta: float, open_known: bool, continues_word: bool = False,
                         folded: FoldIndex | None = None) -> str:
    """The whole line at once: candidates as in rescore_line (plus in-lexicon words when open_known), chosen
    jointly with a bigram language model weighted by beta (context_lm.decode_line). continues_word: the previous
    line ends with a hyphen, so the first word is the tail of a split word and is left as read."""
    runs = _greedy_runs(line, l2c)
    spans = _letter_words(runs, line.top_idx.shape[0])
    slots = [_word_slot(line, runs, span, c2l, n_classes, known, choices, params, open_known, folded)
             for span in spans]
    if continues_word and slots:
        slots[0] = slots[0][:1]
    words = decode_line(slots, lm, beta)
    replacements = {span[0]: (span[1], word) for span, slot, word in zip(spans, slots, words) if word != slot[0][0]}
    return _assemble(runs, replacements)
