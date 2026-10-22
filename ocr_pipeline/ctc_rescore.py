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

FLOOR = 1e-6


@dataclass(frozen=True)
class LineLogits:
    prediction: str
    top_idx: np.ndarray  # (T, k) int16 class ids
    top_val: np.ndarray  # (T, k) float16 probabilities


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


def rescore_line(line: LineLogits, c2l: dict[str, list[int]], l2c: dict[int, str], n_classes: int,
                 known: frozenset[str], choices: tuple[str, ...], params: RescoreParams) -> str:
    runs = _greedy_runs(line, l2c)
    replacements: dict[int, tuple[int, str]] = {}
    for first, last, start, end in _letter_words(runs, line.top_idx.shape[0]):
        original = unicodedata.normalize('NFC', ''.join(r[0] for r in runs[first:last + 1]))
        lower = original.lower()
        if len(original) < params.min_len or lower in known or end <= start:
            continue
        cutoff = max(1, int(len(original) * params.max_ratio))
        found = process.extract(lower, choices, scorer=Levenshtein.distance, score_cutoff=cutoff,
                                limit=params.candidates)
        cands = [_match_case(c, original) for c, _, _ in found]
        encoded = [(c, _encode(c, c2l)) for c in [original] + cands]
        encoded = [(c, e) for c, e in encoded if e and len(e) <= end - start]
        if len(encoded) < 2:
            continue
        scores = _ctc_scores(_log_probs(line, n_classes, start, end), [e for _, e in encoded])
        bonus = torch.tensor([0.0] + [params.alpha] * (len(encoded) - 1))
        best = int(torch.argmax(scores + bonus))
        if best:
            replacements[first] = (last, encoded[best][0])
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
