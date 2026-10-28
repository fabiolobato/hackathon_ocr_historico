"""Word n-gram language model for choosing a line's words in context (built by tools/build_period_lexicon.py
from an external 19th-century corpus, never from this book).

    P(w | v)    = (1 - LAMBDA) * c(v w) / c(v)     + LAMBDA * P_uni(w)       (Jelinek-Mercer interpolation)
    P(w | u v)  = (1 - LAMBDA) * c(u v w) / c(u v) + LAMBDA * P(w | v)       (order 3; falls back when c(u v) = 0)
    P_uni(w)    = (c(w) + 1) / (N + V + 1); unseen words get 1 / (N + V + 1)

decode_line picks one candidate per word position maximizing
    sum_i score_i(c_i) + beta * sum_i log P(c_i | history)
with Viterbi (states: the last order - 1 words), so a word is judged by its neighbours on both sides.
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

LAMBDA = 0.4


def _count(keys: np.ndarray, counts: np.ndarray, key: int) -> int:
    pos = int(np.searchsorted(keys, key))
    return int(counts[pos]) if pos < len(keys) and keys[pos] == key else 0


@dataclass(frozen=True)
class NgramLM:
    index: dict[str, int]
    unigrams: np.ndarray  # counts, aligned with index
    keys: np.ndarray  # sorted bigram keys id1 * V + id2
    counts: np.ndarray
    keys3: np.ndarray  # sorted trigram keys (id1 * V + id2) * V + id3
    counts3: np.ndarray
    total: int

    def unigram(self, word: str) -> float:
        i = self.index.get(word)
        count = int(self.unigrams[i]) if i is not None else 0
        return (count + 1) / (self.total + len(self.index) + 1)

    def bigram(self, word: str, previous: str | None) -> float:
        p_uni = self.unigram(word)
        i, j = self.index.get(previous) if previous else None, self.index.get(word)
        if i is None or j is None:
            return LAMBDA * p_uni
        pair = _count(self.keys, self.counts, i * len(self.index) + j)
        return (1 - LAMBDA) * pair / int(self.unigrams[i]) + LAMBDA * p_uni

    def seen(self, previous: str, word: str) -> bool:
        """The pair 'previous word' occurs in the corpus."""
        i, j = self.index.get(previous), self.index.get(word)
        return i is not None and j is not None and _count(self.keys, self.counts, i * len(self.index) + j) > 0

    def logp(self, word: str, previous: str | None, before: str | None = None) -> float:
        p_bi = self.bigram(word, previous)
        ids = [self.index.get(w) if w else None for w in (before, previous, word)]
        if None in ids:
            return math.log(p_bi)
        size = len(self.index)
        history = _count(self.keys, self.counts, ids[0] * size + ids[1])
        if not history:
            return math.log(p_bi)
        triple = _count(self.keys3, self.counts3, (ids[0] * size + ids[1]) * size + ids[2])
        return math.log((1 - LAMBDA) * triple / history + LAMBDA * p_bi)


@lru_cache(maxsize=2)
def load_lm(path: str) -> NgramLM:
    data = np.load(path)
    words = [str(w) for w in data['words']]
    unigrams = data['unigrams']
    return NgramLM({w: i for i, w in enumerate(words)}, unigrams, data['keys'], data['counts'], data['keys3'],
                   data['counts3'], int(unigrams.sum()))


def decode_line(slots: list[list[tuple[str, float]]], lm: NgramLM, beta: float, order: int = 2) -> list[str]:
    """slots: per word position, (candidate, image score) pairs; returns the best candidate per position."""
    if not slots:
        return []
    # state: the path's last order - 1 words (lowercased) -> (score, path)
    states: dict[tuple[str, ...], tuple[float, list[str]]] = {(): (0.0, [])}
    for slot in slots:
        nxt: dict[tuple[str, ...], tuple[float, list[str]]] = {}
        for history, (total, path) in states.items():
            previous = history[-1] if history else None
            before = history[-2] if len(history) > 1 and order > 2 else None
            for word, score in slot:
                lower = word.lower()
                value = total + score + beta * lm.logp(lower, previous, before)
                key = (history + (lower,))[-(order - 1):]
                if key not in nxt or value > nxt[key][0]:
                    nxt[key] = (value, path + [word])
        states = nxt
    return max(states.values(), key=lambda item: item[0])[1]
