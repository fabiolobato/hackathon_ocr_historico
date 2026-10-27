"""Word bigram language model for choosing a whole line's words in context (built by tools/build_period_lexicon.py
from an external 19th-century corpus, never from this book).

    P(w2 | w1) = (1 - LAMBDA) * c(w1 w2) / c(w1) + LAMBDA * P_uni(w2)      (Jelinek-Mercer interpolation)
    P_uni(w)   = (c(w) + 1) / (N + V + 1); unseen words get 1 / (N + V + 1)

decode_line picks one candidate per word position maximizing
    sum_i score_i(c_i) + beta * sum_i log P(c_i | c_{i-1})
with Viterbi, so a word is judged by its neighbours on both sides (the next word's transition scores it too).
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

LAMBDA = 0.4


@dataclass(frozen=True)
class BigramLM:
    index: dict[str, int]
    unigrams: np.ndarray  # counts, aligned with index
    keys: np.ndarray  # sorted id1 * V + id2
    counts: np.ndarray
    total: int

    def unigram(self, word: str) -> float:
        i = self.index.get(word)
        count = int(self.unigrams[i]) if i is not None else 0
        return (count + 1) / (self.total + len(self.index) + 1)

    def logp(self, word: str, previous: str | None) -> float:
        p_uni = self.unigram(word)
        i, j = self.index.get(previous) if previous else None, self.index.get(word)
        if i is None or j is None:
            return math.log(LAMBDA * p_uni)
        key = i * len(self.index) + j
        pos = int(np.searchsorted(self.keys, key))
        pair = int(self.counts[pos]) if pos < len(self.keys) and self.keys[pos] == key else 0
        return math.log((1 - LAMBDA) * pair / int(self.unigrams[i]) + LAMBDA * p_uni)


@lru_cache(maxsize=2)
def load_lm(path: str) -> BigramLM:
    data = np.load(path)
    words = [str(w) for w in data['words']]
    unigrams = data['unigrams']
    return BigramLM({w: i for i, w in enumerate(words)}, unigrams, data['keys'], data['counts'], int(unigrams.sum()))


def decode_line(slots: list[list[tuple[str, float]]], lm: BigramLM, beta: float) -> list[str]:
    """slots: per word position, (candidate, image score) pairs; returns the best candidate per position."""
    if not slots:
        return []
    best = [(score + beta * lm.logp(word.lower(), None), [word]) for word, score in slots[0]]
    for slot in slots[1:]:
        best = [max(((total + score + beta * lm.logp(word.lower(), path[-1].lower()), path + [word])
                     for total, path in best), key=lambda item: item[0]) for word, score in slot]
    return max(best, key=lambda item: item[0])[1]
