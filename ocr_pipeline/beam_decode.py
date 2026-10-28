"""Line decoding by CTC prefix beam search with a lexicon and a word n-gram LM (Hannun et al. 2014 style).

Unlike rescoring (ctc_rescore), candidates are not lexicon neighbours of the greedy reading: every prefix the
posteriors allow is explored, so a badly read word can still come out as the right one, and spaces are decided
too (words wrongly split or merged). A hypothesis scores
    log P_ctc(labels) + beta * sum log P_lm(word | history) + bonus * #(lexicon words) + oov * #(other words)
Words are matched to the lexicon without accents (the recognizer often misses diacritics); the accents are put
back later by the edit pass. Posteriors are the stored top-k per frame, so this is an approximation.
"""
import math
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass

from .context_lm import NgramLM
from .ctc_rescore import LineLogits
from .postprocess import FoldIndex

NEG = -math.inf


@dataclass(frozen=True)
class BeamParams:
    beam: int = 16
    beta: float = 0.5  # weight of the LM log-probability
    oov: float = -6.0  # log penalty per word not in the lexicon (scored with the LM's unseen-word probability)
    bonus: float = 4.0  # log bonus per lexicon word; without it merging words into one unknown word is cheaper
    order: int = 2
    min_prob: float = 1e-3  # classes below this posterior are not expanded
    boundary_min: float = 0.0  # a space or punctuation mark is only expanded where the model gives it this much
    #                            posterior, so the LM cannot split a word the image shows as one ('e leitos')


@dataclass(frozen=True)
class Lexicon:
    keys: list[str]  # sorted accent-free lowercase words
    surface: dict[str, str]  # accent-free -> preferred spelling (the LM's word)

    def is_prefix(self, part: str) -> bool:
        i = bisect_left(self.keys, part)
        return i < len(self.keys) and self.keys[i].startswith(part)


def build_lexicon(folded: FoldIndex) -> Lexicon:
    return Lexicon(sorted(folded.keys), folded.surface)


def _logadd(a: float, b: float) -> float:
    if a == NEG:
        return b
    if b == NEG:
        return a
    hi = max(a, b)
    return hi + math.log1p(math.exp(-abs(a - b)))


def _is_letter(char: str) -> bool:
    return bool(char) and unicodedata.category(char)[0] in 'LM'


@dataclass
class _Hyp:
    p_blank: float
    p_label: float
    lm: float  # beta * LM + oov penalties of completed words
    history: tuple[str, ...]  # last order - 1 completed words (LM spelling)
    word: str  # accent-free lowercase letters of the word being read
    in_lexicon: bool  # word is still a prefix of a lexicon word
    free: bool = False  # the current word is the tail of a hyphenated word: no OOV penalty

    def total(self) -> float:
        return _logadd(self.p_blank, self.p_label) + self.lm


def _oov_cost(word: str, lm: NgramLM, params: BeamParams) -> float:
    return params.oov + params.beta * lm.logp(word, None)


def _close_word(hyp: _Hyp, lexicon: Lexicon, lm: NgramLM, params: BeamParams, free: bool) -> tuple[float, tuple]:
    """(lm score, history) after the current word ends; free: no OOV penalty (a hyphenated word part).
    An unknown word gets the LM's unseen-word probability plus the oov penalty, and resets the history."""
    if not hyp.word:
        return hyp.lm, hyp.history
    if not (hyp.in_lexicon and hyp.word in lexicon.surface):
        charged = not hyp.in_lexicon  # the penalty was paid when the word left the lexicon
        return hyp.lm + (0.0 if free or hyp.free or charged else _oov_cost(hyp.word, lm, params)), ()
    word = lexicon.surface[hyp.word]
    previous = hyp.history[-1] if hyp.history else None
    before = hyp.history[-2] if len(hyp.history) > 1 and params.order > 2 else None
    score = hyp.lm + params.bonus + params.beta * lm.logp(word, previous, before)
    return score, (hyp.history + (word,))[-(params.order - 1):]


def _extend(hyp: _Hyp, char: str, lexicon: Lexicon, lm: NgramLM, params: BeamParams) -> _Hyp:
    """The state after appending one character (its CTC probabilities are filled in by the caller)."""
    lower = char.lower()
    if _is_letter(char):
        if unicodedata.category(char) == 'Mn':  # combining accent: no effect on the accent-free word
            return _Hyp(NEG, NEG, hyp.lm, hyp.history, hyp.word, hyp.in_lexicon, hyp.free)
        word = hyp.word + unicodedata.normalize('NFD', lower)[0]
        in_lexicon = hyp.in_lexicon and lexicon.is_prefix(word)
        # charge an unknown word as soon as it leaves the lexicon, so that hypotheses which drop the following
        # spaces (merging words into one long unknown word) cannot postpone the penalty past beam pruning
        score = hyp.lm + (_oov_cost(word, lm, params) if hyp.in_lexicon and not in_lexicon and not hyp.free else 0.0)
        return _Hyp(NEG, NEG, score, hyp.history, word, in_lexicon, hyp.free)
    score, history = _close_word(hyp, lexicon, lm, params, free=char in '-=¬')
    return _Hyp(NEG, NEG, score, history, '', True)


def beam_line(line: LineLogits, l2c: dict[int, str], lexicon: Lexicon, lm: NgramLM, params: BeamParams,
              continues_word: bool = False) -> str:
    """Best reading of one line. continues_word: the line starts with the tail of a hyphenated word."""
    start = _Hyp(0.0, NEG, 0.0, (), '', True, continues_word)
    beams: dict[tuple[int, ...], _Hyp] = {(): start}
    for t in range(line.top_idx.shape[0]):
        frame = [(int(c), float(p)) for c, p in zip(line.top_idx[t], line.top_val[t])
                 if float(p) >= max(params.min_prob, 0.0 if int(c) == 0 or _is_letter(l2c.get(int(c), ''))
                                    else params.boundary_min)]
        nxt: dict[tuple[int, ...], _Hyp] = {}
        for prefix, hyp in beams.items():
            for label, p in frame:
                logp = math.log(p)
                if label == 0:
                    target = nxt.setdefault(prefix, _Hyp(NEG, NEG, hyp.lm, hyp.history, hyp.word, hyp.in_lexicon,
                                                          hyp.free))
                    target.p_blank = _logadd(target.p_blank, _logadd(hyp.p_blank, hyp.p_label) + logp)
                    continue
                char = l2c.get(label, '')
                if prefix and label == prefix[-1]:  # repeat: stays the same prefix unless a blank separated them
                    same = nxt.setdefault(prefix, _Hyp(NEG, NEG, hyp.lm, hyp.history, hyp.word, hyp.in_lexicon,
                                                        hyp.free))
                    same.p_label = _logadd(same.p_label, hyp.p_label + logp)
                    source = hyp.p_blank
                else:
                    source = _logadd(hyp.p_blank, hyp.p_label)
                if source == NEG:
                    continue
                new = prefix + (label,)
                if new not in nxt:
                    nxt[new] = _extend(hyp, char, lexicon, lm, params)
                nxt[new].p_label = _logadd(nxt[new].p_label, source + logp)
        beams = dict(sorted(nxt.items(), key=lambda item: -item[1].total())[:params.beam])
    best = max(beams.items(), key=lambda item: _close_word(item[1], lexicon, lm, params, False)[0]
               + _logadd(item[1].p_blank, item[1].p_label))
    return unicodedata.normalize('NFC', ''.join(l2c.get(label, '') for label in best[0]))
