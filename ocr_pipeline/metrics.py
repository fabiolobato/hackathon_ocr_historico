"""WER/CER under two scoring policies.

strict: NFC, case and punctuation preserved (the policy of test_kraken/kraken_pipeline/metrics.py).
norm:   NFC, lowercase, punctuation stripped (the policy of LLM_Transcription.ipynb).
Corpus scores sum edits and reference lengths over pages; they are not page means.
"""
import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str, strict: bool) -> str:
    text = unicodedata.normalize('NFC', text)
    if strict:
        return ' '.join(text.split())
    return ' '.join(_PUNCT.sub(' ', text.lower()).split())


def _word_ids(ref: list[str], hyp: list[str]) -> tuple[list[int], list[int]]:
    vocab: dict[str, int] = {}
    return ([vocab.setdefault(w, len(vocab)) for w in ref], [vocab.setdefault(w, len(vocab)) for w in hyp])


@dataclass(frozen=True)
class Counts:
    word_edits: int
    ref_words: int
    char_edits: int
    ref_chars: int

    def __add__(self, other: 'Counts') -> 'Counts':
        return Counts(*(a + b for a, b in zip(self.astuple(), other.astuple())))

    def astuple(self) -> tuple[int, int, int, int]:
        return self.word_edits, self.ref_words, self.char_edits, self.ref_chars

    @property
    def wer(self) -> float:
        return self.word_edits / self.ref_words

    @property
    def cer(self) -> float:
        return self.char_edits / self.ref_chars


def count(reference: str, hypothesis: str, strict: bool) -> Counts:
    ref, hyp = normalize(reference, strict), normalize(hypothesis, strict)
    ref_ids, hyp_ids = _word_ids(ref.split(), hyp.split())
    return Counts(Levenshtein.distance(ref_ids, hyp_ids), len(ref_ids), Levenshtein.distance(ref, hyp), len(ref))


def score(reference: str, hypothesis: str) -> dict[str, float | int]:
    strict, norm = count(reference, hypothesis, True), count(reference, hypothesis, False)
    return {'wer_strict': strict.wer, 'cer_strict': strict.cer, 'wer_norm': norm.wer, 'cer_norm': norm.cer,
            **{f'strict_{k}': v for k, v in zip(('we', 'rw', 'ce', 'rc'), strict.astuple())},
            **{f'norm_{k}': v for k, v in zip(('we', 'rw', 'ce', 'rc'), norm.astuple())}}


def corpus(rows: list[dict[str, float | int]]) -> dict[str, float]:
    out = {}
    for policy in ('strict', 'norm'):
        total = sum((Counts(*(int(r[f'{policy}_{k}']) for k in ('we', 'rw', 'ce', 'rc'))) for r in rows),
                    Counts(0, 0, 0, 0))
        out[f'wer_{policy}'], out[f'cer_{policy}'] = total.wer, total.cer
    return out
