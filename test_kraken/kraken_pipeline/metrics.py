import unicodedata
from dataclasses import dataclass

from .config import ScoringConfig


@dataclass(frozen=True)
class WordErrors:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.reference_words


def tokenize(text: str, config: ScoringConfig) -> list[str]:
    text = unicodedata.normalize('NFC', text)
    if config.lowercase:
        text = text.lower()
    if config.strip_punctuation:
        text = ''.join(c for c in text if not unicodedata.category(c).startswith('P'))
    return text.split()


def word_errors(reference: str, prediction: str,
                config: ScoringConfig = ScoringConfig()) -> WordErrors:
    ref, hyp = tokenize(reference, config), tokenize(prediction, config)
    if not ref:
        raise ValueError('WER needs a nonempty reference transcription.')
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, ref_word in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hyp, 1):
            if ref_word == hyp_word:
                current.append(previous[j - 1])
                continue
            cost, sub, delete, insert = previous[j - 1]
            substitution = (cost + 1, sub + 1, delete, insert)
            cost, sub, delete, insert = previous[j]
            deletion = (cost + 1, sub, delete + 1, insert)
            cost, sub, delete, insert = current[j - 1]
            insertion = (cost + 1, sub, delete, insert + 1)
            current.append(min((substitution, deletion, insertion), key=lambda x: x[0]))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return WordErrors(substitutions, deletions, insertions, len(ref))
