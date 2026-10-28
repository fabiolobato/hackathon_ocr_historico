"""Text post-processing applied to recognized lines. Lexicons must never be built from evaluation GT."""
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .config import PostConfig

_HYPHEN_END = re.compile(r'[-=¬]\s*$')
_TOKEN = re.compile(r'^(\W*)(\w+)(\W*)$')
_WORD = re.compile(r'[^\W\d_]+')


_SUPERSCRIPT = re.compile(r'\^(\w)')
_ORDINALS = {'o': 'º', 'a': 'ª'}
_DIGIT_ORDINAL = re.compile(r'\b(\d+)([oa])\b')


def map_conventions(line: str) -> str:
    """Map PP-OCR transcription conventions (n^o, D^r, ⁋) to the GT's (nº, Dr, §)."""
    line = _SUPERSCRIPT.sub(lambda m: _ORDINALS.get(m[1], m[1]), line)
    return line.replace('⁋', '§')


def map_ordinals(line: str) -> str:
    """A raised o/a after a number is read as a plain letter: 2o -> 2º, 1a -> 1ª."""
    return _DIGIT_ORDINAL.sub(lambda m: m[1] + _ORDINALS[m[2]], line)


def dehyphenate(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if out and _HYPHEN_END.search(out[-1]) and line.strip():
            head, first, *rest = [_HYPHEN_END.sub('', out[-1])] + line.strip().split(' ', 1)
            out[-1] = head + first
            if rest:
                out.append(rest[0])
        else:
            out.append(line)
    return out


@lru_cache(maxsize=8)
def load_lexicon(path: str, size: int) -> tuple[str, ...]:
    # One word per line, most frequent first; trailing columns (e.g. counts) are ignored.
    words = [w.split()[0].lower() for w in Path(path).read_text(encoding='utf-8').splitlines() if w.strip()]
    words = tuple(dict.fromkeys(w for w in words if w.isalpha()))
    return words[:size] if size > 0 else words


def domain_vocabulary(texts: list[str]) -> tuple[str, ...]:
    """Words of in-domain transcriptions, most frequent first. Callers must exclude the evaluated page."""
    counts = Counter(w for text in texts for w in _WORD.findall(text.lower()))
    return tuple(w for w, _ in counts.most_common())


def fold(word: str) -> str:
    """Drop accents and cedillas (não -> nao, ç -> c): the recognizer often misses faint diacritics."""
    return unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFD', word)
                                                if unicodedata.category(c) != 'Mn'))


@dataclass(frozen=True)
class FoldIndex:
    keys: tuple[str, ...]  # accent-free forms, in lexicon order
    surface: dict[str, str]  # accent-free form -> its first (most frequent) spelling in the lexicon


def fold_index(choices: tuple[str, ...]) -> FoldIndex:
    surface: dict[str, str] = {}
    for word in choices:
        surface.setdefault(fold(word), word)
    return FoldIndex(tuple(surface), surface)


def is_known(word: str, known: frozenset[str], folded: FoldIndex | None) -> bool:
    """In the lexicon; with folded, also the preferred spelling of its accent-free form (nao -> não is fixed)."""
    return word in known and (folded is None or folded.surface.get(fold(word)) == word)


def nearest(word: str, choices: tuple[str, ...], folded: FoldIndex | None, cutoff: int, limit: int) -> list[str]:
    """Lexicon words within cutoff edits, closest first; with folded, distances ignore accents and the
    accented spelling is returned."""
    if folded is None:
        return [c for c, _, _ in process.extract(word, choices, scorer=Levenshtein.distance, score_cutoff=cutoff,
                                                 limit=limit)]
    return [folded.surface[k] for k, _, _ in process.extract(fold(word), folded.keys, scorer=Levenshtein.distance,
                                                             score_cutoff=cutoff, limit=limit)]


def short_words(choices: tuple[str, ...], min_len: int, size: int) -> tuple[str, ...]:
    """The size most frequent lexicon words shorter than min_len; lone consonants (abbreviations) are dropped."""
    return tuple(w for w in choices if len(w) < min_len and (len(w) > 1 or fold(w) in 'aeiou'))[:size]


def _correct_word(word: str, config: PostConfig, known: frozenset[str], choices: tuple[str, ...],
                  folded: FoldIndex | None = None) -> str:
    match = _TOKEN.match(word)
    if not match:
        return word
    pre, core, post = match.groups()
    lower = core.lower()
    if len(core) < config.lexicon_min_len or is_known(lower, known, folded) or not core.isalpha():
        return word
    cutoff = max(1, int(len(core) * config.lexicon_max_ratio))
    if folded is None:
        best = process.extractOne(lower, choices, scorer=Levenshtein.distance, score_cutoff=cutoff)
        if best is None:
            return word
        fixed = best[0]
    else:
        found = nearest(lower, choices, folded, cutoff, 1)
        if not found:
            return word
        fixed = found[0]
    if core[0].isupper():
        fixed = fixed[0].upper() + fixed[1:]
    return pre + fixed + post


def lexicon_choices(config: PostConfig, domain: tuple[str, ...]) -> tuple[str, ...]:
    """Domain words first, then period words, then generic words; each block ordered by frequency."""
    period = load_lexicon(config.period_lexicon, config.period_lexicon_size) if config.period_lexicon else ()
    generic = load_lexicon(config.lexicon, config.lexicon_size) if config.lexicon else ()
    return tuple(dict.fromkeys(domain + period + generic))


def correct(lines: list[str], config: PostConfig, domain: tuple[str, ...] = ()) -> list[str]:
    if config.conventions:
        lines = [map_conventions(line) for line in lines]
    if config.ordinals:
        lines = [map_ordinals(line) for line in lines]
    if config.dehyphenate:
        lines = dehyphenate(lines)
    choices = lexicon_choices(config, domain) if config.decoder in ('edit', 'ctc+edit') else ()
    if choices:
        known = frozenset(choices)
        folded = fold_index(choices) if config.fold_accents else None
        lines = [' '.join(_correct_word(w, config, known, choices, folded) for w in line.split()) for line in lines]
    return lines
