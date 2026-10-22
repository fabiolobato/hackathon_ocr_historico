"""Text post-processing applied to recognized lines. Lexicons must never be built from evaluation GT."""
import re
from collections import Counter
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


def map_conventions(line: str) -> str:
    """Map PP-OCR transcription conventions (n^o, D^r, ⁋) to the GT's (nº, Dr, §)."""
    line = _SUPERSCRIPT.sub(lambda m: _ORDINALS.get(m[1], m[1]), line)
    return line.replace('⁋', '§')


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


def _correct_word(word: str, config: PostConfig, known: frozenset[str], choices: tuple[str, ...]) -> str:
    match = _TOKEN.match(word)
    if not match:
        return word
    pre, core, post = match.groups()
    lower = core.lower()
    if len(core) < config.lexicon_min_len or lower in known or not core.isalpha():
        return word
    cutoff = max(1, int(len(core) * config.lexicon_max_ratio))
    best = process.extractOne(lower, choices, scorer=Levenshtein.distance, score_cutoff=cutoff)
    if best is None:
        return word
    fixed = best[0]
    if core[0].isupper():
        fixed = fixed[0].upper() + fixed[1:]
    return pre + fixed + post


def lexicon_choices(config: PostConfig, domain: tuple[str, ...]) -> tuple[str, ...]:
    """Domain words first, then generic words; each block ordered by frequency."""
    generic = load_lexicon(config.lexicon, config.lexicon_size) if config.lexicon else ()
    return tuple(dict.fromkeys(domain + generic))


def correct(lines: list[str], config: PostConfig, domain: tuple[str, ...] = ()) -> list[str]:
    if config.conventions:
        lines = [map_conventions(line) for line in lines]
    if config.dehyphenate:
        lines = dehyphenate(lines)
    choices = lexicon_choices(config, domain) if config.decoder in ('edit', 'ctc+edit') else ()
    if choices:
        known = frozenset(choices)
        lines = [' '.join(_correct_word(w, config, known, choices) for w in line.split()) for line in lines]
    return lines
