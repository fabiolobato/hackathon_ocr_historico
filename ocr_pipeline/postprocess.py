"""Text post-processing applied to recognized lines. Lexicons must never be built from evaluation GT."""
import re
from functools import lru_cache
from pathlib import Path

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .config import PostConfig

_HYPHEN_END = re.compile(r'[-=¬]\s*$')
_TOKEN = re.compile(r'^(\W*)(\w+)(\W*)$')


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


@lru_cache(maxsize=4)
def load_lexicon(path: str) -> tuple[frozenset[str], tuple[str, ...]]:
    words = [w.strip() for w in Path(path).read_text(encoding='utf-8').splitlines() if w.strip()]
    lowered = tuple(dict.fromkeys(w.lower() for w in words))
    return frozenset(lowered), lowered


def _correct_word(word: str, config: PostConfig, known: frozenset[str], choices: tuple[str, ...]) -> str:
    match = _TOKEN.match(word)
    if not match:
        return word
    pre, core, post = match.groups()
    lower = core.lower()
    if len(core) < config.lexicon_min_len or lower in known or core.isdigit():
        return word
    cutoff = max(1, int(len(core) * config.lexicon_max_ratio))
    best = process.extractOne(lower, choices, scorer=Levenshtein.distance, score_cutoff=cutoff)
    if best is None:
        return word
    fixed = best[0]
    if core[0].isupper():
        fixed = fixed[0].upper() + fixed[1:]
    return pre + fixed + post


def correct(lines: list[str], config: PostConfig) -> list[str]:
    if config.dehyphenate:
        lines = dehyphenate(lines)
    if config.lexicon:
        known, choices = load_lexicon(config.lexicon)
        lines = [' '.join(_correct_word(w, config, known, choices) for w in line.split()) for line in lines]
    return lines
