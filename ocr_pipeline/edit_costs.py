"""Learned edit costs for candidate generation (Tong & Evans 1996 self-training; Pettersson et al. 2013 weights).

The recognizer's confusions are counted between its own reading of a word and the word the pipeline corrected it
to (no GT): aligned character units of up to two letters, so rn -> m or ss -> s count as one operation. An
operation seen E times costs 1 - weight * E / (E + SMOOTH); unseen or rare operations cost 1, as in plain
Levenshtein. Words are compared lowercase and without accents.
Pettersson et al.'s U / (E + U) (2013 §4.1.3, U = times the source unit was left unchanged) was tried first: our
confusions are spread over many operations, so every cost stayed above 0.8 and generation did not change.
"""
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz.distance import Levenshtein

MAX_UNIT = 2  # longest source or target unit of one operation
MIN_COST = 0.1  # no operation is free, however often it was seen
SMOOTH = 10.0  # an operation seen 10 times gets half of the weight


@dataclass
class EditCosts:
    rules: dict[str, dict[str, float]]  # source unit -> {target unit: cost}; '' = insertion source
    _memo: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)

    def distance(self, source: str, target: str) -> float:
        """Cheapest edit of source into target with unit costs 1 and the learned multi-character operations."""
        key = (source, target)
        if key not in self._memo:
            self._memo[key] = self._distance(source, target)
        return self._memo[key]

    def _distance(self, a: str, b: str) -> float:
        rows = [[float(j) for j in range(len(b) + 1)]] + [[float(i)] + [0.0] * len(b) for i in range(1, len(a) + 1)]
        for i in range(len(a) + 1):
            for j in range(len(b) + 1):
                if i == 0 and j == 0:
                    continue
                best = rows[i][j] if i == 0 or j == 0 else min(
                    rows[i - 1][j - 1] + (a[i - 1] != b[j - 1]), rows[i - 1][j] + 1, rows[i][j - 1] + 1)
                for ls in range(min(i, MAX_UNIT) + 1):
                    targets = self.rules.get(a[i - ls:i])
                    if not targets:
                        continue
                    for unit, cost in targets.items():
                        lt = len(unit)
                        if lt <= j and b[j - lt:j] == unit:
                            best = min(best, rows[i - ls][j - lt] + cost)
                rows[i][j] = best
        return rows[len(a)][len(b)]


def _units(source: str, target: str) -> list[tuple[str, str]]:
    """Non-matching aligned blocks (adjacent edits merged, so rn -> m is one block) of at most MAX_UNIT
    characters on each side; longer equal-length blocks count letter by letter, other long blocks not at all."""
    blocks: list[list[int]] = []
    for op in Levenshtein.opcodes(source, target):
        if op.tag == 'equal':
            continue
        if blocks and blocks[-1][1] == op.src_start and blocks[-1][3] == op.dest_start:
            blocks[-1][1], blocks[-1][3] = op.src_end, op.dest_end
        else:
            blocks.append([op.src_start, op.src_end, op.dest_start, op.dest_end])
    out = []
    for s0, s1, d0, d1 in blocks:
        src, dst = source[s0:s1], target[d0:d1]
        if len(src) <= MAX_UNIT and len(dst) <= MAX_UNIT:
            out.append((src, dst))
        elif len(src) == len(dst):
            out.extend(zip(src, dst))
    return out


def count_operations(pairs: list[tuple[str, str]]) -> tuple[Counter, Counter]:
    """(operation counts, source unit occurrence counts) over (read, corrected) word pairs."""
    ops: Counter = Counter()
    seen: Counter = Counter()
    for read, corrected in pairs:
        ops.update(_units(read, corrected))
        seen[''] += len(read) + 1
        for n in range(1, MAX_UNIT + 1):
            seen.update(read[k:k + n] for k in range(len(read) - n + 1))
    return ops, seen


def learn(pairs: list[tuple[str, str]], min_count: int) -> dict[str, dict[str, int]]:
    """Counts of the operations seen at least min_count times, as source unit -> {target unit: count}."""
    ops, _ = count_operations(pairs)
    rules: dict[str, dict[str, int]] = {}
    for (src, dst), n in ops.items():
        if n >= min_count:
            rules.setdefault(src, {})[dst] = n
    return rules


def scaled(rules: dict[str, dict[str, int]], weight: float) -> EditCosts:
    """weight 0 = plain Levenshtein; 1 = an operation seen E times costs 1 - E / (E + SMOOTH) (at least MIN_COST)."""
    return EditCosts({s: {d: max(MIN_COST, 1.0 - weight * n / (n + SMOOTH)) for d, n in t.items()}
                      for s, t in rules.items()})


def save(path: Path, rules: dict[str, dict[str, int]], meta: dict) -> None:
    path.write_text(json.dumps({'meta': meta, 'rules': rules}, ensure_ascii=False, indent=1), encoding='utf-8')


def load_rules(path: str) -> dict[str, dict[str, int]]:
    return json.loads(Path(path).read_text(encoding='utf-8'))['rules']
