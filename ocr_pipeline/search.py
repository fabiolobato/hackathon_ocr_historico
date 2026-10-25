"""Conditional greedy search: every pipeline choice is kept only if it beats the choices already made.

Spec (JSON):
  {"base": {<dotted overrides>}, "metric": "wer_norm", "min_gain": 0.0,
   "stages": [{"name": "seg_image", "candidates": {"nlbin": {"seg_ops": "kraken_nlbin"},
                                                    "t04": {"set": {...}, "requires": {"seg_ops": "kraken_nlbin"}}}}]}
Per stage, the incumbent is every current choice except this stage's own:
  1. each candidate alone; the stage moves only if the best one beats the incumbent by more than min_gain.
  2. forward selection: start from that best one, try every other candidate on top of it (including ones
     that hurt alone), keep the one that helps most, repeat while the combination keeps improving. Tuple fields (ops chains) concatenate in declaration order;
     candidates setting the same scalar to different values cannot be combined.
  3. the stage's picks change only if the new winner beats the current configuration. No gain, no change.
After all stages, a backward pass re-tests every accepted choice in the final context and drops the one whose
removal hurts least, while removal costs no more than min_gain. Passes repeat (each stage re-decided in the
context of all later choices, e.g. image ops once the lexicon is on) until the choice set is stable.
requires: {key: value} is satisfied when the current value equals it, contains it (tuple fields), or is truthy
for "*"; a list of such dicts means any of them.

  python -m ocr_pipeline.search sweeps/search_v1.json --pages tuning
"""
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig, stable_hash, with_overrides
from .data import Page, discover, select, split
from .engine import Engine
from .pipeline import evaluate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    stage: str
    name: str
    order: int
    overrides: dict[str, Any]
    requires: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SearchSpec:
    base: dict[str, Any]
    metric: str
    min_gain: float
    stages: tuple[tuple[str, tuple[Candidate, ...]], ...]


def parse_spec(payload: dict[str, Any]) -> SearchSpec:
    stages = []
    for stage in payload['stages']:
        candidates = []
        for order, (name, body) in enumerate(stage['candidates'].items()):
            overrides = body['set'] if 'set' in body else body
            requires = body.get('requires', ()) if 'set' in body else ()
            requires = (requires,) if isinstance(requires, dict) else tuple(requires)
            candidates.append(Candidate(stage['name'], name, order, overrides, requires))
        stages.append((stage['name'], tuple(candidates)))
    return SearchSpec(payload.get('base', {}), payload.get('metric', 'wer_norm'), payload.get('min_gain', 0.0),
                      tuple(stages))


def field_value(config: PipelineConfig, key: str) -> Any:
    head, _, tail = key.partition('.')
    value = getattr(config, head)
    return getattr(value, tail) if tail else value


def _as_ops(value: Any) -> list[str]:
    return [v for v in value.split(',') if v] if isinstance(value, str) else list(value)


def merge(config: PipelineConfig, candidates: list[Candidate]) -> dict[str, Any] | None:
    """Overrides of several candidates of one stage; None if two set one scalar differently."""
    out: dict[str, Any] = {}
    for candidate in sorted(candidates, key=lambda c: c.order):
        for key, value in candidate.overrides.items():
            if key not in out:
                out[key] = value
            elif isinstance(field_value(config, key), tuple):
                out[key] = ','.join(_as_ops(out[key]) + _as_ops(value))
            elif out[key] != value:
                return None
    return out


def satisfied(config: PipelineConfig, requires: tuple[dict[str, Any], ...]) -> bool:
    def holds(key: str, expected: Any) -> bool:
        value = field_value(config, key)
        if expected == '*':
            return bool(value)
        return expected in value if isinstance(value, tuple) else value == expected
    return not requires or any(all(holds(k, v) for k, v in clause.items()) for clause in requires)


class Search:
    def __init__(self, engine: Engine, pages: list[Page], pool: list[Page], spec: SearchSpec, base: PipelineConfig,
                 log_path: Path) -> None:
        self.engine, self.pages, self.pool, self.spec, self.base = engine, pages, pool, spec, base
        self.log_path = log_path
        self.memo: dict[str, dict[str, Any]] = {}
        self.pass_no = 0
        self.trail: list[dict[str, Any]] = []  # one entry per (pass, stage): metric before/after and picks
        self.records: list[dict[str, Any]] = []
        if log_path.is_file():  # resume: configurations already evaluated by an interrupted run
            for line in log_path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                self.memo[row['key']] = {'corpus': row['corpus'], 'pages': row['pages']}

    def build(self, chosen: list[Candidate]) -> PipelineConfig | None:
        config = self.base
        for stage, _ in self.spec.stages:
            picked = [c for c in chosen if c.stage == stage]
            if picked:
                overrides = merge(config, picked)
                if overrides is None:
                    return None
                config = with_overrides(config, overrides)
        return config

    def score(self, config: PipelineConfig) -> float:
        key = stable_hash(config.to_dict())
        if key not in self.memo:
            self.memo[key] = evaluate(self.engine, self.pages, config, self.pool)
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'key': key, 'config': config.to_dict(), **self.memo[key]},
                                        ensure_ascii=False) + '\n')
        return self.memo[key]['corpus'][self.spec.metric]

    def record(self, stage: str, kind: str, names: list[str], config: PipelineConfig, ref: float) -> float:
        value = self.score(config)
        self.records.append({'pass': self.pass_no, 'stage': stage, 'kind': kind, 'choices': names, 'metric': value, 'delta': value - ref,
                             'accepted': False, 'config': config.to_dict()})
        logger.info('[%s] %-8s %-40s %s %.2f (Δ %+.2f)', stage, kind, '+'.join(names), self.spec.metric,
                    100 * value, 100 * (value - ref))
        return value

    def singles(self, stage: str, candidates: tuple[Candidate, ...], chosen: list[Candidate],
                ref: float) -> list[tuple[float, Candidate]]:
        """Every applicable candidate alone on top of the incumbent, best first."""
        incumbent = self.build(chosen)
        tested = []
        for candidate in candidates:
            config = self.build(chosen + [candidate])
            if not satisfied(incumbent, candidate.requires) or config is None or config == incumbent:
                logger.info('[%s] skip %s (requirement unmet or no-op)', stage, candidate.name)
                continue
            tested.append((self.record(stage, 'single', [candidate.name], config, ref), candidate))
        return sorted(tested, key=lambda item: (item[0], item[1].order))

    def combine(self, stage: str, tested: list[tuple[float, Candidate]], kept: list[Candidate],
                ref: float) -> tuple[float, list[Candidate]]:
        """Forward selection: grow the best single with whichever other candidate helps most on top of it.
        Every candidate is tried on top, including ones that hurt alone; one is kept only if it helps."""
        best, picked = tested[0][0], [tested[0][1]]
        rest = [c for _, c in tested[1:]]
        while rest:
            trials = [(self.record(stage, 'combo', [c.name for c in picked + [x]], cfg, ref), x)
                      for x in rest if (cfg := self.build(kept + picked + [x])) is not None]
            if not trials or min(t[0] for t in trials) >= best - self.spec.min_gain:
                break
            best, winner = min(trials, key=lambda t: t[0])
            picked.append(winner)
            rest.remove(winner)
        return best, picked

    def run_stage(self, stage: str, candidates: tuple[Candidate, ...], chosen: list[Candidate]) -> list[Candidate]:
        """Re-decide one stage from scratch in the context of every other current choice."""
        kept = [c for c in chosen if c.stage != stage]
        current = self.score(self.build(chosen))
        ref = self.record(stage, 'incumbent', [], self.build(kept), self.score(self.build(kept)))
        tested = self.singles(stage, candidates, kept, ref)
        result = kept if ref <= current - self.spec.min_gain or kept == chosen else chosen
        if tested and tested[0][0] < ref - self.spec.min_gain:
            best, picked = self.combine(stage, tested, kept, ref)
            if best < current - self.spec.min_gain or kept == chosen:
                self.mark(stage, [c.name for c in picked])
                result = kept + picked
        picks = [c.name for c in result if c.stage == stage]
        self.trail.append({'pass': self.pass_no, 'stage': stage, 'before': current,
                           'after': self.score(self.build(result)), 'picks': picks})
        logger.info('[%s] picks: %s', stage, picks or 'none')
        return result

    def mark(self, stage: str, names: list[str]) -> None:
        for record in reversed(self.records):
            if record['stage'] == stage and record['choices'] == names and record['pass'] == self.pass_no:
                record['accepted'] = True
                return

    def backward(self, chosen: list[Candidate]) -> list[Candidate]:
        while chosen:
            ref = self.score(self.build(chosen))
            trials = [(self.record('backward', 'backward', [f'-{c.stage}:{c.name}'], cfg, ref), c)
                      for c in chosen if (cfg := self.build([x for x in chosen if x is not c])) is not None]
            value, weakest = min(trials, key=lambda t: t[0])
            if value > ref + self.spec.min_gain:
                break
            logger.info('[backward] dropping %s:%s (removal Δ %+.2f)', weakest.stage, weakest.name,
                        100 * (value - ref))
            self.mark('backward', [f'-{weakest.stage}:{weakest.name}'])
            chosen = [c for c in chosen if c is not weakest]
        return chosen

    def run(self, passes: int) -> list[Candidate]:
        chosen: list[Candidate] = []
        for self.pass_no in range(1, passes + 1):
            before = {(c.stage, c.name) for c in chosen}
            for stage, candidates in self.spec.stages:
                chosen = self.run_stage(stage, candidates, chosen)
            chosen = self.backward(chosen)
            logger.info('pass %d: %s -> %s %.2f', self.pass_no, [f'{c.stage}:{c.name}' for c in chosen],
                        self.spec.metric, 100 * self.score(self.build(chosen)))
            if {(c.stage, c.name) for c in chosen} == before:
                break
        return chosen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('spec', type=Path)
    parser.add_argument('--pages', default='tuning')
    parser.add_argument('--probe', type=int, default=8, help='page shown in the report for every candidate')
    parser.add_argument('--images', type=Path, default=Path('imagens'))
    parser.add_argument('--references', type=Path, default=Path('test_kraken/ground_truth'))
    parser.add_argument('--out', type=Path, default=Path('outputs/search'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--passes', type=int, default=3, help='max forward+backward passes; stops when stable')
    parser.add_argument('--no-report', action='store_true')
    return parser


def write_outputs(out: Path, search: Search, final: list[Candidate]) -> None:
    final_config = search.build(final)
    decisions = {'final': [f'{c.stage}:{c.name}' for c in final], 'metric': search.spec.metric,
                 'final_value': search.score(final_config), 'passes': search.pass_no, 'trail': search.trail,
                 'records': search.records}
    (out / 'decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'best_config.json').write_text(json.dumps(final_config.to_dict(), ensure_ascii=False, indent=2),
                                          encoding='utf-8')


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for noisy in ('kraken', 'lightning', 'lightning.pytorch', 'shapely'):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    spec = parse_spec(json.loads(args.spec.read_text(encoding='utf-8')))
    out = args.out / f'{args.spec.stem}__{args.pages}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run_config.json').write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()},
                                                     'spec': json.loads(args.spec.read_text(encoding='utf-8'))},
                                                    indent=2), encoding='utf-8')
    all_pages = discover(args.images, args.references)
    pool = split(all_pages)[0]
    engine = Engine(args.cache, args.device, args.seed)
    search = Search(engine, select(all_pages, args.pages), pool, spec, with_overrides(PipelineConfig(), spec.base),
                    out / 'evaluations.jsonl')
    final = search.run(args.passes)
    write_outputs(out, search, final)
    logger.info('final:   %s -> %s %.2f', [f'{c.stage}:{c.name}' for c in final], spec.metric,
                100 * search.score(search.build(final)))
    if not args.no_report:
        from .search_report import write_report

        write_report(out, search, final, engine, select(all_pages, str(args.probe))[0], pool)


if __name__ == '__main__':
    main()
