"""Random search over the same space as the conditional greedy search (a search spec's stages and candidates).

One configuration is drawn stage by stage. Inside a stage, candidates that set the same scalar field (gamma 0.7 vs
1.5, alpha 2 vs 8, ...) are mutually exclusive: one of {none, c1, c2, ...} is drawn uniformly. Candidates that only
extend an op chain form their own group, i.e. are switched on with probability 1/2. A candidate whose requires are
unmet in the drawn configuration is dropped. No draw depends on any score, so a choice rejected once can always
come back.

Image stages are expensive (segmentation + recognition, ~30 s per page on CPU) and text stages are cheap once the
recognizer output is cached, so the budget is --n-image image draws x --n-text text draws on each: every pair is
one full random configuration. Image draw i and its text draws are seeded by (seed, i), so --shard k/n splits the
image draws across processes without changing them.

  python -m ocr_pipeline.tuning.random_search sweeps/search_v2.json --n-image 60 --n-text 20 --shard 0/2
  python -m ocr_pipeline.tuning.random_search sweeps/search_v2.json --summarize --final-pages heldout
"""
import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import PipelineConfig, from_dict, load_config, stable_hash, with_overrides
from ..data import Page, discover, select
from ..engine import Engine
from ..pipeline import evaluate, trace_page
from .search import Candidate, SearchSpec, build, field_value, parse_spec, satisfied

logger = logging.getLogger(__name__)

IMAGE_STAGES = frozenset({'recognizer', 'seg_image', 'seg_params', 'geometry', 'line_filter', 'rec_image',
                          'rec_params'})


def scalar_keys(base: PipelineConfig, candidate: Candidate) -> set[str]:
    return {k for k in candidate.overrides if not isinstance(field_value(base, k), tuple)}


def exclusive_groups(base: PipelineConfig, candidates: tuple[Candidate, ...]) -> list[list[Candidate]]:
    """Connected components of 'sets the same scalar field'; a pure op-chain candidate is its own group."""
    groups: list[list[Candidate]] = []
    for candidate in candidates:
        keys = scalar_keys(base, candidate)
        touching = [g for g in groups if keys & set().union(*(scalar_keys(base, c) for c in g))]
        merged = [candidate] + [c for g in touching for c in g]
        groups = [g for g in groups if g not in touching] + [sorted(merged, key=lambda c: c.order)]
    return groups


def draw_stages(rng: random.Random, base: PipelineConfig, spec: SearchSpec, stages: set[str]) -> list[Candidate]:
    picked = []
    for stage, candidates in spec.stages:
        if stage in stages:
            for group in exclusive_groups(base, candidates):
                choice = rng.choice([None, *group])
                if choice is not None:
                    picked.append(choice)
    return picked


def resolve(base: PipelineConfig, spec: SearchSpec, picked: list[Candidate]) -> tuple[list[Candidate], PipelineConfig]:
    """Drop candidates whose requires are unmet by the rest of the draw (repeat until stable)."""
    while True:
        config = build(base, spec, picked)
        kept = [c for c in picked if satisfied(config, c.requires)]
        if kept == picked:
            return picked, config
        picked = kept


def warm(engine: Engine, pages: list[Page], config: PipelineConfig) -> None:
    """Run segmentation + recognition with CTC posteriors once, so every text draw reuses them."""
    for page in pages:
        trace_page(engine, page, with_overrides(config, {'post.decoder': 'ctc', 'post.lexicon': None,
                                                        'post.period_lexicon': None}))


def check_no_gt_resources(spec: SearchSpec) -> None:
    for _, candidates in spec.stages:
        for candidate in candidates:
            if candidate.overrides.get('post.domain_lexicon'):
                raise ValueError(f'{candidate.stage}:{candidate.name} builds a lexicon from GT; not allowed here')


def run_shard(args: argparse.Namespace, spec: SearchSpec, base: PipelineConfig, out: Path) -> None:
    pages = select(discover(args.images, args.references), args.pages)
    engine = Engine(args.cache, args.device, args.seed)
    shard, shards = (int(x) for x in args.shard.split('/'))
    log = out / f'evaluations.shard{shard}.jsonl'
    done = {json.loads(line)['key'] for line in log.read_text(encoding='utf-8').splitlines()} if log.is_file() else set()
    for i in range(shard, args.n_image, shards):
        rng = random.Random(f'{args.seed}-{i}')
        image_picks = draw_stages(rng, base, spec, IMAGE_STAGES)
        text_stages = {s for s, _ in spec.stages} - IMAGE_STAGES
        warmed = False
        for j in range(args.n_text):
            picks, config = resolve(base, spec, image_picks + draw_stages(rng, base, spec, text_stages))
            key = stable_hash(config.to_dict())
            if key in done:
                continue
            if not warmed:
                warm(engine, pages, config)
                warmed = True
            result = evaluate(engine, pages, config, [])
            done.add(key)
            row = {'key': key, 'image_draw': i, 'text_draw': j, 'choices': [f'{c.stage}:{c.name}' for c in picks],
                   'config': config.to_dict(), **result}
            with log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            logger.info('image %d text %d: %s %.2f  %s', i, j, spec.metric, 100 * result['corpus'][spec.metric],
                        '+'.join(row['choices']))


def load_rows(out: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for log in sorted(out.glob('evaluations.shard*.jsonl'))
            for line in log.read_text(encoding='utf-8').splitlines()]
    return sorted(rows, key=lambda r: (r['image_draw'], r['text_draw']))


def best_so_far(rows: list[dict[str, Any]], metric: str) -> list[tuple[int, float]]:
    """Best metric after the first n configurations, in draw order, at a few checkpoints."""
    curve, best = [], float('inf')
    marks = {10, 25, 50, 100, 200, 400, 800, len(rows)}
    for n, row in enumerate(rows, 1):
        best = min(best, row['corpus'][metric])
        if n in marks:
            curve.append((n, best))
    return curve


def score_pages(engine: Engine, pages: list[Page], config: PipelineConfig) -> dict[str, float]:
    return {k: v for k, v in evaluate(engine, pages, config, [])['corpus'].items() if k.startswith(('wer', 'cer'))}


def summarize(args: argparse.Namespace, spec: SearchSpec, out: Path) -> None:
    rows = load_rows(out)
    metric = spec.metric
    values = sorted(r['corpus'][metric] for r in rows)
    best = min(rows, key=lambda r: (r['corpus'][metric], r['image_draw'], r['text_draw']))
    per_image: dict[int, float] = defaultdict(lambda: float('inf'))
    for row in rows:
        per_image[row['image_draw']] = min(per_image[row['image_draw']], row['corpus'][metric])
    best_config = from_dict(best['config'])
    all_pages = discover(args.images, args.references)
    engine = Engine(args.cache, args.device, args.seed)
    compared = {'random_best': best_config, **{p.stem: load_config(p) for p in args.compare}}
    scores = {name: {subset: score_pages(engine, select(all_pages, subset), config)
                     for subset in (args.pages, *args.final_pages)} for name, config in compared.items()}
    summary = {'metric': metric, 'configs': len(rows), 'image_draws': len(per_image),
               'quantiles': {q: values[min(len(values) - 1, int(q * len(values)))] for q in (0.0, 0.1, 0.25, 0.5)},
               'best_so_far': best_so_far(rows, metric), 'best': {k: best[k] for k in ('image_draw', 'text_draw',
                                                                                        'choices', 'config')},
               'scores': scores}
    for name, by_subset in scores.items():
        reference = by_subset[args.pages][metric]
        summary.setdefault('beaten_by_random', {})[name] = sum(v < reference for v in values)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'best_config.json').write_text(json.dumps(best['config'], ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('%d configurations; best %s %.2f: %s', len(rows), metric, 100 * values[0], '+'.join(best['choices']))
    for name, by_subset in scores.items():
        logger.info('%-24s %s', name, '  '.join(f'{s} {100 * v[metric]:.2f}' for s, v in by_subset.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('spec', type=Path)
    parser.add_argument('--pages', default='tuning', help='pages the search scores (never the held-out ones)')
    parser.add_argument('--n-image', type=int, default=60)
    parser.add_argument('--n-text', type=int, default=20)
    parser.add_argument('--shard', default='0/1', help='k/n: this process runs image draws k, k+n, k+2n, ...')
    parser.add_argument('--summarize', action='store_true', help='merge the shards, score the best on --final-pages')
    parser.add_argument('--final-pages', nargs='*', default=['heldout', 'all'])
    parser.add_argument('--compare', type=Path, nargs='*', default=[], help='configs scored beside the best')
    parser.add_argument('--images', type=Path, default=Path('imagens'))
    parser.add_argument('--references', type=Path, default=Path('test_kraken/ground_truth'))
    parser.add_argument('--out', type=Path, default=Path('outputs/random_search'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for noisy in ('kraken', 'lightning', 'lightning.pytorch', 'shapely'):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    payload = json.loads(args.spec.read_text(encoding='utf-8'))
    spec = parse_spec(payload)
    check_no_gt_resources(spec)
    if 'heldout' in args.pages or args.pages == 'all':
        raise ValueError('the search must not score held-out pages; use --final-pages for the chosen config')
    out = args.out / f'{args.spec.stem}__{args.pages}'
    out.mkdir(parents=True, exist_ok=True)
    mode = 'summary' if args.summarize else f"shard{args.shard.split('/')[0]}"
    (out / f'run_config.{mode}.json').write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()},
                                                     'spec': payload}, indent=2), encoding='utf-8')
    if args.summarize:
        summarize(args, spec, out)
    else:
        run_shard(args, spec, with_overrides(PipelineConfig(), spec.base), out)


if __name__ == '__main__':
    main()
