"""Run a list of configuration variants against a page subset and print a leaderboard.

Sweep file (JSON): {"base": {<dotted overrides>}, "variants": {"name": {<dotted overrides>}, ...}}
Each variant is applied on top of base. Results are appended to <out>/<sweep name>.jsonl.
"""
import argparse
import json
import logging
from pathlib import Path

from .config import load_config, with_overrides
from .data import discover, select
from .engine import Engine
from .pipeline import evaluate

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sweep', type=Path)
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--pages', default='tuning')
    parser.add_argument('--images', type=Path, default=Path('imagens'))
    parser.add_argument('--references', type=Path, default=Path('test_kraken/ground_truth'))
    parser.add_argument('--out', type=Path, default=Path('outputs/sweeps'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only', default=None, help='comma list of variant names to run')
    return parser


def leaderboard(rows: list[dict]) -> str:
    lines = [f"{'variant':40s} {'WERs':>7s} {'WERn':>7s} {'CERn':>7s}"]
    for row in sorted(rows, key=lambda r: r['corpus']['wer_norm']):
        c = row['corpus']
        lines.append(f"{row['variant']:40s} {100 * c['wer_strict']:7.2f} {100 * c['wer_norm']:7.2f} "
                     f"{100 * c['cer_norm']:7.2f}")
    return '\n'.join(lines)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for noisy in ('kraken', 'lightning', 'lightning.pytorch', 'shapely'):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    spec = json.loads(args.sweep.read_text(encoding='utf-8'))
    base = with_overrides(load_config(args.config), spec.get('base', {}))
    pages = select(discover(args.images, args.references), args.pages)
    engine = Engine(args.cache, args.device, args.seed)
    only = set(args.only.split(',')) if args.only else None
    out = args.out / f'{args.sweep.stem}__{args.pages}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, overrides in spec['variants'].items():
        if only and name not in only:
            continue
        config = with_overrides(base, overrides)
        result = evaluate(engine, pages, config)
        row = {'variant': name, 'overrides': overrides, 'pages_subset': args.pages,
               'config': config.to_dict(), **result}
        rows.append(row)
        with out.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        c = result['corpus']
        logger.info('%s: WERs %.2f WERn %.2f CERn %.2f', name, 100 * c['wer_strict'], 100 * c['wer_norm'],
                    100 * c['cer_norm'])
    print(leaderboard(rows))


if __name__ == '__main__':
    main()
