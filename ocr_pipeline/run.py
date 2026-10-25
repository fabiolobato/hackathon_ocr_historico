"""Evaluate one pipeline configuration. Writes metrics + predicted text; --viz adds report.html (images ↔ text).

Example:
  python -m ocr_pipeline.run --config configs/baseline.json --pages tuning --set rec_ops=grayscale
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, with_overrides
from .data import discover, select, split
from .engine import Engine
from .pipeline import evaluate

logger = logging.getLogger(__name__)


def parse_sets(items: list[str]) -> dict[str, str]:
    return dict(item.split('=', 1) for item in items)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--set', action='append', default=[], help='dotted override, e.g. params.upscale_factor=2')
    parser.add_argument('--pages', default='tuning', help="tuning | heldout | all | comma list of page numbers")
    parser.add_argument('--images', type=Path, default=Path('imagens'))
    parser.add_argument('--references', type=Path, default=Path('test_kraken/ground_truth'))
    parser.add_argument('--out', type=Path, default=Path('outputs/runs'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', default=None)
    parser.add_argument('--viz', action='store_true', help='also write report.html pairing images with text')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = with_overrides(load_config(args.config), parse_sets(args.set))
    all_pages = discover(args.images, args.references)
    pages = select(all_pages, args.pages)
    tuning_pool = split(all_pages)[0]
    engine = Engine(args.cache, args.device, args.seed)
    result = evaluate(engine, pages, config, tuning_pool)
    name = args.name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    payload = {'config': config.to_dict(), 'pages_subset': args.pages, 'seed': args.seed, **result}
    (out / 'result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in result['pages']:
        (out / f"page_{row['page']:02d}.txt").write_text(row['text'], encoding='utf-8')
    if args.viz:
        from .visualize import build_report

        report = build_report(engine, pages, {name: config}, tuning_pool, title=f'OCR run {name}')
        (out / 'report.html').write_text(report, encoding='utf-8')
    c = result['corpus']
    logger.info('WER strict %.2f%% | WER norm %.2f%% | CER norm %.2f%% -> %s',
                100 * c['wer_strict'], 100 * c['wer_norm'], 100 * c['cer_norm'], out)
    for row in result['pages']:
        logger.info('page %2d lines %3d words %3d WERn %.1f%% CERn %.1f%%', row['page'], row['lines'],
                    row['pred_words'], 100 * row['wer_norm'], 100 * row['cer_norm'])


if __name__ == '__main__':
    main()
