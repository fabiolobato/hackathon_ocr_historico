"""Score final configurations on all 12 pages and report tuning, held-out and all (corpus) plus per-page numbers.

Held-out pages are scored once, for the configurations already chosen on the tuning pages; nothing here feeds back
into a choice.

  python tools/score_final.py configs/greedy_v4_medium.json configs/final_v4.json \\
      --labels "gulosa v4 (medium)" "final v4 (st_v3)" --out outputs/search/final_scores_v4.json --workers 4
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_pipeline.config import load_config  # noqa: E402  (after the path insert)
from ocr_pipeline.data import discover, split  # noqa: E402
from ocr_pipeline.engine import Engine  # noqa: E402
from ocr_pipeline.metrics import corpus  # noqa: E402
from ocr_pipeline.pipeline import evaluate, page_executor  # noqa: E402

logger = logging.getLogger(__name__)


def summarize(rows: list[dict], heldout: set[int]) -> dict:
    def pct(subset: list[dict]) -> dict:
        c = corpus(subset)
        return {'wer_norm': round(100 * c['wer_norm'], 2), 'cer_norm': round(100 * c['cer_norm'], 2)}

    return {'tuning': pct([r for r in rows if r['page'] not in heldout]),
            'heldout': pct([r for r in rows if r['page'] in heldout]),
            'all': pct(rows),
            'pages': {r['page']: [round(100 * r['wer_norm'], 1), round(100 * r['cer_norm'], 1)] for r in rows}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configs', type=Path, nargs='+')
    parser.add_argument('--labels', nargs='+', help='one per config (default: file stem)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--images', type=Path, default=Path('imagens'))
    parser.add_argument('--references', type=Path, default=Path('test_kraken/ground_truth'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    labels = args.labels or [c.stem for c in args.configs]
    if len(labels) != len(args.configs):
        raise SystemExit('one label per config')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for noisy in ('kraken', 'lightning', 'lightning.pytorch', 'shapely'):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    pages = sorted(discover(args.images, args.references), key=lambda p: p.number)
    pool, held = split(pages)
    heldout = {p.number for p in held}
    engine = Engine(args.cache, args.device, 42)
    out = {}
    with page_executor(engine, args.workers) as executor:
        for label, path in zip(labels, args.configs):
            result = evaluate(engine, pages, load_config(path), pool, executor)
            out[label] = {'config': str(path), **summarize(result['pages'], heldout)}
            logger.info('%s: %s', label, {k: out[label][k] for k in ('tuning', 'heldout', 'all')})
            out[label]['texts'] = {r['page']: r['text'] for r in result['pages']}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
