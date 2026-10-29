"""Self-training data: run the pipeline on pages without GT and keep the lines it reads confidently.

label:  for every page, saves the recognition image (PNG) and one row per line in lines.jsonl: geometry, the
        recognizer's greedy reading, the pipeline's label (rescoring + per-line text post-processing, no
        dehyphenation so labels stay line-aligned) and confidence features (character posteriors along the
        greedy path, how much the post-processing changed the reading, share of words in the lexicon).
select: writes PageXML training files with only the lines that pass the thresholds (for `ketos train -f page`).

Pages must come from outside the evaluation set (e.g. the other pages of the same book, see page_crop): the
labels are pipeline output, but training on the evaluated images would still adapt the model to them.

  python -m ocr_pipeline.tuning.pseudo_labels label --config configs/v4_pass1.json \\
      --images imagens_extra/ldl1_crop --out outputs/pseudo/ldl1 --workers 3
  python -m ocr_pipeline.tuning.pseudo_labels select outputs/pseudo/ldl1 --min-mean-prob 0.9 --max-change 0.1
"""
import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

from ..batch import discover_batch
from ..config import PipelineConfig, load_config
from ..engine import Engine
from ..pipeline import cache_keys, page_logits, trace_page
from ..postprocess import correct, lexicon_choices
from .finetune_data import write_pagexml

logger = logging.getLogger(__name__)
_ENGINE: Engine | None = None


def path_probs(top_idx: np.ndarray, top_val: np.ndarray, blank: int = 0) -> np.ndarray:
    """Posterior of each character the greedy path emits (first frame of each non-blank run)."""
    best = top_idx[:, 0].astype(int)
    if not len(best):
        return np.zeros(0)
    start = np.r_[True, best[1:] != best[:-1]] & (best != blank)
    return top_val[start, 0].astype(float)


def line_rows(page_name: str, trace: Any, logits: list[Any], labels: list[str], known: frozenset[str]) -> list[dict]:
    rows = []
    for index, (line, greedy, label, logit) in enumerate(zip(trace.segmentation.lines, trace.greedy, labels, logits)):
        probs = path_probs(logit.top_idx, logit.top_val)
        words = label.split()
        rows.append({'page': page_name, 'line': index, 'greedy': greedy, 'label': label,
                     'baseline': [[int(x), int(y)] for x, y in line.baseline],
                     'boundary': [[int(x), int(y)] for x, y in line.boundary],
                     'chars': len(probs), 'mean_prob': float(probs.mean()) if len(probs) else 0.0,
                     'min_prob': float(probs.min()) if len(probs) else 0.0,
                     'change': Levenshtein.normalized_distance(greedy, label), 'words': len(words),
                     'known': sum(w.strip('.,;:!?()').lower() in known for w in words) / max(1, len(words))})
    return rows


def label_page(image: Path, config: PipelineConfig, out: Path) -> list[dict]:
    page = SimpleNamespace(number=0, image=image)
    trace = trace_page(_ENGINE, page, config)  # type: ignore[arg-type]  # duck-typed page: image + number
    _, rec_key = cache_keys(page, config)  # type: ignore[arg-type]
    logits = page_logits(_ENGINE, trace.base, trace.rec_image, trace.segmentation, config, rec_key)  # cached
    labels = correct(trace.lines, replace(config.post, dehyphenate=False))
    trace.rec_image.save(out / f'{image.stem}.png')
    return line_rows(image.stem, trace, logits, labels, frozenset(w.lower() for w in lexicon_choices(config.post, ())))


def _init(cache: Path, device: str, seed: int) -> None:
    global _ENGINE
    for noisy in ('kraken', 'lightning', 'lightning.pytorch', 'shapely'):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    _ENGINE = Engine(cache, device, seed, threads=1)


def _label_job(args: tuple[Path, PipelineConfig, Path]) -> list[dict]:
    return label_page(*args)


def label(config: PipelineConfig, images: Path, out: Path, cache: Path, device: str, seed: int, workers: int) -> None:
    (out / 'images').mkdir(parents=True, exist_ok=True)
    pages = [p.image for p in discover_batch(images)]
    jobs = [(page, config, out / 'images') for page in pages]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init,
                             initargs=(cache, device, seed)) as pool, \
            (out / 'lines.jsonl').open('w', encoding='utf-8') as stream:
        for done, rows in enumerate(pool.map(_label_job, jobs), 1):
            stream.writelines(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
            logger.info('%d/%d pages, %d lines', done, len(jobs), len(rows))
    (out / 'label_config.json').write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))


def passes(row: dict, args: argparse.Namespace) -> bool:
    return (row['chars'] >= args.min_chars and row['mean_prob'] >= args.min_mean_prob
            and row['min_prob'] >= args.min_min_prob and row['change'] <= args.max_change
            and row['known'] >= args.min_known)


def select(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in (args.run / 'lines.jsonl').read_text(encoding='utf-8').splitlines()]
    kept = [row for row in rows if passes(row, args)]
    target = args.run / 'train'
    target.mkdir(exist_ok=True)
    for old in target.glob('*.xml'):
        old.unlink()
    for page in sorted({row['page'] for row in kept}):
        image = args.run / 'images' / f'{page}.png'
        lines = [(SimpleNamespace(baseline=r['baseline'], boundary=r['boundary']), r['label'])
                 for r in kept if r['page'] == page]
        with Image.open(image) as im:
            write_pagexml(target / f'{page}.xml', f'../images/{page}.png', im.size, lines)
    summary = {'thresholds': {k: v for k, v in vars(args).items() if k.startswith(('min_', 'max_'))},
               'lines': len(rows), 'kept': len(kept), 'pages': len({r['page'] for r in kept}),
               'kept_chars': sum(r['chars'] for r in kept)}
    (args.run / 'select.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    lab = sub.add_parser('label')
    lab.add_argument('--config', type=Path, required=True)
    lab.add_argument('--images', type=Path, required=True)
    lab.add_argument('--out', type=Path, required=True)
    lab.add_argument('--cache', type=Path, help='default: <out>/cache')
    lab.add_argument('--device', default='cpu')
    lab.add_argument('--seed', type=int, default=42)
    lab.add_argument('--workers', type=int, default=1)
    sel = sub.add_parser('select')
    sel.add_argument('run', type=Path)
    sel.add_argument('--min-chars', type=int, default=10)
    sel.add_argument('--min-mean-prob', type=float, default=0.9)
    sel.add_argument('--min-min-prob', type=float, default=0.0)
    sel.add_argument('--max-change', type=float, default=0.1, help='normalized edit distance greedy -> label')
    sel.add_argument('--min-known', type=float, default=0.0, help='share of label words in the lexicon')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.command == 'label':
        label(load_config(args.config), args.images, args.out, args.cache or args.out / 'cache', args.device,
              args.seed, args.workers)
    else:
        select(args)


if __name__ == '__main__':
    main()
