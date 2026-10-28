"""Learn the recognizer's edit costs by self-training (Tong & Evans 1996): no GT is read.

Each round runs the pipeline on the chosen pages, pairs every word the recognizer read with the word the
pipeline turned it into, and counts the character operations between them (ocr_pipeline.edit_costs). The next
round generates candidates with those costs. Only the tuning pages are used, so held-out pages stay unseen.

  python tools/learn_edit_costs.py --config configs/final_v3.json --out models/edit_costs/tuning_r2.json
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from ocr_pipeline.config import load_config, with_overrides  # noqa: E402
from ocr_pipeline.data import discover, select  # noqa: E402
from ocr_pipeline.edit_costs import count_operations, learn, save  # noqa: E402
from ocr_pipeline.engine import Engine  # noqa: E402
from ocr_pipeline.pipeline import trace_page  # noqa: E402
from ocr_pipeline.postprocess import fold  # noqa: E402

_WORDS = re.compile(r'[^\W\d_]+')
logger = logging.getLogger(__name__)


def word_pairs(read: str, corrected: str) -> list[tuple[str, str]]:
    """(read word, corrected word), accent-free lowercase, for words the decoder kept or replaced one to one."""
    a = [fold(w.lower()) for w in _WORDS.findall(read)]
    b = [fold(w.lower()) for w in _WORDS.findall(corrected)]
    vocab: dict[str, int] = {}
    ia, ib = [vocab.setdefault(w, len(vocab)) for w in a], [vocab.setdefault(w, len(vocab)) for w in b]
    pairs = []
    for op in Levenshtein.opcodes(ia, ib):
        if op.tag in ('equal', 'replace') and op.src_end - op.src_start == op.dest_end - op.dest_start:
            pairs += zip(a[op.src_start:op.src_end], b[op.dest_start:op.dest_end])
    return pairs


def collect(engine: Engine, pages: list, config) -> list[tuple[str, str]]:
    pairs = []
    for page in pages:
        trace = trace_page(engine, page, config)
        for read, corrected in zip(trace.greedy, trace.lines):
            pairs += word_pairs(read, corrected)
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='rules of the last round; each round also saved')
    parser.add_argument('--pages', default='tuning', help='never heldout')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--min-count', type=int, default=3)
    parser.add_argument('--images', type=Path, default=Path('imagens'))
    parser.add_argument('--gt', type=Path, default=Path('test_kraken/ground_truth'), help='only to list pages')
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache'))
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.pages == 'heldout':
        raise SystemExit('edit costs are learned on tuning pages only')
    pages = select(discover(args.images, args.gt), args.pages)
    engine = Engine(args.cache, 'cpu', args.seed)
    config = load_config(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for round_ in range(1, args.rounds + 1):
        pairs = collect(engine, pages, config)
        rules = learn(pairs, args.min_count)
        ops, _ = count_operations(pairs)
        path = args.out.with_name(f'{args.out.stem}_round{round_}.json')
        save(path, rules, {'config': str(args.config), 'pages': [p.number for p in pages], 'round': round_,
                           'min_count': args.min_count, 'pairs': len(pairs),
                           'changed_pairs': sum(r != c for r, c in pairs),
                           'top_operations': [[s, d, n] for (s, d), n in ops.most_common(40)]})
        logger.info('round %d: %d pairs, %d rules -> %s', round_, len(pairs), sum(map(len, rules.values())), path)
        config = with_overrides(config, {'post.edit_costs': str(path)})
    args.out.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')


if __name__ == '__main__':
    main()
