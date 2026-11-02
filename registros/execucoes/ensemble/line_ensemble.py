"""Line-level ensemble of two recognizers sharing one segmentation (rule fixed before evaluation).

Per line: pick the model whose greedy text has the larger share of lexicon-known words; ties -> first model.
Output is the chosen model's CTC-rescored line; page post-processing then runs as usual.
"""
import argparse
import json
import logging
import re
from dataclasses import replace
from pathlib import Path

from ocr_pipeline.config import load_config, with_overrides
from ocr_pipeline.data import discover, select, split
from ocr_pipeline.engine import Engine
from ocr_pipeline.metrics import corpus
from ocr_pipeline.pipeline import domain_for, score_trace, trace_page
from ocr_pipeline.postprocess import lexicon_choices

WORD = re.compile(r'[^\W\d_]+')


def known_share(line: str, known: frozenset[str]) -> float:
    words = WORD.findall(line.lower())
    return sum(w in known for w in words) / len(words) if words else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', action='append', required=True, help='first = tie winner')
    parser.add_argument('--pages', default='heldout')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    base = load_config(args.config)
    configs = [with_overrides(base, {'rec.model': m}) for m in args.model]
    all_pages = discover(Path('imagens'), Path('test_kraken/ground_truth'))
    pool = split(all_pages)[0]
    engine = Engine(Path('outputs/cache'))
    rows, picks = [], {}
    for page in select(all_pages, args.pages):
        domain = domain_for(page, pool)
        known = frozenset(lexicon_choices(base.post, domain))
        traces = [trace_page(engine, page, c, domain) for c in configs]
        assert len({len(t.lines) for t in traces}) == 1, 'segmentations differ'
        chosen = [max(range(len(traces)), key=lambda i: (known_share(traces[i].greedy[j], known), -i))
                  for j in range(len(traces[0].lines))]
        lines = [traces[i].lines[j] for j, i in enumerate(chosen)]
        row = score_trace(page, replace(traces[0], lines=lines), base, domain)
        picks[page.number] = [chosen.count(i) for i in range(len(traces))]
        logging.info('page %d picks %s WERn %.1f', page.number, picks[page.number], 100 * row['wer_norm'])
        rows.append(row)
    result = {'corpus': corpus(rows), 'picks': picks, 'models': args.model, 'pages': rows}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logging.info('corpus %s', {k: round(100 * v, 2) for k, v in result['corpus'].items() if isinstance(v, float)})


if __name__ == '__main__':
    main()
