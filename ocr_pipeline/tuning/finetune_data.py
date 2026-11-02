"""Build line-level PageXML training data from page-level GT by aligning GT words to OCR lines.

The supplied GT is per page, not per line. For each page: segment (cached), recognize greedily (cached),
align GT words to OCR words with a weighted edit distance, and give every OCR line the GT words aligned
to it. Lines whose OCR and assigned GT disagree too much are dropped. Only pages passed with --pages are
used; for honest evaluation pass the tuning split and evaluate the fine-tuned model on held-out pages.

  python -m ocr_pipeline.tuning.finetune_data --config configs/best_v1.json --pages tuning --out outputs/finetune/v1
"""
import argparse
import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image
from rapidfuzz.distance import Levenshtein

from ..config import PipelineConfig, load_config
from ..data import Page, discover, select
from ..engine import Engine, filter_lines
from ..pipeline import cache_keys
from ..preprocessing import apply_chain

logger = logging.getLogger(__name__)
PAGE_NS = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'


def _sub_cost(a: str, b: str) -> float:
    return Levenshtein.normalized_distance(a.lower(), b.lower())


def align_words(ref: list[str], hyp: list[str], gap: float = 0.75) -> list[int | None]:
    """For each ref word, the index of its aligned hyp word (None if deleted)."""
    n, m = len(ref), len(hyp)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i * gap
    for j in range(1, m + 1):
        cost[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + _sub_cost(ref[i - 1], hyp[j - 1]),
                             cost[i - 1][j] + gap, cost[i][j - 1] + gap)
    match: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if cost[i][j] == cost[i - 1][j - 1] + _sub_cost(ref[i - 1], hyp[j - 1]):
            match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
    return match


def assign_lines(ref: list[str], ocr_lines: list[str]) -> list[str]:
    hyp, hyp_line = [], []
    for index, line in enumerate(ocr_lines):
        for word in line.split():
            hyp.append(word)
            hyp_line.append(index)
    match = align_words(ref, hyp)
    owner = [hyp_line[j] if j is not None else None for j in match]
    for k in range(len(owner)):  # deleted words follow the previous aligned word
        if owner[k] is None:
            owner[k] = owner[k - 1] if k else next((o for o in owner if o is not None), 0)
    texts: list[list[str]] = [[] for _ in ocr_lines]
    for word, line in zip(ref, owner):
        texts[line].append(word)
    return [' '.join(words) for words in texts]


def _points(coords: Any) -> str:
    return ' '.join(f'{int(x)},{int(y)}' for x, y in coords)


def write_pagexml(path: Path, image_name: str, size: tuple[int, int], lines: list[tuple[Any, str]]) -> None:
    ET.register_namespace('', PAGE_NS)
    root = ET.Element(f'{{{PAGE_NS}}}PcGts')
    page = ET.SubElement(root, f'{{{PAGE_NS}}}Page', imageFilename=image_name,
                         imageWidth=str(size[0]), imageHeight=str(size[1]))
    region = ET.SubElement(page, f'{{{PAGE_NS}}}TextRegion', id='r0')
    ET.SubElement(region, f'{{{PAGE_NS}}}Coords', points=f'0,0 {size[0]},0 {size[0]},{size[1]} 0,{size[1]}')
    for index, (line, text) in enumerate(lines):
        node = ET.SubElement(region, f'{{{PAGE_NS}}}TextLine', id=f'l{index}')
        ET.SubElement(node, f'{{{PAGE_NS}}}Coords', points=_points(line.boundary))
        ET.SubElement(node, f'{{{PAGE_NS}}}Baseline', points=_points(line.baseline))
        equiv = ET.SubElement(node, f'{{{PAGE_NS}}}TextEquiv')
        ET.SubElement(equiv, f'{{{PAGE_NS}}}Unicode').text = text
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def build_page(engine: Engine, page: Page, config: PipelineConfig, out: Path, max_cer: float) -> dict[str, Any]:
    seg_key, rec_key = cache_keys(page, config)
    with Image.open(page.image) as original:
        base = apply_chain(original.convert('RGB'), config.geometric_ops, config.params, geometric=True)
    segmentation = filter_lines(engine.segment(apply_chain(base, config.seg_ops, config.params, geometric=False),
                                               config.seg, seg_key), config.seg, *base.size)
    rec_image = apply_chain(base, config.rec_ops, config.params, geometric=False)
    ocr_lines = engine.recognize(rec_image, segmentation, config.rec, rec_key)
    if len(ocr_lines) != len(segmentation.lines):
        raise RuntimeError(f'Page {page.number}: {len(ocr_lines)} predictions for {len(segmentation.lines)} lines')
    gt_lines = assign_lines(page.reference_text().split(), ocr_lines)
    kept = [(line, gt) for line, ocr, gt in zip(segmentation.lines, ocr_lines, gt_lines)
            if gt and Levenshtein.normalized_distance(ocr.lower(), gt.lower()) <= max_cer]
    image_name = f'page_{page.number:02d}.png'
    rec_image.save(out / image_name)
    write_pagexml(out / f'page_{page.number:02d}.xml', image_name, rec_image.size, kept)
    return {'page': page.number, 'lines': len(ocr_lines), 'kept': len(kept),
            'gt_words': len(page.reference_text().split()), 'kept_words': sum(len(t.split()) for _, t in kept)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--pages', default='tuning')
    parser.add_argument('--images', type=Path, default=Path('imagens'))
    parser.add_argument('--references', type=Path, default=Path('test_kraken/ground_truth'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-cer', type=float, default=0.6, help='drop lines with OCR/GT distance above this')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = load_config(args.config)
    pages = select(discover(args.images, args.references), args.pages)
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    engine = Engine(args.cache, 'cpu', args.seed)
    stats = [build_page(engine, page, config, args.out, args.max_cer) for page in pages]
    (args.out / 'stats.json').write_text(json.dumps({'pages': args.pages, 'max_cer': args.max_cer,
                                                     'config': config.to_dict(), 'stats': stats}, indent=2))
    for row in stats:
        logger.info('%s', row)


if __name__ == '__main__':
    main()
