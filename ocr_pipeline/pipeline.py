"""Page-level pipeline: base geometry -> segmentation image -> line filter -> recognition image -> text -> scores."""
import logging
from dataclasses import asdict
from typing import Any

from PIL import Image

from .config import OpParams, PipelineConfig, stable_hash
from .data import Page
from .engine import Engine, filter_lines
from .metrics import corpus, score
from .postprocess import correct
from .preprocessing import apply_chain

logger = logging.getLogger(__name__)

OP_PARAMS = {
    'upscale': ('upscale_factor',), 'deskew': ('deskew_limit', 'deskew_step'),
    'contrast': ('contrast_factor',), 'clahe': ('clahe_clip', 'clahe_grid'), 'denoise': ('denoise_strength',),
    'illumination': ('background_sigma',), 'kraken_nlbin': ('nlbin_threshold', 'nlbin_zoom'),
    'sharpen': ('sharpen_amount',), 'gamma': ('gamma',), 'remove_rules': ('rule_kernel_frac',),
}


def _ops_key(ops: tuple[str, ...], params: OpParams) -> list[Any]:
    return [(op, [getattr(params, name) for name in OP_PARAMS.get(op, ())]) for op in ops]


def cache_keys(page: Page, config: PipelineConfig) -> tuple[str, str]:
    base = [page.image.name, _ops_key(config.geometric_ops, config.params)]
    seg_key = stable_hash([base, _ops_key(config.seg_ops, config.params), config.seg.model])
    seg_filter = [config.seg.min_line_frac, config.seg.min_line_px, config.seg.drop_top_frac]
    rec_key = stable_hash([seg_key, seg_filter, _ops_key(config.rec_ops, config.params), asdict(config.rec)])
    return seg_key, rec_key


def recognize_page(engine: Engine, page: Page, config: PipelineConfig) -> tuple[list[str], int]:
    seg_key, rec_key = cache_keys(page, config)
    with Image.open(page.image) as original:
        base = apply_chain(original.convert('RGB'), config.geometric_ops, config.params, geometric=True)
    segmentation = engine.segment(apply_chain(base, config.seg_ops, config.params, geometric=False),
                                  config.seg, seg_key)
    segmentation = filter_lines(segmentation, config.seg, *base.size)
    rec_image = apply_chain(base, config.rec_ops, config.params, geometric=False)
    lines = engine.recognize(rec_image, segmentation, config.rec, rec_key)
    return lines, len(segmentation.lines)


def run_page(engine: Engine, page: Page, config: PipelineConfig) -> dict[str, Any]:
    lines, n_lines = recognize_page(engine, page, config)
    text = '\n'.join(correct(lines, config.post))
    return {'page': page.number, 'lines': n_lines, 'pred_words': len(text.split()), 'text': text,
            **score(page.reference_text(), text)}


def evaluate(engine: Engine, pages: list[Page], config: PipelineConfig) -> dict[str, Any]:
    rows = [run_page(engine, page, config) for page in pages]
    return {'corpus': corpus(rows), 'pages': rows}
