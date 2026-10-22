"""Page-level pipeline: base geometry -> segmentation image -> line filter -> recognition image -> text -> scores."""
import logging
from dataclasses import asdict
from typing import Any

from PIL import Image

from .config import OpParams, PipelineConfig, stable_hash
from .data import Page
from .engine import Engine, filter_lines
from .metrics import corpus, score
from .ctc_rescore import RescoreParams, rescore_line
from .postprocess import correct, domain_vocabulary, lexicon_choices
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


def recognize_page(engine: Engine, page: Page, config: PipelineConfig,
                   domain: tuple[str, ...] = ()) -> tuple[list[str], int]:
    seg_key, rec_key = cache_keys(page, config)
    with Image.open(page.image) as original:
        base = apply_chain(original.convert('RGB'), config.geometric_ops, config.params, geometric=True)
    segmentation = engine.segment(apply_chain(base, config.seg_ops, config.params, geometric=False),
                                  config.seg, seg_key)
    segmentation = filter_lines(segmentation, config.seg, *base.size)
    rec_image = apply_chain(base, config.rec_ops, config.params, geometric=False)
    if config.post.decoder in ('ctc', 'ctc+edit'):
        return rescore_page(engine, rec_image, segmentation, config, rec_key, domain), len(segmentation.lines)
    lines = engine.recognize(rec_image, segmentation, config.rec, rec_key)
    return lines, len(segmentation.lines)


def rescore_page(engine: Engine, image: Image.Image, segmentation: Any, config: PipelineConfig, key: str,
                 domain: tuple[str, ...]) -> list[str]:
    logits = engine.recognize_logits(image, segmentation, config.rec, key)
    choices = lexicon_choices(config.post, domain)
    if not choices:
        return [line.prediction for line in logits]
    c2l, l2c, n_classes = engine.codec(config.rec.model)
    params = RescoreParams(config.post.ctc_alpha, config.post.lexicon_max_ratio, config.post.ctc_candidates,
                           config.post.lexicon_min_len)
    known = frozenset(choices)
    return [rescore_line(line, c2l, l2c, n_classes, known, choices, params) for line in logits]


def run_page(engine: Engine, page: Page, config: PipelineConfig, domain: tuple[str, ...] = ()) -> dict[str, Any]:
    lines, n_lines = recognize_page(engine, page, config, domain)
    text = '\n'.join(correct(lines, config.post, domain))
    return {'page': page.number, 'lines': n_lines, 'pred_words': len(text.split()), 'text': text,
            **score(page.reference_text(), text)}


def domain_for(page: Page, pool: list[Page]) -> tuple[str, ...]:
    """Leave-one-out: in-domain words from tuning pages other than the evaluated one."""
    return domain_vocabulary([p.reference_text() for p in pool if p.number != page.number])


def evaluate(engine: Engine, pages: list[Page], config: PipelineConfig,
             lexicon_pool: list[Page] | None = None) -> dict[str, Any]:
    """lexicon_pool: pages whose GT may feed the domain lexicon (the tuning split, never held-out pages)."""
    pool = lexicon_pool if lexicon_pool is not None else []
    rows = [run_page(engine, page, config, domain_for(page, pool) if config.post.domain_lexicon else ())
            for page in pages]
    return {'corpus': corpus(rows), 'pages': rows}
