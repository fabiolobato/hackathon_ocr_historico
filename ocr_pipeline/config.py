"""Typed configuration for the integrated OCR pipeline.

The pipeline separates three image streams that share one geometry:
  base  = original -> geometric ops (upscale, deskew)          (defines coordinates)
  seg   = base -> seg_ops (photometric, e.g. nlbin)            (fed to the line segmenter)
  rec   = base -> rec_ops (photometric, e.g. grayscale/clahe)  (fed to the recognizer)
Photometric ops never move pixels, so segmentation polygons stay valid on the rec image.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

GEOMETRIC_OPS = frozenset({'upscale', 'deskew'})
PHOTOMETRIC_OPS = frozenset({'grayscale', 'contrast', 'clahe', 'denoise', 'illumination',
                             'otsu', 'kraken_nlbin', 'sharpen', 'remove_rules', 'gamma'})


@dataclass(frozen=True)
class OpParams:
    upscale_factor: float = 2.0
    deskew_limit: float = 5.0
    deskew_step: float = 0.25
    contrast_factor: float = 1.8
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    denoise_strength: float = 7.0
    background_sigma: float = 25.0
    nlbin_threshold: float = 0.5
    nlbin_zoom: float = 0.5
    sharpen_amount: float = 1.0
    gamma: float = 1.0
    rule_kernel_frac: float = 0.08


@dataclass(frozen=True)
class SegConfig:
    model: str | None = None  # None -> Kraken default blla.mlmodel
    min_line_frac: float = 0.0  # drop lines whose baseline spans less than this fraction of page width
    min_line_px: int = 0  # drop lines whose baseline is shorter than this many base-image pixels
    drop_top_frac: float = 0.0  # drop lines whose baseline lies in the top fraction (rubrics/page numbers)


@dataclass(frozen=True)
class RecConfig:
    model: str = 'medium'
    padding: int = 16
    no_legacy_polygons: bool = False


@dataclass(frozen=True)
class PostConfig:
    dehyphenate: bool = False  # join word split by a line-final hyphen
    lexicon: str | None = None  # path to a wordlist; None disables correction
    lexicon_max_ratio: float = 0.34  # max normalized edit distance accepted for a correction
    lexicon_min_len: int = 4


@dataclass(frozen=True)
class PipelineConfig:
    geometric_ops: tuple[str, ...] = ()
    seg_ops: tuple[str, ...] = ('kraken_nlbin',)
    rec_ops: tuple[str, ...] = ()
    params: OpParams = field(default_factory=OpParams)
    seg: SegConfig = field(default_factory=SegConfig)
    rec: RecConfig = field(default_factory=RecConfig)
    post: PostConfig = field(default_factory=PostConfig)

    def __post_init__(self) -> None:
        bad_geo = set(self.geometric_ops) - GEOMETRIC_OPS
        bad_photo = (set(self.seg_ops) | set(self.rec_ops)) - PHOTOMETRIC_OPS
        if bad_geo or bad_photo:
            raise ValueError(f'Unknown or misplaced ops: geometric={bad_geo}, photometric={bad_photo}')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _coerce(value: Any, current: Any) -> Any:
    if isinstance(current, tuple):
        if isinstance(value, str):
            return tuple(v for v in value.split(',') if v)
        return tuple(value)
    if isinstance(current, bool):
        return value if isinstance(value, bool) else str(value).lower() in {'1', 'true', 'yes'}
    if isinstance(current, (int, float)) and isinstance(value, str):
        return type(current)(value)
    if value in ('None', 'none', 'null'):
        return None
    return value


def with_overrides(config: PipelineConfig, overrides: dict[str, Any]) -> PipelineConfig:
    """Apply dotted-key overrides, e.g. {'params.upscale_factor': 2.5, 'rec_ops': 'grayscale'}."""
    for key, value in overrides.items():
        head, _, tail = key.partition('.')
        if tail:
            child = getattr(config, head)
            if tail not in {f.name for f in fields(child)}:
                raise KeyError(key)
            config = replace(config, **{head: replace(child, **{tail: _coerce(value, getattr(child, tail))})})
        else:
            if head not in {f.name for f in fields(config)}:
                raise KeyError(key)
            config = replace(config, **{head: _coerce(value, getattr(config, head))})
    return config


def from_dict(payload: dict[str, Any]) -> PipelineConfig:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            flat.update({f'{key}.{k}': v for k, v in value.items()})
        else:
            flat[key] = value
    return with_overrides(PipelineConfig(), flat)


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    return from_dict(json.loads(path.read_text(encoding='utf-8')))
