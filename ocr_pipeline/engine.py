"""Kraken wrapper: one segmenter, lazily loaded recognizers, disk caches keyed by config hashes."""
import logging
import pickle
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import RecConfig, SegConfig

logger = logging.getLogger(__name__)

MODELS = {
    'medium': Path('models/10.5281_zenodo.21788410/medium.safetensors'),
    'small': Path('models/10.5281_zenodo.21788405/small.safetensors'),
    'traprinq': Path('models/10.5281_zenodo.14543904/traprinq.mlmodel'),
}


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Engine:
    def __init__(self, cache_dir: Path, device: str = 'cpu', seed: int = 42) -> None:
        from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
        from kraken.ketos.util import to_ptl_device

        seed_everything(seed)
        self.seed = seed
        self.cache_dir = cache_dir
        accelerator, devices = to_ptl_device(device)
        self._common = dict(accelerator=accelerator, device=devices, precision='32-true', num_threads=4)
        self._seg_config_cls, self._rec_config_cls = SegmentationInferenceConfig, RecognitionInferenceConfig
        self._segmenters: dict[str | None, Any] = {}
        self._recognizers: dict[str, Any] = {}

    def _segmenter(self, model: str | None) -> Any:
        from kraken.tasks import SegmentationTaskModel

        if model not in self._segmenters:
            self._segmenters[model] = SegmentationTaskModel.load_model(model)
        return self._segmenters[model]

    def _recognizer(self, model: str) -> Any:
        from kraken.tasks import RecognitionTaskModel

        if model not in self._recognizers:
            path = MODELS.get(model, Path(model))
            if not path.is_file():
                raise FileNotFoundError(f'Recognition weights missing: {path}')
            self._recognizers[model] = RecognitionTaskModel.load_model(path)
        return self._recognizers[model]

    def _cached(self, kind: str, key: str, compute: Any) -> Any:
        path = self.cache_dir / kind / f'{key}.pkl'
        if path.is_file():
            return pickle.loads(path.read_bytes())
        value = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(value))
        return value

    def segment(self, image: Image.Image, config: SegConfig, key: str) -> Any:
        def compute() -> Any:
            seed_everything(self.seed)
            return self._segmenter(config.model).predict(image, self._seg_config_cls(**self._common))
        return self._cached('seg', key, compute)

    def recognize(self, image: Image.Image, segmentation: Any, config: RecConfig, key: str) -> list[str]:
        def compute() -> list[str]:
            if not segmentation.lines:
                return []
            seed_everything(self.seed)
            rec_config = self._rec_config_cls(**self._common, num_line_workers=0, padding=config.padding,
                                              no_legacy_polygons=config.no_legacy_polygons)
            return [r.prediction for r in self._recognizer(config.model).predict(image, segmentation, rec_config)]
        return self._cached('rec', key, compute)


def filter_lines(segmentation: Any, config: SegConfig, width: int, height: int) -> Any:
    def keep(line: Any) -> bool:
        xs = [p[0] for p in line.baseline]
        ys = [p[1] for p in line.baseline]
        span = max(xs) - min(xs)
        if span < config.min_line_frac * width or span < config.min_line_px:
            return False
        return float(np.mean(ys)) >= config.drop_top_frac * height
    return replace(segmentation, lines=[line for line in segmentation.lines if keep(line)])
