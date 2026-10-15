import logging
import random
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def validate_device(device: str) -> None:
    import torch

    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False. '
                           'Enable a GPU runtime in Colab, or set device="cpu", '
                           'then rerun configuration and Kraken initialization. '
                           f'Torch version: {torch.__version__}; CUDA build: {torch.version.cuda}.')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is unavailable. Set device="cpu" and initialize again.')


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class KrakenEngine:
    def __init__(self, config: PipelineConfig) -> None:
        from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
        from kraken.ketos.util import to_ptl_device
        from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel

        validate_device(config.device)
        seed_everything(config.seed)
        self.config = config
        accelerator, devices = to_ptl_device(config.device)
        common = dict(accelerator=accelerator, device=devices,
                      precision='32-true', num_threads=1)
        self.seg_config = SegmentationInferenceConfig(**common)
        self.rec_config = RecognitionInferenceConfig(**common, num_line_workers=0)
        self.segmenter = SegmentationTaskModel.load_model(config.segmentation_model)
        self.recognizer = None
        if config.recognition_model is not None:
            if not config.recognition_model.is_file():
                raise FileNotFoundError(config.recognition_model)
            self.recognizer = RecognitionTaskModel.load_model(config.recognition_model)
        logger.info('Kraken initialized on %s; recognition model: %s',
                    config.device, config.recognition_model)

    def segment(self, image: Image.Image) -> Any:
        return self.segmenter.predict(image, self.seg_config)

    def recognize(self, image: Image.Image, segmentation: Any) -> list[Any]:
        if self.recognizer is None:
            raise ValueError('Set recognition_model to a handwriting model weights file, then initialize again.')
        if self.recognizer.seg_type != segmentation.type:
            logger.warning('Recognition model was trained on %s; using %s extraction, as requested.',
                           self.recognizer.seg_type, segmentation.type)
        return list(self.recognizer.predict(image, segmentation, self.rec_config))


def initialize_kraken(config: PipelineConfig) -> KrakenEngine:
    return KrakenEngine(config)
