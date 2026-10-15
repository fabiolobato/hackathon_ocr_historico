from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from .config import PreprocessingConfig


def deskew(image: Image.Image, config: PreprocessingConfig) -> tuple[Image.Image, float]:
    gray = np.asarray(image.convert('L'))
    margin_y, margin_x = gray.shape[0] // 10, gray.shape[1] // 10
    crop = gray[margin_y:-margin_y or None, margin_x:-margin_x or None]
    mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    angles = np.arange(-config.deskew_limit, config.deskew_limit + config.deskew_step / 2,
                       config.deskew_step)
    scores = []
    for angle in angles:
        matrix = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), angle, 1)
        rotated = cv2.warpAffine(mask, matrix, (crop.shape[1], crop.shape[0]))
        scores.append(float(np.var(rotated.sum(axis=1))))
    best = max(range(len(angles)), key=lambda i: (scores[i], -abs(angles[i])))
    angle = float(angles[best])
    if angle == 0:
        return image.copy(), angle
    background = tuple(int(v) for v in np.median(np.asarray(image.convert('RGB')).reshape(-1, 3), axis=0))
    fill = background if image.mode == 'RGB' else int(np.median(gray))
    return image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=fill), angle


def dewarp(image: Image.Image, baselines: Sequence[Sequence[Sequence[float]]],
           config: PreprocessingConfig) -> tuple[Image.Image, float]:
    """Correct shared page curvature inferred from baseline deviations from chords."""
    width, height = image.size
    x_grid = np.arange(width)
    curves = []
    for baseline in baselines:
        points = np.asarray(baseline, dtype=float)
        if len(points) < 3:
            continue
        points = points[np.argsort(points[:, 0])]
        x, indices = np.unique(points[:, 0], return_index=True)
        if len(x) < 3 or x[-1] - x[0] < width * 0.3:
            continue
        y = points[indices, 1]
        chord = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
        curve = np.interp(x_grid, x, y - chord, left=np.nan, right=np.nan)
        curves.append(curve)
    if not curves:
        return image.copy(), 0.0
    stack = np.asarray(curves)
    valid = np.any(np.isfinite(stack), axis=0)
    curve = np.zeros(width)
    curve[valid] = np.nanmedian(stack[:, valid], axis=0)
    curve = cv2.GaussianBlur(curve.reshape(1, -1), (0, 0), 10).ravel()
    curve = np.clip(curve, -config.dewarp_limit, config.dewarp_limit)
    map_x, map_y = np.meshgrid(x_grid.astype('float32'), np.arange(height, dtype='float32'))
    map_y += curve.astype('float32')
    result = cv2.remap(np.asarray(image), map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(result), float(np.max(np.abs(curve)))


def process_image(image: Image.Image, operation: str,
                  config: PreprocessingConfig) -> Image.Image:
    if operation == 'kraken_nlbin':
        from kraken.binarization import nlbin

        return nlbin(image, threshold=config.nlbin_threshold)
    if operation == 'grayscale':
        return image.convert('L')
    if operation == 'contrast':
        return ImageEnhance.Contrast(image).enhance(config.contrast_factor)
    if operation == 'upscale':
        size = tuple(round(v * config.upscale_factor) for v in image.size)
        return image.resize(size, Image.Resampling.LANCZOS)
    gray = np.asarray(image.convert('L'))
    if operation == 'illumination':
        background = cv2.GaussianBlur(gray, (0, 0), config.background_sigma)
        normalized = gray.astype('float32') / np.maximum(background, 1) * 235
        return Image.fromarray(np.clip(normalized, 0, 255).astype('uint8'))
    if operation == 'otsu':
        return Image.fromarray(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    if operation == 'denoise':
        if image.mode == 'L':
            return Image.fromarray(cv2.fastNlMeansDenoising(gray, h=config.denoise_strength))
        rgb = np.asarray(image.convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = cv2.fastNlMeansDenoisingColored(bgr, None, config.denoise_strength,
                                               config.denoise_strength, 7, 21)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    if operation == 'clahe':
        clahe = cv2.createCLAHE(config.clahe_clip, (config.clahe_grid, config.clahe_grid))
        if image.mode == 'L':
            return Image.fromarray(clahe.apply(gray))
        lab = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    raise ValueError(f'Unknown processing operation: {operation}')
