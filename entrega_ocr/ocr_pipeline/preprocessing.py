"""Image operations. Geometric ops change coordinates; photometric ops never do.

deskew and the photometric ops are adapted from test_kraken/kraken_pipeline/preprocessing.py.
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .config import OpParams


def deskew(image: Image.Image, params: OpParams) -> Image.Image:
    gray = np.asarray(image.convert('L'))
    margin_y, margin_x = gray.shape[0] // 10, gray.shape[1] // 10
    crop = gray[margin_y:-margin_y or None, margin_x:-margin_x or None]
    mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    angles = np.arange(-params.deskew_limit, params.deskew_limit + params.deskew_step / 2, params.deskew_step)
    scores = []
    for angle in angles:
        matrix = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), angle, 1)
        rotated = cv2.warpAffine(mask, matrix, (crop.shape[1], crop.shape[0]))
        scores.append(float(np.var(rotated.sum(axis=1))))
    best = max(range(len(angles)), key=lambda i: (scores[i], -abs(angles[i])))
    angle = float(angles[best])
    if angle == 0:
        return image.copy()
    fill = tuple(int(v) for v in np.median(np.asarray(image.convert('RGB')).reshape(-1, 3), axis=0))
    return image.convert('RGB').rotate(angle, Image.Resampling.BICUBIC, expand=False, fillcolor=fill)


def slant_angle(image: Image.Image, params: OpParams) -> float:
    """Stroke tilt from vertical (degrees, clipped to slant_limit): the magnitude-weighted median orientation of
    strong, mostly-vertical edges in the page centre (margins and the binding are left out). A projection-profile
    criterion was tried first: at these resolutions it was dominated by page borders and returned 0."""
    gray = np.asarray(image.convert('L')).astype(np.float32)
    height, width = gray.shape
    centre = gray[height // 8:height - height // 8, width // 8:width - width // 8]
    gx = cv2.Sobel(centre, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(centre, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    edges = (magnitude > np.percentile(magnitude, 90)) & (np.abs(gx) > 2 * np.abs(gy))
    if not edges.any():
        return 0.0
    tilt = np.degrees(np.arctan(gy[edges] / gx[edges]))
    order = np.argsort(tilt)
    weights = np.cumsum(magnitude[edges][order])
    median = float(tilt[order][np.searchsorted(weights, weights[-1] / 2)])
    return float(np.clip(median, -params.slant_limit, params.slant_limit))


def shear(image: Image.Image, angle: float) -> Image.Image:
    """x' = x + tan(angle) * y; lines stay horizontal, so segmentation still works on the result."""
    if angle == 0:
        return image.copy()
    rgb = np.asarray(image.convert('RGB'))
    height, width = rgb.shape[:2]
    shift = np.tan(np.radians(angle)) * height
    matrix = np.float32([[1, np.tan(np.radians(angle)), max(0.0, -shift)], [0, 1, 0]])
    fill = tuple(int(v) for v in np.median(rgb.reshape(-1, 3), axis=0))
    sheared = cv2.warpAffine(rgb, matrix, (width + int(abs(shift)) + 1, height), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return Image.fromarray(sheared)


def deslant(image: Image.Image, params: OpParams) -> Image.Image:
    """Shear the page by the estimated stroke tilt times slant_gain (1 = upright, the sign is checked
    empirically through the gain: -1 doubles the slant)."""
    return shear(image, params.slant_gain * slant_angle(image, params))


def remove_rules(image: Image.Image, params: OpParams) -> Image.Image:
    """Suppress long, thin horizontal ruling lines by painting them with the local background."""
    gray = np.asarray(image.convert('L'))
    inverted = 255 - gray
    width = max(15, int(gray.shape[1] * params.rule_kernel_frac))
    tophat = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    rules = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1)))
    mask = cv2.threshold(rules, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = cv2.dilate(mask, np.ones((3, 1), np.uint8))
    background = cv2.medianBlur(gray, 7)
    result = np.where(mask > 0, np.maximum(gray, background), gray).astype('uint8')
    return Image.fromarray(result)


def apply_geometric(image: Image.Image, operation: str, params: OpParams) -> Image.Image:
    if operation == 'upscale':
        size = tuple(round(v * params.upscale_factor) for v in image.size)
        return image.resize(size, Image.Resampling.LANCZOS)
    if operation == 'deskew':
        return deskew(image, params)
    if operation == 'deslant':
        return deslant(image, params)
    raise ValueError(f'Unknown geometric op: {operation}')


def apply_photometric(image: Image.Image, operation: str, params: OpParams) -> Image.Image:
    if operation == 'kraken_nlbin':
        from kraken.binarization import nlbin

        return nlbin(image, threshold=params.nlbin_threshold, zoom=params.nlbin_zoom)
    if operation == 'grayscale':
        return image.convert('L')
    if operation == 'contrast':
        return ImageEnhance.Contrast(image).enhance(params.contrast_factor)
    if operation == 'sharpen':
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=int(100 * params.sharpen_amount), threshold=2))
    if operation == 'remove_rules':
        return remove_rules(image, params)
    if operation == 'channel':  # iron-gall ink on yellowed paper often contrasts best in one channel
        rgb = np.asarray(image.convert('RGB'))
        return Image.fromarray(rgb.min(axis=2) if params.channel == 'min' else rgb[:, :, 'rgb'.index(params.channel)])
    if operation in ('thicken', 'thin'):  # dark ink on light paper: a min filter thickens strokes, max thins
        size = params.stroke_size
        return image.filter(ImageFilter.MinFilter(size) if operation == 'thicken' else ImageFilter.MaxFilter(size))
    gray = np.asarray(image.convert('L'))
    if operation == 'gamma':
        lut = (255 * (np.arange(256) / 255) ** params.gamma).astype('uint8')
        return Image.fromarray(lut[gray])
    if operation == 'illumination':
        background = cv2.GaussianBlur(gray, (0, 0), params.background_sigma)
        normalized = gray.astype('float32') / np.maximum(background, 1) * 235
        return Image.fromarray(np.clip(normalized, 0, 255).astype('uint8'))
    if operation == 'otsu':
        return Image.fromarray(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    if operation == 'denoise':
        if image.mode == 'L':
            return Image.fromarray(cv2.fastNlMeansDenoising(gray, h=params.denoise_strength))
        bgr = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        result = cv2.fastNlMeansDenoisingColored(bgr, None, params.denoise_strength, params.denoise_strength, 7, 21)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    if operation == 'clahe':
        clahe = cv2.createCLAHE(params.clahe_clip, (params.clahe_grid, params.clahe_grid))
        if image.mode == 'L':
            return Image.fromarray(clahe.apply(gray))
        lab = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    raise ValueError(f'Unknown photometric op: {operation}')


def apply_chain(image: Image.Image, operations: tuple[str, ...], params: OpParams, geometric: bool) -> Image.Image:
    apply = apply_geometric if geometric else apply_photometric
    for operation in operations:
        image = apply(image, operation, params)
    return image
