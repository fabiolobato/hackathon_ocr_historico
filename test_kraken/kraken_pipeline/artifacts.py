import hashlib
import json
from dataclasses import asdict
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .config import PipelineConfig, Variant


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def run_manifest(config: PipelineConfig, variants: tuple[Variant, ...],
                 images: list[Path]) -> dict[str, Any]:
    segmentation_model = config.segmentation_model or Path(str(resources.files('kraken') / 'blla.mlmodel'))
    models = [segmentation_model]
    if config.recognition_model is not None:
        models.append(config.recognition_model)
    return {
        'config': asdict(config),
        'variants': [asdict(v) for v in variants],
        'packages': {dist.metadata['Name']: metadata.version(dist.metadata['Name'])
                     for dist in metadata.distributions()},
        'model_sha256': {str(p): file_hash(p) for p in models},
        'image_sha256': {str(p): file_hash(p) for p in images},
        'pipeline_sha256': {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        'reference_sha256': {p.stem: file_hash(p) for p in config.reference_dir.glob('*.txt')},
        'source_document_sha256': {p.name: file_hash(p) for p in config.reference_dir.glob('*.docx')},
        'reference_review': json.loads((config.reference_dir / 'review.json').read_text(encoding='utf-8'))
            if (config.reference_dir / 'review.json').exists() else {},
    }


def segmentation_overlay(image: Image.Image, segmentation: Any) -> Image.Image:
    overlay = image.convert('RGB')
    draw = ImageDraw.Draw(overlay)
    for index, line in enumerate(segmentation.lines, 1):
        bbox = getattr(line, 'bbox', None)
        if bbox:
            draw.rectangle(tuple(bbox), outline='orange', width=2)
            draw.text(tuple(bbox[:2]), str(index), fill='blue')
        boundary = [tuple(p) for p in getattr(line, 'boundary', []) or []]
        baseline = [tuple(p) for p in getattr(line, 'baseline', []) or []]
        if boundary:
            draw.line(boundary + boundary[:1], fill='orange', width=1)
        if len(baseline) > 1:
            draw.line(baseline, fill='red', width=2)
            draw.text(baseline[0], str(index), fill='blue')
    return overlay
