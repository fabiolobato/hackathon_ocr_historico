import csv
import json
import logging
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .artifacts import run_manifest, segmentation_overlay, write_json
from .config import PipelineConfig, PreprocessingConfig, Variant
from .engine import KrakenEngine, seed_everything
from .metrics import word_errors
from .preprocessing import deskew, dewarp, process_image

logger = logging.getLogger(__name__)


def dataset_images(config: PipelineConfig) -> list[Path]:
    images = list(config.image_dir.glob('*.jpg'))
    if not images:
        raise FileNotFoundError(f'No JPEG images in {config.image_dir}')
    return sorted(images, key=lambda path: int(path.name.split('.')[0]))


def prepare_image(image: Image.Image, variant: Variant, engine: KrakenEngine,
                  config: PipelineConfig, stage_dir: Path | None = None) -> tuple[Image.Image, list[dict[str, Any]]]:
    prepared = image.copy()
    details = []
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)
        prepared.save(stage_dir / '00_original.png')
    for index, operation in enumerate(variant.operations, 1):
        detail: dict[str, Any] = {'operation': operation}
        if operation == 'deskew':
            prepared, angle = deskew(prepared, config.preprocessing)
            detail['rotation_degrees'] = angle
        elif operation == 'dewarp':
            segmentation = engine.segment(prepared)
            baselines = [line.baseline for line in segmentation.lines]
            prepared, displacement = dewarp(prepared, baselines, config.preprocessing)
            detail['maximum_displacement_pixels'] = displacement
        else:
            parameters = PreprocessingConfig() if variant.name == 'kraken_neural' else config.preprocessing
            prepared = process_image(prepared, operation, parameters)
            if operation == 'kraken_nlbin':
                detail['threshold'] = parameters.nlbin_threshold
        if stage_dir is not None:
            prepared.save(stage_dir / f'{index:02d}_{operation}.png')
        details.append(detail)
    return prepared, details


def reference_status(config: PipelineConfig, image: Path) -> str:
    review_path = config.reference_dir / 'review.json'
    if not review_path.exists():
        return 'unreviewed'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    return review.get('pages', {}).get(image.stem, {}).get('status', 'unreviewed')


def run_page(config: PipelineConfig, engine: KrakenEngine, image_path: Path,
             variant: Variant, run_dir: Path) -> dict[str, Any]:
    seed_everything(config.seed)
    page_dir = run_dir / variant.name / image_path.stem
    page_dir.mkdir(parents=True)
    with Image.open(image_path) as original:
        image, processing = prepare_image(original, variant, engine, config, page_dir / 'stages')
    image.save(page_dir / 'input.png')
    segmentation = engine.segment(image)
    write_json(page_dir / 'segmentation.json', asdict(segmentation))
    segmentation_overlay(image, segmentation).save(page_dir / 'segmentation_overlay.png')
    records = engine.recognize(image, segmentation)
    text = '\n'.join(record.prediction for record in records)
    (page_dir / 'prediction.txt').write_text(text, encoding='utf-8')
    write_json(page_dir / 'recognition.json', [asdict(record) for record in records])
    diagnostic = 'zero_lines' if not segmentation.lines else 'empty_recognition' if not text.strip() else 'text'
    if diagnostic != 'text':
        logger.warning('%s em %s (%s)', diagnostic, image_path.name, variant.name)
    result = {'detected_lines': len(segmentation.lines), 'recognized_lines': len(records),
              'predicted_words': len(text.split()), 'empty_output': not bool(text.strip()),
              'diagnostic': diagnostic, 'page': image_path.name, 'variant': variant.name, 'status': 'ok',
              'reference_status': reference_status(config, image_path),
              'processing': processing, 'artifacts': str(page_dir)}
    reference = config.reference_dir / f'{image_path.stem}.txt'
    if reference.is_file():
        score = word_errors(reference.read_text(encoding='utf-8'), text, config.scoring)
        result.update(asdict(score), errors=score.errors, wer=score.wer)
    else:
        result.update(wer=None, reference_status='missing')
    write_json(page_dir / 'result.json', result)
    return result


def summarize(results: list[dict[str, Any]], expected_pages: int) -> list[dict[str, Any]]:
    summary = []
    for name in dict.fromkeys(row['variant'] for row in results):
        rows = [row for row in results if row['variant'] == name]
        scored = [row for row in rows if row.get('wer') is not None]
        complete = len(scored) == expected_pages and all(row['status'] == 'ok' for row in rows)
        words = sum(row['reference_words'] for row in scored)
        summary.append({'variant': name, 'pages_scored': len(scored),
                        'pages_expected': expected_pages, 'complete': complete,
                        'nonempty_pages': sum(not row.get('empty_output', False) for row in scored),
                        'empty_pages': sum(row.get('empty_output', False) for row in scored),
                        'reference_status': 'reviewed' if scored and all(
                            row['reference_status'] == 'reviewed' for row in scored) else 'provisional',
                        'wer': sum(row['errors'] for row in scored) / words if complete else None})
    baseline_name = next((name for name in ('kraken_neural', 'raw')
                          if any(row['variant'] == name for row in summary)), 'raw')
    baseline = next((row['wer'] for row in summary if row['variant'] == baseline_name), None)
    for row in summary:
        row['baseline_variant'] = baseline_name
        row['delta_wer'] = row['wer'] - baseline if baseline is not None and row['wer'] is not None else None
    return summary


def save_results(run_dir: Path, results: list[dict[str, Any]], expected_pages: int) -> None:
    write_json(run_dir / 'results.json', results)
    summary = summarize(results, expected_pages)
    write_json(run_dir / 'summary.json', summary)
    if summary:
        with (run_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)


def run_experiments(config: PipelineConfig, engine: KrakenEngine,
                    variants: tuple[Variant, ...], images: list[Path] | None = None,
                    fail_fast: bool = False) -> Path:
    if engine.recognizer is None:
        raise ValueError('Choose a recognition model before running experiments.')
    if any(getattr(config, key) != getattr(engine.config, key)
           for key in ('recognition_model', 'segmentation_model', 'device')):
        raise ValueError('Model or device configuration changed. Initialize Kraken again.')
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError('Variant names must be unique.')
    if any(variant.segmenter != 'neural' for variant in variants):
        raise ValueError('Este pipeline usa somente segmentação neural.')
    images = images if images is not None else dataset_images(config)
    if not images:
        raise ValueError('Select at least one image.')
    run_dir = config.output_dir / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir.mkdir(parents=True)
    write_json(run_dir / 'config.json', run_manifest(config, variants, images))
    logger.info('Starting run: %s', run_dir)
    results = []
    for variant in variants:
        for image in images:
            logger.info('Processing %s: %s', variant.name, image.name)
            try:
                result = run_page(config, engine, image, variant, run_dir)
            except Exception as error:
                logger.exception('Failed %s on %s', variant.name, image.name)
                result = {'page': image.name, 'variant': variant.name,
                          'status': 'failed', 'error': repr(error),
                          'traceback': traceback.format_exc(), 'wer': None}
                if fail_fast:
                    save_results(run_dir, results + [result], len(images))
                    raise RuntimeError(f'{variant.name} failed on {image.name}: {error}. Details: {run_dir / "results.json"}') from error
            results.append(result)
            save_results(run_dir, results, len(images))
    return run_dir
