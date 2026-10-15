from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from kraken_pipeline.config import PipelineConfig, PreprocessingConfig, RAW, ScoringConfig, Variant
from kraken_pipeline.experiments import prepare_image, run_experiments, summarize
from kraken_pipeline.metrics import word_errors
from kraken_pipeline.preprocessing import deskew, dewarp, process_image
from kraken_pipeline.references import split_sections


def test_wer_edit_counts_and_empty_prediction() -> None:
    score = word_errors('a b c', 'a x c extra')
    assert (score.substitutions, score.deletions, score.insertions) == (1, 0, 1)
    assert score.wer == pytest.approx(2 / 3)
    assert word_errors('a b', '').wer == 1
    assert word_errors('a', 'a b c').wer == 2
    with pytest.raises(ValueError):
        word_errors('', 'a')


def test_scoring_preserves_case_punctuation_and_accents() -> None:
    assert word_errors('São', 'Sa\u0303o').wer == 0
    assert word_errors('São, Carlos', 'são carlos').wer == 1
    assert word_errors('São, Carlos', 'são carlos', ScoringConfig(True, True)).wer == 0


def test_micro_wer_and_failure_prevents_partial_ranking() -> None:
    rows = [{'variant': 'raw', 'status': 'ok', 'wer': 1.0, 'errors': 1,
             'reference_words': 1, 'reference_status': 'reviewed'},
            {'variant': 'raw', 'status': 'ok', 'wer': 0.0, 'errors': 0,
             'reference_words': 9, 'reference_status': 'reviewed'}]
    assert summarize(rows, 2)[0]['wer'] == 0.1
    rows[1] = {'variant': 'raw', 'status': 'failed', 'wer': None}
    assert summarize(rows, 2)[0]['wer'] is None


def test_raw_and_individual_operations_start_from_original() -> None:
    image = Image.fromarray(np.arange(300, dtype='uint8').reshape(10, 10, 3))
    original = np.asarray(image).copy()
    config = PipelineConfig()
    raw, details = prepare_image(image, RAW, None, config)  # type: ignore[arg-type]
    assert np.array_equal(np.asarray(raw), original)
    assert raw.mode == image.mode and details == []
    prepare_image(image, Variant('contrast', ('contrast',)), None, config)  # type: ignore[arg-type]
    assert np.array_equal(np.asarray(image), original)
    gray, _ = prepare_image(image, Variant('grayscale', ('grayscale',)), None, config)  # type: ignore[arg-type]
    assert np.array_equal(np.asarray(gray), np.asarray(image.convert('L')))


def test_deskew_known_tilt() -> None:
    image = Image.new('L', (400, 400), 255)
    draw = ImageDraw.Draw(image)
    for y in range(50, 350, 25):
        draw.rectangle((50, y, 350, y + 3), fill=0)
    tilted = image.rotate(3, fillcolor=255)
    _, angle = deskew(tilted, PreprocessingConfig())
    assert angle == pytest.approx(-3, abs=0.5)


def test_dewarp_curved_baseline_and_no_baseline_fallback() -> None:
    image = Image.new('L', (200, 100), 255)
    draw = ImageDraw.Draw(image)
    curve = [[x, 40 + 12 * np.sin(np.pi * x / 199)] for x in range(200)]
    draw.line([tuple(p) for p in curve], fill=0, width=2)
    flattened, displacement = dewarp(image, [curve], PreprocessingConfig())
    assert displacement > 10
    assert np.std(np.argmin(np.asarray(flattened), axis=0)[20:-20]) < 1
    unchanged, displacement = dewarp(image, [], PreprocessingConfig())
    assert np.array_equal(np.asarray(unchanged), np.asarray(image)) and displacement == 0


def test_docx_sections_do_not_leak_headers_or_adjacent_pages() -> None:
    paragraphs = ['metadata', '5.BR FPMSC FUN CAM LDL - Livro. Página 3.-5',
                  'first page', '', '_____',
                  '6.BR FPMSC FUN CAM LDL - Livro. Página 4.-6', 'second page']
    assert split_sections(paragraphs) == {5: [(2, 'first page')], 6: [(6, 'second page')]}


def test_all_dataset_references_exist() -> None:
    images = list(Path('Imagens').glob('*.jpg'))
    if not images:
        pytest.skip('Local dataset not included')
    for image in images:
        assert (Path('ground_truth') / f'{image.stem}.txt').read_text(encoding='utf-8').strip()


def test_inference_configs_preserve_krakens_recoverable_error_defaults() -> None:
    from kraken_pipeline.engine import initialize_kraken

    with patch('kraken.tasks.SegmentationTaskModel.load_model'):
        engine = initialize_kraken(PipelineConfig())
    assert engine.seg_config.raise_on_error is False
    assert engine.rec_config.raise_on_error is False


def test_orchestrated_run_saves_provenance_and_fair_scores(tmp_path: Path) -> None:
    from kraken.containers import Segmentation

    @dataclass
    class Record:
        prediction: str

    image_path = tmp_path / '1.page.jpg'
    Image.new('RGB', (30, 30), (120, 100, 80)).save(image_path)
    refs = tmp_path / 'refs'
    refs.mkdir()
    (refs / '1.page.txt').write_text('a b', encoding='utf-8')
    config = PipelineConfig(reference_dir=refs, output_dir=tmp_path / 'runs')
    engine = SimpleNamespace(config=config, recognizer=object(),
                             segment=lambda im: Segmentation(type='baselines', lines=[],
                                                             imagename=str(image_path),
                                                             text_direction='horizontal-lr',
                                                             script_detection=False),
                             recognize=lambda im, seg: [Record('a b' if im.mode == 'RGB' else 'a extra b')])
    run_dir = run_experiments(config, engine, (RAW, Variant('gray', ('grayscale',))), [image_path])
    rows = json.loads((run_dir / 'summary.json').read_text())
    assert rows[0]['wer'] == 0 and rows[1]['wer'] == 0.5
    assert rows[1]['delta_wer'] == 0.5
    assert rows[0]['reference_status'] == 'provisional'
    manifest = json.loads((run_dir / 'config.json').read_text())
    assert manifest['image_sha256'] and manifest['reference_sha256'] and manifest['model_sha256']
    assert (run_dir / 'raw' / '1.page' / 'segmentation_overlay.png').is_file()


@pytest.mark.parametrize('operation', ['grayscale', 'contrast', 'clahe', 'denoise',
                                     'illumination', 'upscale', 'otsu'])
def test_processing_outputs_valid_pixels(operation: str) -> None:
    image = Image.new('RGB', (100, 100), (180, 160, 120))
    result = process_image(image, operation, PreprocessingConfig())
    assert result.mode in ('L', 'RGB') and np.asarray(result).dtype == np.uint8
