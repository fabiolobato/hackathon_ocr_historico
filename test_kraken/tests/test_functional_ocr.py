from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from kraken_pipeline.config import INDIVIDUAL, COMBINED, PipelineConfig, PreprocessingConfig, RAW
from kraken_pipeline.experiments import run_page, summarize
from kraken_pipeline.tuning import BINARIZATION_SPACES, SearchConfig, generate_trials


def test_treatments_and_search_use_neural_segmentation_by_default() -> None:
    assert all(variant.segmenter == 'neural' for variant in INDIVIDUAL + COMBINED)
    assert all(trial.variant.segmenter == 'neural' for trial in generate_trials(PreprocessingConfig(), SearchConfig()))


def test_empty_segmentation_is_exposed_in_page_and_corpus(tmp_path: Path) -> None:
    @dataclass
    class Segmentation:
        lines: tuple = ()

    image = tmp_path / '1.page.jpg'
    Image.new('RGB', (20, 20), 'white').save(image)
    (tmp_path / '1.page.txt').write_text('palavra', encoding='utf-8')
    config = PipelineConfig(reference_dir=tmp_path)
    engine = SimpleNamespace(segment=lambda im: Segmentation(), recognize=lambda im, seg: [])
    row = run_page(config, engine, image, RAW, tmp_path / 'run')
    assert row['detected_lines'] == 0
    assert row['empty_output'] is True
    assert row['diagnostic'] == 'zero_lines'
    assert row['wer'] == 1
    summary = summarize([row], 1)[0]
    assert summary['nonempty_pages'] == 0
    assert summary['empty_pages'] == 1


def test_threshold_grid_and_rejection_of_legacy_mode() -> None:
    search = SearchConfig(method='grid', spaces=BINARIZATION_SPACES, max_trials=5)
    trials = generate_trials(PreprocessingConfig(), search)
    assert [trial.preprocessing.nlbin_threshold for trial in trials] == [0.2, 0.35, 0.5, 0.65, 0.8]
    with pytest.raises(ValueError, match='neural'):
        generate_trials(PreprocessingConfig(), SearchConfig(segmenter='legacy'))
    from kraken_pipeline.config import Variant
    with pytest.raises(ValueError, match='neural'):
        Variant('old', (), 'legacy')
