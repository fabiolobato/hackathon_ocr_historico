import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kraken_pipeline import reporting


def make_run(root: Path, name: str, variant: str, wer: float) -> Path:
    directory = root / name
    directory.mkdir()
    manifest = {key: {} for key in ('model_sha256', 'image_sha256', 'reference_sha256', 'pipeline_sha256')}
    manifest['config'] = {'scoring': {}, 'seed': 42}
    (directory / 'config.json').write_text(json.dumps(manifest))
    row = {'variant': variant, 'wer': wer, 'complete': True}
    (directory / 'summary.json').write_text(json.dumps([row]))
    return directory


def test_compare_runs_uses_same_pages_and_recalculates_delta(tmp_path: Path) -> None:
    baseline = make_run(tmp_path, 'control', 'kraken_neural', 0.8)
    candidate = make_run(tmp_path, 'candidate', 'denoise', 0.6)
    with patch.object(reporting, 'show_comparison', side_effect=lambda path: json.loads((path / 'summary.json').read_text())):
        rows = reporting.compare_runs((baseline, candidate))
    assert rows[1]['delta_wer'] == pytest.approx(-0.2)
    assert rows[1]['baseline_variant'] == 'kraken_neural'
    manifest = json.loads((candidate / 'config.json').read_text())
    manifest['image_sha256'] = {'different.jpg': 'hash'}
    (candidate / 'config.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='image_sha256'):
        reporting.compare_runs((baseline, candidate))


def test_compare_runs_rejects_duplicate_variants(tmp_path: Path) -> None:
    first = make_run(tmp_path, 'first', 'raw', 0.6)
    second = make_run(tmp_path, 'second', 'raw', 0.5)
    with pytest.raises(ValueError, match='repetidas'):
        reporting.compare_runs((first, second))
