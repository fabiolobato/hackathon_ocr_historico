import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kraken_pipeline.reporting import show_page


def test_viewer_reports_original_failure_before_opening_images(tmp_path: Path) -> None:
    (tmp_path / 'results.json').write_text(json.dumps([
        {'page': '11.page.jpg', 'variant': 'raw', 'status': 'failed',
         'error': "RuntimeError('segmentation failed')"},
    ]))
    with patch('kraken_pipeline.reporting.plt.subplots') as subplots:
        with pytest.raises(RuntimeError, match='segmentation failed'):
            show_page(tmp_path, '11.page')
        subplots.assert_not_called()


def test_cuda_preflight_rejects_unavailable_gpu() -> None:
    from kraken_pipeline.engine import validate_device

    with patch('torch.cuda.is_available', return_value=False):
        with pytest.raises(RuntimeError, match='Enable a GPU runtime in Colab'):
            validate_device('cuda:0')
        validate_device('cpu')
