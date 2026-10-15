import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from kraken_pipeline.model_management import ZenodoModel, download_portuguese_model


def model_for(payload: bytes) -> ZenodoModel:
    return ZenodoModel('10.5281/zenodo.14543904', 'traprinq.mlmodel',
                       hashlib.md5(payload, usedforsecurity=False).hexdigest())


def test_download_verifies_weights_and_reuses_cache(tmp_path: Path) -> None:
    payload = b'model bytes'
    model = model_for(payload)
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [payload[:4], b'', payload[4:]]
    with patch('kraken_pipeline.model_management.requests.get', return_value=response) as get:
        weights = download_portuguese_model(tmp_path, model)
        assert weights.read_bytes() == payload
        assert download_portuguese_model(tmp_path, model) == weights
        get.assert_called_once_with(model.url, stream=True, timeout=(15, 120))
    provenance = json.loads((weights.parent / 'download.json').read_text())
    assert provenance['sha256'] == hashlib.sha256(payload).hexdigest()
    assert provenance['doi'] == model.doi


def test_corrupt_download_is_not_promoted(tmp_path: Path) -> None:
    model = model_for(b'correct')
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [b'wrong']
    with patch('kraken_pipeline.model_management.requests.get', return_value=response):
        with pytest.raises(ValueError, match='checksum mismatch'):
            download_portuguese_model(tmp_path, model)
    assert not list(tmp_path.rglob('*.mlmodel'))


def test_corrupt_cache_is_not_reused_or_overwritten(tmp_path: Path) -> None:
    model = model_for(b'correct')
    destination = tmp_path / model.doi.replace('/', '_')
    destination.mkdir()
    weights = destination / model.filename
    weights.write_bytes(b'wrong')
    with patch('kraken_pipeline.model_management.requests.get') as get:
        with pytest.raises(ValueError, match='checksum mismatch'):
            download_portuguese_model(tmp_path, model)
        get.assert_not_called()
    assert weights.read_bytes() == b'wrong'


def test_http_failure_does_not_create_weights(tmp_path: Path) -> None:
    response = MagicMock()
    response.__enter__.return_value = response
    response.raise_for_status.side_effect = requests.HTTPError('503')
    with patch('kraken_pipeline.model_management.requests.get', return_value=response):
        with pytest.raises(requests.HTTPError):
            download_portuguese_model(tmp_path, model_for(b'correct'))
    assert not list(tmp_path.rglob('*.mlmodel'))
