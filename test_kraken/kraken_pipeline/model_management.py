import subprocess
import sys
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZenodoModel:
    doi: str
    filename: str
    md5: str

    @property
    def url(self) -> str:
        record = self.doi.rsplit('.', 1)[-1]
        return f'https://zenodo.org/records/{record}/files/{self.filename}?download=1'


PORTUGUESE_MODEL = ZenodoModel(
    doi='10.5281/zenodo.14543904',
    filename='traprinq.mlmodel',
    md5='b7f84ede75a0665c275346d94fdc1b95',
)

MULTILINGUAL_MODEL = ZenodoModel(
    doi='10.5281/zenodo.21788405',
    filename='small.safetensors',
    md5='07279cd68cd4402807542799f12d0f4a',
)
MULTILINGUAL_MEDIUM_MODEL = ZenodoModel(
    doi='10.5281/zenodo.21788410',
    filename='medium.safetensors',
    md5='e3411a453ce3b9e9efae3f8631d85762',
)


def _verify_model(path: Path, model: ZenodoModel) -> str:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(chunk)
            sha256.update(chunk)
    if md5.hexdigest() != model.md5:
        raise ValueError(f'Model checksum mismatch: {path}. Expected {model.md5}; got {md5.hexdigest()}.')
    return sha256.hexdigest()


def download_portuguese_model(output_dir: Path = Path('models'),
                              model: ZenodoModel = PORTUGUESE_MODEL) -> Path:
    """Retrieve pinned TraPrInq weights, checking Zenodo's checksum on every use."""
    destination = output_dir / model.doi.replace('/', '_')
    destination.mkdir(parents=True, exist_ok=True)
    weights = destination / model.filename
    if weights.exists():
        sha256 = _verify_model(weights, model)
        logger.info('Using verified cached model: %s', weights)
    else:
        temporary = weights.with_suffix(weights.suffix + '.part')
        logger.info('Downloading %s from %s', model.filename, model.url)
        with requests.get(model.url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                downloaded = 0
                reported = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - reported >= 16 * 1024 * 1024:
                            logger.info('Downloaded %.1f MiB of %s', downloaded / (1024 * 1024), model.filename)
                            reported = downloaded
        sha256 = _verify_model(temporary, model)
        temporary.replace(weights)
    provenance = {**asdict(model), 'url': model.url, 'sha256': sha256}
    (destination / 'download.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return weights


def discover_models(language: str = 'por') -> None:
    """Display model metadata; does not download recognition weights."""
    subprocess.run([sys.executable, '-c', 'from kraken.kraken import cli; cli()',
                    'list', '--recognition', '--language', language], check=True)


def download_multilingual_model(output_dir: Path = Path('models'),
                               variant: Literal['small', 'medium'] = 'small') -> Path:
    """Baixa o PP-OCRv6 multilíngue com português, verificando o MD5."""
    models = {'small': MULTILINGUAL_MODEL, 'medium': MULTILINGUAL_MEDIUM_MODEL}
    return download_portuguese_model(output_dir, models[variant])


def inspect_model(doi: str) -> None:
    subprocess.run([sys.executable, '-c', 'from kraken.kraken import cli; cli()',
                    'show', doi], check=True)


def download_model(doi: str, output_dir: Path = Path('models')) -> Path:
    """Download a user-selected version DOI into local model storage."""
    from htrmopo import get_model

    destination = output_dir / doi.replace('/', '_')
    if destination.exists():
        raise FileExistsError(f'Inspect existing model download before retrying: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(get_model(doi, destination))
