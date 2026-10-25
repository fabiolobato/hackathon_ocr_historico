"""Colab (free tier) setup for the main pipeline only; tuning needs requirements-dev.txt and is not set up here.

Run from the repository root:
  !python colab/setup_pipeline.py --ft-weights /content/drive/MyDrive/ocr/ft_v1.safetensors
  !python -m ocr_pipeline.run --config configs/best_v3_ft.json --pages heldout --device cuda:0
Installs Kraken and the text deps on top of Colab's preinstalled torch (never reinstalls torch), downloads
the PP-OCRv6 medium weights and the pt-BR lexicon, and installs ft_v1 from a path you provide (it is not public).
Every file is hash-checked. Without ft_v1, use configs/best_v4.json.
"""
import argparse
import hashlib
import logging
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
KRAKEN = 'kraken @ git+https://github.com/mittagessen/kraken.git@bb487219e7d8289274443d1ad0989efb4924dc2a'
PACKAGES = [KRAKEN, 'transformers==4.57.1', 'rapidfuzz==3.14.1']
DOWNLOADS = {  # path (as engine.MODELS / configs expect) -> (url, algorithm, digest)
    Path('models/10.5281_zenodo.21788410/medium.safetensors'):
        ('https://zenodo.org/records/21788410/files/medium.safetensors?download=1', 'md5',
         'e3411a453ce3b9e9efae3f8631d85762'),
    Path('models/lexicon/pt_br_50k.txt'):
        ('https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/pt_br/pt_br_50k.txt',
         'sha256', 'a61d6f2ede97c5daad5fb3907b72a228f0aff12668be6d24da590d20804fa611'),
}
FT_V1 = (Path('models/finetuned/ft_v1.safetensors'), 'sha256',
         '986c7670905e54ae999de983640836ea9291dbc8040435e251a06cc4d3fea72f')


def digest(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, path.read_bytes(), usedforsecurity=False).hexdigest()


def fetch(path: Path, url: str, algorithm: str, expected: str) -> None:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info('downloading %s', path)
        urllib.request.urlretrieve(url, path)
    actual = digest(path, algorithm)
    if actual != expected:
        raise ValueError(f'{path}: {algorithm} {actual} != {expected}; delete it and rerun')
    logger.info('ok %s', path)


def install_ft(source: Path | None) -> bool:
    target, algorithm, expected = FT_V1
    if source is not None and not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if not target.is_file():
        logger.warning('ft_v1 missing: configs/best_v3_ft.json will not run; use configs/best_v4.json '
                       'or pass --ft-weights')
        return False
    actual = digest(target, algorithm)
    if actual != expected:
        raise ValueError(f'{target}: sha256 {actual} != {expected}')
    logger.info('ok %s', target)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ft-weights', type=Path, default=None, help='path to ft_v1.safetensors (e.g. on Drive)')
    parser.add_argument('--skip-install', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if not Path('ocr_pipeline/run.py').is_file():
        raise SystemExit('run from the repository root')
    if not args.skip_install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PACKAGES], check=True)
    for path, (url, algorithm, expected) in DOWNLOADS.items():
        fetch(path, url, algorithm, expected)
    install_ft(args.ft_weights)


if __name__ == '__main__':
    main()
