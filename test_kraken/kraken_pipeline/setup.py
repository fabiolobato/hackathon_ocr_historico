import argparse
import importlib.util
import logging
import os
import subprocess
import sys
import json
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_notebook(root: Path = Path.cwd()) -> None:
    cache = root / 'outputs' / '.cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache / 'matplotlib'))
    os.environ.setdefault('IPYTHONDIR', str(cache / 'ipython'))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('kraken').setLevel(logging.WARNING)


def install_dependencies(requirements: Path = Path('requirements.txt')) -> None:
    if not requirements.is_file():
        raise FileNotFoundError(f'Open the project directory first: missing {requirements}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)], check=True)


def dependencies_ready(requirements: Path = Path('requirements.txt')) -> bool:
    from packaging.requirements import Requirement

    for line in requirements.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            return False
        if requirement.url:
            source = metadata.distribution(requirement.name).read_text('direct_url.json')
            commit = json.loads(source or '{}').get('vcs_info', {}).get('commit_id')
            if commit != requirement.url.rsplit('@', 1)[-1]:
                return False
        elif installed not in requirement.specifier:
            return False
    return True


def ensure_dependencies(requirements: Path = Path('requirements.txt')) -> None:
    """Install missing pins and reject stale binary modules before initialization."""
    from packaging.version import Version

    if not dependencies_ready(requirements):
        install_dependencies(requirements)
    modules = {'numpy': 'numpy', 'scipy': 'scipy', 'torch': 'torch',
               'torchvision': 'torchvision', 'PIL': 'Pillow', 'matplotlib': 'matplotlib'}
    stale = []
    for module_name, package in modules.items():
        module = sys.modules.get(module_name)
        loaded = getattr(module, '__version__', None)
        if loaded is not None and Version(loaded).public != Version(metadata.version(package)).public:
            stale.append(f'{package}: loaded {loaded}, installed {metadata.version(package)}')
    if stale:
        raise RuntimeError('As dependências estão instaladas, mas há versões antigas na memória. '
                           'Reinicie a sessão do Colab e execute desde a primeira célula. '
                           'Os arquivos e pacotes instalados são mantidos. ' + '; '.join(stale))


def environment_info() -> dict[str, str | bool]:
    import torch
    from importlib.metadata import version

    return {'python': sys.version.split()[0], 'interpreter': sys.executable,
            'kraken': version('kraken'), 'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'in_colab': importlib.util.find_spec('google.colab') is not None
                if importlib.util.find_spec('google') is not None else False}


def default_device() -> str:
    import torch

    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main() -> None:
    parser = argparse.ArgumentParser(description='Install the pinned Kraken notebook environment.')
    parser.add_argument('--requirements', type=Path, default=Path('requirements.txt'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    install_dependencies(args.requirements)
    logger.info('Environment: %s', environment_info())


if __name__ == '__main__':
    main()
