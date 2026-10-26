"""Assemble the self-contained delivery folder (runtime code, configs, data, weights, notebook, docs).

  python tools/build_entrega.py --out entrega_ocr --execute
The folder is gitignored: it holds model weights. Sources live in docs/entrega/ and the repository.
--execute runs the notebook inside the folder so it opens with every output already rendered.
"""
import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)
RUNTIME = ['__init__.py', 'config.py', 'ctc_rescore.py', 'data.py', 'engine.py', 'metrics.py', 'pipeline.py',
           'postprocess.py', 'preprocessing.py', 'presentation.py', 'run.py', 'visualize.py']
FILES = {  # destination (inside the folder) -> source (repository)
    'README.md': 'docs/entrega/README.md',
    'PROCESSO.md': 'docs/entrega/PROCESSO.md',
    'apresentacao.ipynb': 'docs/entrega/apresentacao.ipynb',
    'requirements.txt': 'requirements.txt',
    'setup_pipeline.py': 'colab/setup_pipeline.py',
    'models/10.5281_zenodo.21788410/medium.safetensors': 'models/10.5281_zenodo.21788410/medium.safetensors',
    'models/finetuned/ft_v1.safetensors': 'models/finetuned/ft_v1.safetensors',
    'models/lexicon/pt_br_50k.txt': 'models/lexicon/pt_br_50k.txt',
    'relatorios/busca_gulosa.html': 'outputs/search/search_v1__tuning/report.html',
    'relatorios/reservadas_v2_v4_v3.html': 'outputs/viz/heldout_v2_v4_ft.html',
}
TREES = {'imagens': ('imagens', '*.jpg'), 'ground_truth': ('test_kraken/ground_truth', '*.txt'),
         'configs': ('configs', '*.json')}


def copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def assemble(out: Path) -> None:
    for name in RUNTIME:
        copy(Path('ocr_pipeline') / name, out / 'ocr_pipeline' / name)
    for destination, source in FILES.items():
        copy(Path(source), out / destination)
    for destination, (source, pattern) in TREES.items():
        for path in sorted(Path(source).glob(pattern)):
            copy(path, out / destination / path.name)
    logger.info('assembled %s', out)


def execute(out: Path, timeout: int) -> None:
    import nbformat
    from nbclient import NotebookClient

    path = out / 'apresentacao.ipynb'
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=timeout, kernel_name='python3', resources={'metadata': {'path': str(out)}}).execute()
    nbformat.write(notebook, path)
    logger.info('executed %s', path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('entrega_ocr'))
    parser.add_argument('--execute', action='store_true', help='run the notebook inside the folder')
    parser.add_argument('--timeout', type=int, default=1800, help='per-cell seconds when executing')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not Path('ocr_pipeline/run.py').is_file():
        raise SystemExit('run from the repository root')
    if args.out.exists():
        if not (args.out / 'apresentacao.ipynb').is_file():
            raise SystemExit(f'{args.out} exists and is not a delivery folder; refusing to delete it')
        shutil.rmtree(args.out)
    assemble(args.out)
    if args.execute:
        execute(args.out.resolve(), args.timeout)


if __name__ == '__main__':
    main()
