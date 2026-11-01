"""Assemble the self-contained delivery folder (runtime code, final config, data, weights, notebook, docs).

  python tools/build_entrega.py --out entrega_ocr --execute
The folder is committed (recognizer, lexicons and LM included) so it can be browsed and run from a clone; only
cache/ is gitignored. Sources live in docs/entrega/ and the repository. The recognizer and lexicon/LM files are the
ones FINAL_CONFIG names; reports that are missing (e.g. in a fresh clone) are skipped.
--execute runs the notebook inside the folder so it opens with every output already rendered.
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ocr_pipeline.engine.MODELS

logger = logging.getLogger(__name__)
SOURCE_NOTEBOOK = Path('docs/entrega/apresentacao.ipynb')
FINAL_CONFIG = Path('configs/final_v5.json')
RUNTIME = ['__init__.py', 'batch.py', 'beam_decode.py', 'config.py', 'context_lm.py', 'ctc_rescore.py', 'data.py',
           'edit_costs.py', 'engine.py', 'metrics.py', 'page_crop.py', 'pipeline.py', 'postprocess.py',
           'preprocessing.py', 'presentation.py', 'run.py', 'visualize.py']
FILES = {  # destination (inside the folder) -> source (repository)
    'README.md': 'docs/entrega/README.md',
    'PROCESSO.md': 'docs/entrega/PROCESSO.md',
    'apresentacao.ipynb': str(SOURCE_NOTEBOOK),
    'novas_imagens.ipynb': 'docs/entrega/novas_imagens.ipynb',
    'requirements.txt': 'requirements.txt',
    'setup_pipeline.py': 'docs/entrega/setup_pipeline.py',
    'configs/final.json': str(FINAL_CONFIG),
}
OPTIONAL = {  # copied when present, otherwise skipped
    'relatorios/busca_gulosa.html': 'outputs/search/search_st__tuning/report.html',
}
RESOURCE_KEYS = ('lexicon', 'period_lexicon', 'context_lm', 'edit_costs')  # post.* paths the pipeline loads
TREES = {'imagens': ('imagens', '*.jpg'), 'ground_truth': ('test_kraken/ground_truth', '*.txt'),
         'ground_truth_gemini': ('test_kraken/ground_truth_gemini', '*.txt'),  # same 12 pages, Gemini reference
         'teste_gemini/imagens': ('imagens_extra/teste/imagens', '*.jpg'),  # 10 new pages, reference = Gemini
         'teste_gemini/referencias': ('imagens_extra/teste/referencias', '*.txt')}


def copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def resources(config: Path) -> list[Path]:
    """Recognizer weights and lexicon/LM files the final configuration names (not in git outside the folder)."""
    from ocr_pipeline.engine import MODELS

    data = json.loads(config.read_text(encoding='utf-8'))
    post = data['post']
    return [MODELS[data['rec']['model']], *(Path(post[k]) for k in RESOURCE_KEYS if post.get(k))]


def only_shipped_models(text: str, shipped: str) -> str:
    """Keep the published models and the shipped fine-tuned one; drop the other fine-tuned entries and their
    comments (the shipped one is self-trained, E14)."""
    def dropped(line: str) -> bool:
        if 'finetuned/' in line:
            return f"'{shipped}'" not in line
        return line.lstrip().startswith('#') and 'fine-tuned' in line

    return ''.join(line for line in text.splitlines(keepends=True) if not dropped(line))


def assemble(out: Path) -> None:
    for name in RUNTIME:
        copy(Path('ocr_pipeline') / name, out / 'ocr_pipeline' / name)
    engine = out / 'ocr_pipeline' / 'engine.py'
    model = json.loads(FINAL_CONFIG.read_text(encoding='utf-8'))['rec']['model']
    engine.write_text(only_shipped_models(engine.read_text(encoding='utf-8'), model), encoding='utf-8')
    for path in resources(FINAL_CONFIG):
        copy(path, out / path)
    for destination, source in FILES.items():
        copy(Path(source), out / destination)
    for destination, source in OPTIONAL.items():
        if Path(source).is_file():
            copy(Path(source), out / destination)
        else:
            logger.warning('skipped missing %s', source)
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
        cache = args.out / 'cache'
        kept = args.out.parent / f'.{args.out.name}_cache'
        if cache.is_dir():  # recognition cache is keyed by config hash, safe to reuse
            shutil.move(cache, kept)
        shutil.rmtree(args.out)
    assemble(args.out)
    kept = args.out.parent / f'.{args.out.name}_cache'
    if kept.is_dir():
        shutil.move(kept, args.out / 'cache')
    if args.execute:
        execute(args.out.resolve(), args.timeout)


if __name__ == '__main__':
    main()
