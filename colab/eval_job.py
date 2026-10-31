"""Colab job: score recognizers x context LMs with the final configuration on the 9 tuning pages (GPU), so the local
CPU stays free. Runs in the self-training VM after its job (repo unpacked in /content/repo, trained models in
/content/<RUN>/model_<init>_b*/best_*.safetensors). Uploaded beforehand into /content/repo: the updated
ocr_pipeline/engine.py, configs/final_v4.json, the lexicon/LM it names plus the LM variants, imagens/ and
test_kraken/ground_truth/. Held-out pages are never scored here.
Edit the constants before `colab exec -f` (the CLI sends the file content; no argparse).
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
RUN = 'st_g1'
REPO = Path('/content/repo')
OUT = Path('/content') / RUN / 'eval_tuning.json'
TRAINED = {'st_g1m': 'medium', 'st_g1s': 'st_v3'}  # registry name -> init tag of the training run
MODELS = ['st_v3', 'st_g1m', 'st_g1s']
LMS = {'w4': 'models/lexicon/pt_hist_legal_w4.ngrams.npz', 'livro8': 'models/lexicon/pt_hist_legal_w4_livro8.ngrams.npz'}
WORKERS = min(os.cpu_count() or 2, 8)  # ~2 GB of GPU each (E14: 12 ran out of memory on the L4)


def install_trained() -> None:
    for name, init in TRAINED.items():
        best = sorted((Path('/content') / RUN).glob(f'model_{init}_b*/best_*.safetensors'))
        assert len(best) == 1, best
        target = REPO / 'models/finetuned' / f'{name}.safetensors'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best[0], target)
        print(name, best[0].name, flush=True)


def main() -> None:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    install_trained()
    from ocr_pipeline.config import load_config, with_overrides
    from ocr_pipeline.data import discover, select, split
    from ocr_pipeline.engine import Engine
    from ocr_pipeline.pipeline import evaluate, page_executor

    pages_all = discover(Path('imagens'), Path('test_kraken/ground_truth'))
    pages, pool = select(pages_all, 'tuning'), split(pages_all)[0]
    final = load_config(Path('configs/final_v4.json'))
    engine = Engine(Path('/content/evalcache'), 'cuda:0', 42)
    out = {}
    with page_executor(engine, WORKERS) as executor:
        for model in MODELS:
            for lm, path in LMS.items():
                result = evaluate(engine, pages, with_overrides(final, {'rec.model': model, 'post.context_lm': path}),
                                  pool, executor)
                c = result['corpus']
                out[f'{model}/{lm}'] = {'wer': round(100 * c['wer_norm'], 2), 'cer': round(100 * c['cer_norm'], 2),
                                        'pages': [round(100 * p['wer_norm'], 1) for p in result['pages']]}
                print(model, lm, out[f'{model}/{lm}'], flush=True)
                OUT.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
