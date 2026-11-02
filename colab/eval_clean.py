"""Score recognizers with configs/final_v5.json on the 9 tuning pages, against the official GT and the Gemini
reference (E16). The clean bundle carries only the tuning pages (no held-out page, no Gemini test page), so nothing
here can look at them. Runs on the Colab GPU after colab/clean_train.py, from /content/repo.

  python colab/eval_clean.py OUT_JSON MODEL [MODEL ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
REFS = {'oficial': Path('test_kraken/ground_truth'), 'gemini': Path('test_kraken/ground_truth_gemini')}
TUNING = {5, 6, 8, 9, 11, 13, 14, 15, 16}
WORKERS = min(os.cpu_count() or 2, 8)  # ~2 GB of GPU each (E14: 12 ran out of memory on the L4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', type=Path)
    parser.add_argument('models', nargs='+')
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    from ocr_pipeline.config import load_config, with_overrides
    from ocr_pipeline.data import discover
    from ocr_pipeline.engine import Engine
    from ocr_pipeline.pipeline import evaluate, page_executor

    sets = {ref: discover(Path('imagens'), gt) for ref, gt in REFS.items()}
    for ref, pages in sets.items():
        assert {p.number for p in pages} == TUNING, (ref, [p.number for p in pages])
    final = load_config(Path('configs/final_v5.json'))
    engine = Engine(Path('/content/evalcache'), 'cuda:0', 42)
    out = {}
    with page_executor(engine, WORKERS) as executor:
        for model in args.models:
            config = with_overrides(final, {'rec.model': model})
            for ref, pages in sets.items():
                result = evaluate(engine, pages, config, sets['oficial'], executor)
                c = result['corpus']
                out[f'{model}/{ref}'] = {'wer': round(100 * c['wer_norm'], 2), 'cer': round(100 * c['cer_norm'], 2),
                                         'pages': [round(100 * p['wer_norm'], 1) for p in result['pages']]}
                print(model, ref, out[f'{model}/{ref}'], flush=True)
                args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
