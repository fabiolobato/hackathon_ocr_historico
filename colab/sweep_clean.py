"""Blocking Colab job (E16): pipeline sweep around configs/final_v5.json with the clean recognizer, tuning pages only.

1. One change at a time (VARIANTS), each scored on the 9 tuning pages against the Gemini reference (criterion) and
   the official GT (check).
2. A change is accepted when it lowers the Gemini WER by at least ACCEPT points and does not raise the official WER.
3. Accepted changes are added greedily (largest gain first); each is kept only if the combination still gains at
   least KEEP points on the Gemini reference without losing on the official GT.
Bundle: colab/pack_clean.sh m plus the recognizer (only tuning pages travel). Writes /content/<RUN>/sweep.json and
final_overrides.json; keeps the kernel busy until /content/<RUN>/FETCHED exists.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
RUN = 'sweep_c1m'
BUNDLE = Path('/content/clean.zip')
MODEL = 'st_c1m'
ACCEPT, KEEP = 0.3, 0.15  # WER points; 0.3 is ~5 of the 1.772 tuning words
VARIANTS = {
    'pad16': {'rec.padding': 16}, 'pad32': {'rec.padding': 32},
    'no_deskew': {'geometric_ops': ['upscale']},
    'lm_w4': {'post.context_lm': 'models/lexicon/pt_hist_legal_w4.ngrams.npz'},
    'alpha8': {'post.ctc_alpha': 8}, 'alpha24': {'post.ctc_alpha': 24},
    'beta05': {'post.context_beta': 0.5}, 'beta15': {'post.context_beta': 1.5},
    'ratio04': {'post.lexicon_max_ratio': 0.4}, 'ratio06': {'post.lexicon_max_ratio': 0.6},
    'short40': {'post.context_short': 40}, 'short80': {'post.context_short': 80},
    'cand20': {'post.ctc_candidates': 20}, 'cand50': {'post.ctc_candidates': 50},
    'nbest3': {'post.context_nbest': 3}, 'nbest8': {'post.context_nbest': 8},
    'beam0': {'post.beam': 0}, 'beam128': {'post.beam': 128},
    'minlen3': {'post.lexicon_min_len': 3}, 'fold': {'post.fold_accents': True},
}
REFS = {'oficial': Path('test_kraken/ground_truth'), 'gemini': Path('test_kraken/ground_truth_gemini')}
TUNING = {5, 6, 8, 9, 11, 13, 14, 15, 16}
KRAKEN = 'kraken @ git+https://github.com/mittagessen/kraken.git@bb487219e7d8289274443d1ad0989efb4924dc2a'
PACKAGES = [KRAKEN, 'transformers==4.57.1', 'rapidfuzz==3.14.1', 'opencv-python-headless==4.12.0.88']
REPO = Path('/content/repo')
OUT = Path('/content') / RUN
WORKERS = min(os.cpu_count() or 2, 8)
HOLD_HOURS = 6


def log(message: str) -> None:
    line = f'{time.strftime("%H:%M:%S")} {message}'
    print(line, flush=True)
    with open(OUT / 'job.log', 'a') as stream:
        stream.write(line + '\n')


def setup() -> None:
    if shutil.which('ketos') is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PACKAGES], check=True)
    subprocess.run(['unzip', '-oq', str(BUNDLE), '-d', str(REPO)], check=True)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))


def scorer():
    from ocr_pipeline.config import load_config, with_overrides
    from ocr_pipeline.data import discover
    from ocr_pipeline.engine import Engine
    from ocr_pipeline.pipeline import evaluate, page_executor

    sets = {ref: discover(Path('imagens'), gt) for ref, gt in REFS.items()}
    for ref, pages in sets.items():
        assert {p.number for p in pages} == TUNING, ref
    base = with_overrides(load_config(Path('configs/final_v5.json')), {'rec.model': MODEL})
    engine = Engine(Path('/content/sweepcache'), 'cuda:0', 42)
    executor = page_executor(engine, WORKERS).__enter__()

    def score(overrides: dict) -> dict:
        config = with_overrides(base, overrides)
        out = {}
        for ref, pages in sets.items():
            result = evaluate(engine, pages, config, sets['oficial'], executor)
            c = result['corpus']
            out[ref] = {'wer': round(100 * c['wer_norm'], 2), 'cer': round(100 * c['cer_norm'], 2),
                        'pages': [round(100 * p['wer_norm'], 1) for p in result['pages']]}
        return out

    return score


def sweep(score) -> tuple[dict, dict]:
    results = {'base': score({})}
    log(f"base {results['base']['gemini']['wer']} / {results['base']['oficial']['wer']}")
    for name, overrides in VARIANTS.items():
        results[name] = score(overrides)
        log(f"{name} {results[name]['gemini']['wer']} / {results[name]['oficial']['wer']}")
        (OUT / 'sweep.json').write_text(json.dumps(results, indent=1))
    return results, combine(score, results)


def combine(score, results: dict) -> dict:
    base = results['base']
    gains = {n: base['gemini']['wer'] - r['gemini']['wer'] for n, r in results.items() if n != 'base'
             and base['gemini']['wer'] - r['gemini']['wer'] >= ACCEPT and r['oficial']['wer'] <= base['oficial']['wer']}
    current, chosen, best = {}, [], base
    for name in sorted(gains, key=gains.get, reverse=True):
        trial = {**current, **VARIANTS[name]}
        result = score(trial) if chosen else results[name]
        if best['gemini']['wer'] - result['gemini']['wer'] >= KEEP and result['oficial']['wer'] <= best['oficial']['wer']:
            current, best = trial, result
            chosen.append(name)
        log(f"combine +{name}: {result['gemini']['wer']} / {result['oficial']['wer']} -> {chosen}")
    final = {'accepted': gains, 'chosen': chosen, 'overrides': current, 'result': best}
    (OUT / 'final_overrides.json').write_text(json.dumps(final, indent=1))
    return final


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in ('READY', 'FETCHED', 'result.zip'):
        (OUT / stale).unlink(missing_ok=True)
    code = 1
    try:
        setup()
        sweep(scorer())
        code = 0
    finally:
        archive = shutil.make_archive(f'/content/{RUN}_result', 'zip', OUT)  # outside OUT: not inside itself
        shutil.move(archive, OUT / 'result.zip')
        (OUT / 'READY').write_text(str(code))
        log(f'READY {code}')
    deadline = time.time() + HOLD_HOURS * 3600
    while time.time() < deadline and not (OUT / 'FETCHED').exists():
        time.sleep(30)


main()
