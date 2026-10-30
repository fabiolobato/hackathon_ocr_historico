"""Blocking Colab job for self-training: pseudo-labels on the GPU VM -> select -> compile -> fine-tune -> hold.

Bundle (/content/selftrain.zip, built by colab/pack_selftrain.sh): ocr_pipeline/, the labelling config, the lexicon
and LM it names, and the cropped extra pages of the same book (never evaluation pages).
GPU use: labelling runs one pipeline process per CPU core, each with its own model on the GPU (recognition on
cuda, post-processing on the cores); training compiles the lines to an Arrow file once, uses bf16 mixed precision
and the largest batch that fits (OOM -> halve it, gradient accumulation keeps the effective batch at EFFECTIVE).
Keeps the kernel busy until /content/<RUN>/FETCHED exists, as in train_and_hold.py (an idle VM is reclaimed).
Edit the constants before `colab exec -f` (the CLI sends the file content; no argparse).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RUN = 'st_g1'
BUNDLE = Path('/content/selftrain.zip')
CONFIG = 'configs/v4_pass1_st_v3.json'  # labels from the pipeline with st_v3; training still starts from medium
IMAGES = 'imagens_extra/ldl1_crop'
# recalibrated for st_v3: 0.92 keeps 64% of tuning lines at label CER 9.6 (same noise as round 3; 0.90 would be 10.3)
SELECT = ['--min-chars', '10', '--min-mean-prob', '0.92', '--max-change', '1.0', '--min-known', '0.9']
# Gemini transcriptions of pages 57-104 (never evaluation pages): those pages train on the aligned transcript lines
# within 0.4 normalized edit distance of the reading (beyond that: table columns split by the segmenter, misalignment)
TRANSCRIPTS = 'imagens_extra/transcriptions_txt'
SELECT += ['--max-transcript-cer', '0.4']
INITS = {'medium': 'models/10.5281_zenodo.21788410/medium.safetensors', 'st_v3': 'models/finetuned/st_v3.safetensors'}
EPOCHS = 20
EFFECTIVE = 16  # lines per optimizer step (E07: 4 x 4)
BATCHES = (32, 16, 8, 4)
TRAIN = ['-r', '1e-4', '--augment', '-q', 'fixed', '-N', str(EPOCHS), '-p', '0.9', '-u', 'NFD', '--resize', 'union']
HOLD_HOURS = 6
KRAKEN = 'kraken @ git+https://github.com/mittagessen/kraken.git@bb487219e7d8289274443d1ad0989efb4924dc2a'
PACKAGES = [KRAKEN, 'transformers==4.57.1', 'rapidfuzz==3.14.1', 'opencv-python-headless==4.12.0.88', 'tensorboard']
MEDIUM_URL = 'https://zenodo.org/records/21788410/files/medium.safetensors?download=1'
MEDIUM_MD5 = 'e3411a453ce3b9e9efae3f8631d85762'
REPO = Path('/content/repo')
OUT = Path('/content') / RUN
CORES = os.cpu_count() or 2


def log(message: str) -> None:
    line = f'{time.strftime("%H:%M:%S")} {message}'
    print(line, flush=True)
    with open(OUT / 'job.log', 'a') as stream:
        stream.write(line + '\n')


def setup() -> Path:
    if shutil.which('ketos') is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PACKAGES], check=True)
    subprocess.run(['unzip', '-oq', str(BUNDLE), '-d', str(REPO)], check=True)
    weights = REPO / 'models/10.5281_zenodo.21788410/medium.safetensors'  # where engine.MODELS looks
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        urllib.request.urlretrieve(MEDIUM_URL, weights)
    assert hashlib.md5(weights.read_bytes(), usedforsecurity=False).hexdigest() == MEDIUM_MD5
    return weights


def run(cmd: list[str], name: str) -> int:
    log(f'{name}: {" ".join(cmd)[:300]}')
    with open(OUT / f'{name}.log', 'w') as stream:
        code = subprocess.run(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT).returncode
    log(f'{name}: exit {code}')
    return code


def pseudo_labels() -> Path:
    labels = OUT / 'labels'
    py = [sys.executable, '-m', 'ocr_pipeline.tuning.pseudo_labels']
    if not (labels / 'label_config.json').exists():  # labelling finished in an earlier run of this job
        assert run([*py, 'label', '--config', CONFIG, '--images', IMAGES, '--out', str(labels), '--device', 'cuda:0',
                    '--workers', str(CORES)], 'label') == 0
    assert run([sys.executable, '-m', 'ocr_pipeline.tuning.transcripts', str(labels), '--transcripts', TRANSCRIPTS],
               'transcripts') == 0
    assert run([*py, 'select', str(labels), *SELECT], 'select') == 0
    arrow = OUT / 'train.arrow'
    assert run(['ketos', '--workers', str(CORES), 'compile', '-f', 'page', '-o', str(arrow),
                *sorted(str(p) for p in (labels / 'train').glob('*.xml'))], 'compile') == 0
    return arrow


def train(weights: Path, arrow: Path, tag: str) -> int:
    for batch in BATCHES:
        model = OUT / f'model_{tag}_b{batch}'
        tb = OUT / f'tb_{tag}_b{batch}'
        tb.mkdir(parents=True, exist_ok=True)  # ketos requires an existing --log-dir
        cmd = ['ketos', '-d', 'cuda:0', '-s', '42', '--workers', str(min(CORES, 8)), '--threads', str(CORES),
               '--precision', 'bf16-mixed', 'train', '-f', 'binary', '-i', str(weights), '-o', str(model), '--logger', 'tensorboard',
               '--log-dir', str(tb), '-B', str(batch), '--accumulate-grad-batches', str(max(1, EFFECTIVE // batch)),
               *TRAIN, str(arrow)]
        code = run(cmd, f'train_{tag}_b{batch}')
        if code == 0 or 'out of memory' not in (OUT / f'train_{tag}_b{batch}.log').read_text(errors='ignore'):
            (OUT / f'batch_{tag}.txt').write_text(str(batch))
            return code
        shutil.rmtree(model, ignore_errors=True)
    return 1


def export_metrics() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    metrics = {}
    for events in OUT.glob('tb_*/**/events.out.tfevents.*'):
        acc = EventAccumulator(str(events))
        acc.Reload()
        init = events.relative_to(OUT).parts[0].split('_b')[0].removeprefix('tb_')
        for tag in acc.Tags().get('scalars', []):
            metrics[f'{init}/{tag}'] = [(e.step, e.value) for e in acc.Scalars(tag)]
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=1))


def pack() -> None:
    """result.zip: logs, label table and selection, metrics and the best checkpoint (not every epoch's)."""
    stage = Path('/content/pack') / RUN
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    keep = [*OUT.glob('*.log'), *OUT.glob('*.json'), *OUT.glob('*.txt'), *OUT.glob('labels/*.json*'),
            *OUT.glob('model_*/*best*')]
    for path in keep:
        target = stage / path.relative_to(OUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.make_archive(str(OUT / 'result'), 'zip', stage)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)  # labels from an earlier run are reused
    for stale in ('READY', 'FETCHED', 'result.zip'):
        (OUT / stale).unlink(missing_ok=True)
    log(f'cores {CORES}')
    code = 1
    try:
        setup()
        arrow = pseudo_labels()
        codes = [train(REPO / path, arrow, tag) for tag, path in INITS.items()]  # same data, two starting points
        code = max(codes)
        export_metrics()
    finally:
        pack()
        (OUT / 'READY').write_text(str(code))
        log(f'READY {code}')
    deadline = time.time() + HOLD_HOURS * 3600
    while time.time() < deadline and not (OUT / 'FETCHED').exists():
        time.sleep(30)


main()
