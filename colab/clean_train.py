"""Blocking Colab job for the clean chain (E16): no evaluation page and no Gemini test page ever reaches labelling,
training or model selection, and the first labeller is the published medium (no model that saw them).

Branch 'm' (Gemini chain): label the 78 training pages with the medium pipeline -> align the Gemini transcripts ->
  train st_c1m from medium -> relabel with the st_c1m pipeline -> realign -> train st_c2m from medium.
Branch 's' (self-training first): label the 81 clean pages (78 training + 3 untranscribed front/back pages) with the
  medium pipeline -> confident pseudo-labels (E14 round-1 filter) -> train st_c0 from medium -> align the Gemini
  transcripts on the same labels -> train st_c1s from st_c0.
Then colab/eval_clean.py scores the new models with configs/final_v5.json on the 9 tuning pages only (official GT and
Gemini reference). Bundle: colab/pack_clean.sh. Keeps the kernel busy until /content/<RUN>/FETCHED exists.
Edit BRANCH before `colab exec -f` (the CLI sends the file content; no argparse).
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

BRANCH = 'm'
RUN = f'st_c_{BRANCH}'
BUNDLE = Path('/content/clean.zip')
LABELLER = 'configs/v4_pass1.json'  # medium recognizer
TRANSCRIPTS = 'imagens_extra/transcricoes/treino'  # configs/gemini_split.json training pages only
GEM = ['--min-chars', '10', '--max-transcript-cer', '0.4', '--transcripts-only']
PSEUDO = ['--min-chars', '10', '--min-mean-prob', '0.85', '--max-change', '1.0', '--min-known', '0.9']  # E14 st_v1
MEDIUM = 'models/10.5281_zenodo.21788410/medium.safetensors'
EVAL = {'m': ['st_c1m', 'st_c2m', 'st_g3m'], 's': ['st_c0', 'st_c1s']}  # st_g3m: reference, same GPU numerics
EPOCHS = 20
EFFECTIVE = 16
BATCHES = (16, 8, 4)  # 32 ran out of memory on the L4 in every earlier round
TRAIN = ['-r', '1e-4', '--augment', '-q', 'fixed', '-N', str(EPOCHS), '-p', '0.9', '-u', 'NFD', '--resize', 'union']
HOLD_HOURS = 8
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


def setup() -> None:
    if shutil.which('ketos') is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PACKAGES], check=True)
    subprocess.run(['unzip', '-oq', str(BUNDLE), '-d', str(REPO)], check=True)
    weights = REPO / MEDIUM
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        urllib.request.urlretrieve(MEDIUM_URL, weights)
    assert hashlib.md5(weights.read_bytes(), usedforsecurity=False).hexdigest() == MEDIUM_MD5


def run(cmd: list[str], name: str) -> int:
    log(f'{name}: {" ".join(cmd)[:300]}')
    with open(OUT / f'{name}.log', 'w') as stream:
        code = subprocess.run(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT).returncode
    log(f'{name}: exit {code}')
    return code


def label(config: str, images: str, name: str) -> Path:
    labels = OUT / name
    if not (labels / 'label_config.json').exists():  # finished in an earlier run of this job
        assert run([sys.executable, '-m', 'ocr_pipeline.tuning.pseudo_labels', 'label', '--config', config, '--images',
                    images, '--out', str(labels), '--device', 'cuda:0', '--workers', str(CORES)], f'label_{name}') == 0
    return labels


def selected(labels: Path, thresholds: list[str], tag: str, transcripts: bool) -> Path:
    """Select lines into labels/train and compile them to train_<tag>.arrow (select overwrites labels/train)."""
    if transcripts:
        assert run([sys.executable, '-m', 'ocr_pipeline.tuning.transcripts', str(labels), '--transcripts',
                    TRANSCRIPTS], f'transcripts_{tag}') == 0
    assert run([sys.executable, '-m', 'ocr_pipeline.tuning.pseudo_labels', 'select', str(labels), *thresholds],
               f'select_{tag}') == 0
    shutil.copy2(labels / 'select.json', OUT / f'select_{tag}.json')
    arrow = OUT / f'train_{tag}.arrow'
    assert run(['ketos', '--workers', str(CORES), 'compile', '-f', 'page', '-o', str(arrow),
                *map(str, sorted((labels / 'train').glob('*.xml')))], f'compile_{tag}') == 0
    return arrow


def train(init: Path, arrow: Path, name: str) -> None:
    """Fine-tune init on arrow and install the best checkpoint as models/finetuned/<name>.safetensors."""
    for batch in BATCHES:
        model, tb = OUT / f'model_{name}_b{batch}', OUT / f'tb_{name}_b{batch}'
        tb.mkdir(parents=True, exist_ok=True)  # ketos requires an existing --log-dir
        cmd = ['ketos', '-d', 'cuda:0', '-s', '42', '--workers', str(min(CORES, 8)), '--threads', str(CORES),
               '--precision', 'bf16-mixed', 'train', '-f', 'binary', '-i', str(init), '-o', str(model), '--logger',
               'tensorboard', '--log-dir', str(tb), '-B', str(batch), '--accumulate-grad-batches',
               str(max(1, EFFECTIVE // batch)), *TRAIN, str(arrow)]
        code = run(cmd, f'train_{name}_b{batch}')
        if code == 0:
            best = sorted(model.glob('*best*.safetensors'))
            assert len(best) == 1, best
            target = REPO / 'models/finetuned' / f'{name}.safetensors'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best[0], target)
            log(f'{name}: {best[0].name} sha256 {hashlib.sha256(target.read_bytes()).hexdigest()}')
            return
        assert 'out of memory' in (OUT / f'train_{name}_b{batch}.log').read_text(errors='ignore'), code
        shutil.rmtree(model, ignore_errors=True)
    raise RuntimeError(f'{name}: out of memory at every batch size')


def branch_m() -> None:
    first = label(LABELLER, 'imagens_extra/treino_crop', 'labels_r1')
    train(REPO / MEDIUM, selected(first, GEM, 'gem_r1', True), 'st_c1m')
    second = label('configs/v4_pass1_st_c1m.json', 'imagens_extra/treino_crop', 'labels_r2')
    train(REPO / MEDIUM, selected(second, GEM, 'gem_r2', True), 'st_c2m')


def branch_s() -> None:
    labels = label(LABELLER, 'imagens_extra/ldl1_clean', 'labels_r1')
    train(REPO / MEDIUM, selected(labels, PSEUDO, 'pseudo', False), 'st_c0')
    train(REPO / 'models/finetuned/st_c0.safetensors', selected(labels, GEM, 'gem_r1', True), 'st_c1s')


def export_metrics() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    metrics = {}
    for events in OUT.glob('tb_*/**/events.out.tfevents.*'):
        acc = EventAccumulator(str(events))
        acc.Reload()
        name = events.relative_to(OUT).parts[0].rsplit('_b', 1)[0].removeprefix('tb_')
        for tag in acc.Tags().get('scalars', []):
            metrics[f'{name}/{tag}'] = [(e.step, e.value) for e in acc.Scalars(tag)]
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=1))


def pack() -> None:
    """result.zip: logs, selections, label tables, metrics and the installed best checkpoints."""
    stage = Path('/content/pack') / RUN
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    keep = [*OUT.glob('*.log'), *OUT.glob('*.json'), *OUT.glob('labels_*/*.json*')]
    for path in keep:
        target = stage / path.relative_to(OUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in EVAL[BRANCH]:
        weights = REPO / 'models/finetuned' / f'{name}.safetensors'
        if weights.is_file() and name != 'st_g3m':
            shutil.copy2(weights, stage / weights.name)
    shutil.make_archive(str(OUT / 'result'), 'zip', stage)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)  # labels from an earlier run are reused
    for stale in ('READY', 'FETCHED', 'result.zip'):
        (OUT / stale).unlink(missing_ok=True)
    log(f'branch {BRANCH}, cores {CORES}')
    code = 1
    try:
        setup()
        {'m': branch_m, 's': branch_s}[BRANCH]()
        code = run([sys.executable, 'colab/eval_clean.py', str(OUT / 'eval_tuning.json'), *EVAL[BRANCH]], 'eval')
        export_metrics()
    finally:
        pack()
        (OUT / 'READY').write_text(str(code))
        log(f'READY {code}')
    deadline = time.time() + HOLD_HOURS * 3600
    while time.time() < deadline and not (OUT / 'FETCHED').exists():
        time.sleep(30)


main()
