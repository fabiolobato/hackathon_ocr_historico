"""Blocking Colab job: setup -> ketos fine-tune -> metrics JSON -> zip -> hold the kernel until fetched.

A detached process does not count as kernel activity, so the VM was reclaimed after the first run
finished. This script keeps the kernel busy until /content/<RUN>/FETCHED exists (or HOLD_HOURS pass).
Edit the constants before `colab exec -f` (the CLI sends the file content; no argparse).
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RUN = 'ft_v2'
DATA_ZIP = '/content/v1.zip'
ARGS = ['-B', '4', '--accumulate-grad-batches', '4', '-r', '1e-4', '--augment', '-q', 'fixed', '-N', '100',
        '-p', '0.9', '-u', 'NFD', '--resize', 'union']
HOLD_HOURS = 6
KRAKEN = 'kraken @ git+https://github.com/mittagessen/kraken.git@bb487219e7d8289274443d1ad0989efb4924dc2a'
MEDIUM_URL = 'https://zenodo.org/records/21788410/files/medium.safetensors?download=1'
MEDIUM_MD5 = 'e3411a453ce3b9e9efae3f8631d85762'
WEIGHTS = Path('/content/models/medium.safetensors')
OUT = Path('/content') / RUN


def setup() -> Path:
    if shutil.which('ketos') is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', KRAKEN, 'transformers==4.57.1',
                        'tensorboard'], check=True)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists():
        urllib.request.urlretrieve(MEDIUM_URL, WEIGHTS)
    assert hashlib.md5(WEIGHTS.read_bytes(), usedforsecurity=False).hexdigest() == MEDIUM_MD5
    data = Path('/content') / Path(DATA_ZIP).stem
    subprocess.run(['unzip', '-oq', DATA_ZIP, '-d', str(data)], check=True)
    return data


def train(data: Path) -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'tb').mkdir(parents=True)  # ketos requires an existing --log-dir
    cmd =['ketos', '-d', 'cuda:0', '-s', '42', '--workers', '2', 'train', '-f', 'page', '-i', str(WEIGHTS),
           '-o', str(OUT / 'model'), '--logger', 'tensorboard', '--log-dir', str(OUT / 'tb'), *ARGS,
           *sorted(str(p) for p in data.glob('*.xml'))]
    (OUT / 'cmd.txt').write_text(' '.join(cmd))
    with open(OUT / 'train.log', 'w') as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def export_metrics() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    metrics = {}
    for events in (OUT / 'tb').rglob('events.out.tfevents.*'):
        acc = EventAccumulator(str(events))
        acc.Reload()
        for tag in acc.Tags().get('scalars', []):
            metrics[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=1))


def main() -> None:
    code = train(setup())
    export_metrics()
    (OUT / 'files.txt').write_text('\n'.join(str(p) for p in OUT.rglob('*') if p.is_file()))
    shutil.make_archive(str(OUT / 'result'), 'zip', OUT, 'model')
    (OUT / 'READY').write_text(str(code))
    deadline = time.time() + HOLD_HOURS * 3600
    while time.time() < deadline and not (OUT / 'FETCHED').exists():
        time.sleep(30)


main()
