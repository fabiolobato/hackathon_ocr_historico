"""Blocking Colab job for a (reduced) greedy search on the tuning pages -> hold.

Bundle (/content/search.zip, built by colab/pack_search.sh): ocr_pipeline/, the spec, the --init decisions, the
recognizer weights and the lexicon/LM the initial configuration names, the 12 page images and their GT (the search
only scores tuning pages; held-out GT is never read by the search, it only defines the split).
Parallelism: one process per CPU core, each with its own model on the GPU; every candidate of a step is evaluated in
one batch (Search.prefetch), so the cores stay busy. Keeps the kernel busy until /content/<RUN>/FETCHED exists.
Edit the constants before `colab exec -f` (the CLI sends the file content; no argparse).
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RUN = 'search_st'
BUNDLE = Path('/content/search.zip')
SPEC = 'sweeps/search_st.json'
INIT = 'outputs/search/init_v4_pass2.json'  # greedy v4 pass 2 (medium); pass 3 only revisited frozen stages
PASSES = 10
HOLD_HOURS = 6
KRAKEN = 'kraken @ git+https://github.com/mittagessen/kraken.git@bb487219e7d8289274443d1ad0989efb4924dc2a'
PACKAGES = [KRAKEN, 'transformers==4.57.1', 'rapidfuzz==3.14.1', 'opencv-python-headless==4.12.0.88']
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


def search() -> int:
    cmd = [sys.executable, '-m', 'ocr_pipeline.tuning.search', SPEC, '--pages', 'tuning', '--passes', str(PASSES),
           '--workers', str(CORES), '--device', 'cuda:0', '--init', INIT, '--out', str(OUT / 'search'),
           '--cache', str(OUT / 'cache')]
    log(f'search: {" ".join(cmd)}')
    with open(OUT / 'search.log', 'w') as stream:
        code = subprocess.run(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT).returncode
    log(f'search: exit {code}')
    return code


def pack() -> None:
    """result.zip: logs and the search folder (decisions, evaluations, best config, report), not the cache."""
    stage = Path('/content/pack') / RUN
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for path in [*OUT.glob('*.log'), *(OUT / 'search').rglob('*')]:
        if path.is_file():
            target = stage / path.relative_to(OUT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.make_archive(str(OUT / 'result'), 'zip', stage)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)  # an interrupted search resumes from its evaluations.jsonl
    for stale in ('READY', 'FETCHED', 'result.zip'):
        (OUT / stale).unlink(missing_ok=True)
    log(f'cores {CORES}')
    code = 1
    try:
        setup()
        code = search()
    finally:
        pack()
        (OUT / 'READY').write_text(str(code))
        log(f'READY {code}')
    deadline = time.time() + HOLD_HOURS * 3600
    while time.time() < deadline and not (OUT / 'FETCHED').exists():
        time.sleep(30)


main()
