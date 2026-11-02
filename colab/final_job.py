"""Blocking Colab job for the frozen final configuration (E16), so the local CPU stays free.

MODE 'score': the single look. tools/score_final.py with FINAL on the 12 evaluated pages against the official GT and
  the Gemini reference, and on the 10 Gemini test pages; writes /content/<RUN>/final_scores_*.json.
MODE 'entrega': tools/build_entrega.py --execute on the GPU (the notebook runs with every output rendered); zips the
  delivery folder without its cache.
Bundle (colab/pack_final.sh): the tracked repository, the recognizer and lexicon/LM files FINAL names, and the Gemini
test pages. Keeps the kernel busy until /content/<RUN>/FETCHED exists. Edit MODE before `colab exec -f`.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODE = 'score'
RUN = f'final_v6_{MODE}'
FINAL = 'configs/final_v6.json'
BUNDLE = Path('/content/final.zip')
SCORES = {  # name -> (images, references)
    'oficial': ('imagens', 'test_kraken/ground_truth'),
    'gemini': ('imagens', 'test_kraken/ground_truth_gemini'),
    'teste_gemini': ('imagens_extra/teste/imagens', 'imagens_extra/teste/referencias'),
}
KRAKEN = 'kraken @ git+https://github.com/mittagessen/kraken.git@bb487219e7d8289274443d1ad0989efb4924dc2a'
PACKAGES = [KRAKEN, 'transformers==4.57.1', 'rapidfuzz==3.14.1', 'opencv-python-headless==4.12.0.88', 'nbclient',
            'nbformat', 'ipykernel']
REPO = Path('/content/repo')
OUT = Path('/content') / RUN
HOLD_HOURS = 4


def log(message: str) -> None:
    line = f'{time.strftime("%H:%M:%S")} {message}'
    print(line, flush=True)
    with open(OUT / 'job.log', 'a') as stream:
        stream.write(line + '\n')


def run(cmd: list[str], name: str) -> int:
    log(f'{name}: {" ".join(cmd)[:300]}')
    with open(OUT / f'{name}.log', 'w') as stream:
        code = subprocess.run(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT).returncode
    log(f'{name}: exit {code}')
    return code


def score() -> int:
    codes = []
    for name, (images, references) in SCORES.items():
        codes.append(run([sys.executable, 'tools/score_final.py', FINAL, '--labels', 'final_v6', '--images', images,
                          '--references', references, '--out', str(OUT / f'final_scores_{name}.json'),
                          '--cache', '/content/cache', '--device', 'cuda:0', '--workers', '8'], f'score_{name}'))
    return max(codes)


def entrega() -> int:
    code = run([sys.executable, 'tools/build_entrega.py', '--out', 'entrega_ocr', '--execute'], 'build_entrega')
    archive = shutil.make_archive('/content/entrega_ocr', 'zip', REPO, 'entrega_ocr')
    subprocess.run(['zip', '-qd', archive, 'entrega_ocr/cache/*'], check=False)  # cache stays out of the delivery
    shutil.move(archive, OUT / 'entrega_ocr.zip')
    return code


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in ('READY', 'FETCHED', 'result.zip'):
        (OUT / stale).unlink(missing_ok=True)
    code = 1
    try:
        if shutil.which('ketos') is None:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PACKAGES], check=True)
        subprocess.run(['unzip', '-oq', str(BUNDLE), '-d', str(REPO)], check=True)
        code = {'score': score, 'entrega': entrega}[MODE]()
    finally:
        archive = shutil.make_archive(f'/content/{RUN}_result', 'zip', OUT)
        shutil.move(archive, OUT / 'result.zip')
        (OUT / 'READY').write_text(str(code))
        log(f'READY {code}')
    deadline = time.time() + HOLD_HOURS * 3600
    while time.time() < deadline and not (OUT / 'FETCHED').exists():
        time.sleep(30)


main()
