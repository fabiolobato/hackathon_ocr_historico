"""Colab setup: install the pinned Kraken commit and fetch PP-OCRv6 medium weights (MD5-verified)."""
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

KRAKEN = 'kraken @ git+https://github.com/mittagessen/kraken.git@bb487219e7d8289274443d1ad0989efb4924dc2a'
MEDIUM_URL = 'https://zenodo.org/records/21788410/files/medium.safetensors?download=1'
MEDIUM_MD5 = 'e3411a453ce3b9e9efae3f8631d85762'
WEIGHTS = Path('/content/models/medium.safetensors')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', KRAKEN, 'transformers==4.57.1'], check=True)
WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
if not WEIGHTS.exists():
    urllib.request.urlretrieve(MEDIUM_URL, WEIGHTS)
digest = hashlib.md5(WEIGHTS.read_bytes(), usedforsecurity=False).hexdigest()
assert digest == MEDIUM_MD5, digest
subprocess.run(['unzip', '-oq', '/content/v1.zip', '-d', '/content/data_v1'], check=True)
out = subprocess.run(['ketos', '--version'], capture_output=True, text=True)
print('ok', digest, out.stdout.strip() or out.stderr.strip()[-200:])
