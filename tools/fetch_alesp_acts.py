"""Download São Paulo state laws and decrees of a period from the ALESP legislation API (original spelling:
municipaes, fiscalisação, intendencia) as plain text, for the period lexicon and n-gram LM.

Acts are selected by a fixed rule (type and year), never by looking at this project's pages or GT. Acts whose
text mentions São Carlos are dropped, so nothing from or about the transcribed book can leak in.

  python tools/fetch_alesp_acts.py --out models/lexicon/sources/alesp --year-min 1889 --year-max 1910
"""
import argparse
import hashlib
import html
import json
import logging
import re
import time
from pathlib import Path

import requests

API = 'https://baleg-api-prd.al.sp.gov.br/normas'
TYPES = {'lei': 9, 'decreto': 3}
HEADERS = {'User-Agent': 'Mozilla/5.0 (research corpus; polite crawler)'}
_EXCLUDE = re.compile(r's(ão|\.)\s*carlos|carlos\s+do\s+pinhal', re.IGNORECASE)
_TAG = re.compile(r'<[^>]+>')
_BLOCK = re.compile(r'<(script|style)\b.*?</\1>', re.IGNORECASE | re.DOTALL)

logger = logging.getLogger(__name__)


def get(url: str, params: dict | None = None, tries: int = 4) -> requests.Response:
    """GET with a browser user agent; dropped connections are retried with a growing pause."""
    for attempt in range(tries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=60)
            response.raise_for_status()
            return response
        except requests.ConnectionError:
            if attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError('unreachable')


def list_acts(year: int, type_id: int) -> list[dict]:
    return get(API, {'ano': year, 'idsTipo': type_id, 'page': 0, 'size': 1000}).json()['content']


def html_text(raw: bytes) -> str:
    return html.unescape(_TAG.sub(' ', _BLOCK.sub(' ', _decode(raw))))


def _decode(raw: bytes) -> str:
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return raw.decode('latin-1')


def fetch(url: str, path: Path, delay: float) -> bytes:
    if path.is_file():
        return path.read_bytes()
    time.sleep(delay)
    response = get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('models/lexicon/sources/alesp'))
    parser.add_argument('--year-min', type=int, default=1889)
    parser.add_argument('--year-max', type=int, default=1910)
    parser.add_argument('--delay', type=float, default=1.0, help='seconds between downloads')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    manifest = {'rule': {k: str(v) for k, v in vars(args).items()}, 'api': API, 'acts': []}
    for year in range(args.year_min, args.year_max + 1):
        for kind, type_id in TYPES.items():
            for act in list_acts(year, type_id):
                url = act.get('urlIntegraListaPesquisa')
                if not url:
                    continue
                raw_path = args.out / 'html' / str(year) / url.rsplit('/', 1)[-1]
                try:
                    raw = fetch(url, raw_path, args.delay)
                except (requests.HTTPError, requests.ConnectionError) as error:
                    logger.warning('skip %s: %s', url, error)
                    continue
                text = html_text(raw)
                excluded = bool(_EXCLUDE.search(text))
                if not excluded:
                    (args.out / 'text').mkdir(parents=True, exist_ok=True)
                    (args.out / 'text' / f'{raw_path.stem}.txt').write_text(text, encoding='utf-8')
                manifest['acts'].append({'url': url, 'year': year, 'type': kind, 'excluded_sao_carlos': excluded,
                                         'sha256': hashlib.sha256(raw).hexdigest()})
            logger.info('%d %s: %d acts so far', year, kind, len(manifest['acts']))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    kept = sum(not a['excluded_sao_carlos'] for a in manifest['acts'])
    logger.info('%d acts, %d kept (%d mention São Carlos)', len(manifest['acts']), kept, len(manifest['acts']) - kept)


if __name__ == '__main__':
    main()
