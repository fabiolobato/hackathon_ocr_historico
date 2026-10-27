"""Build a 19th-century Portuguese wordlist from Project Gutenberg (pre-1911-reform spelling: pharmacia, effectivo).

Books are selected by a fixed rule on the Gutenberg catalog, never by looking at this project's pages or GT:
language pt, at least one author born in [--born-min, --born-max] and alive after --alive-after.
Output: one "word count" per line, most frequent first (the FrequencyWords format read by postprocess.load_lexicon),
a manifest with the selected book ids and the SHA-256 of every downloaded text, and the bigram counts over the kept
words (<out>.bigrams.npz: words, unigram counts, sorted pair keys id1 * V + id2, pair counts) for context_lm.

  python tools/build_period_lexicon.py --out models/lexicon/pt_hist_gutenberg.txt
"""
import argparse
import csv
import hashlib
import json
import logging
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import requests

CATALOG_URL = 'https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv'
TEXT_URL = 'https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt'
_YEARS = re.compile(r'(\d{4})\??-(\d{4})')
_WORD = re.compile(r'[^\W\d_]+')
_START = re.compile(r'\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n')
_END = re.compile(r'\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG')

logger = logging.getLogger(__name__)


def fetch(url: str, path: Path, delay: float) -> bytes:
    if path.is_file():
        return path.read_bytes()
    time.sleep(delay)  # be polite to gutenberg.org
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def author_matches(authors: str, born: tuple[int, int], alive_after: int) -> bool:
    for author in authors.split(';'):
        match = _YEARS.search(author)
        if match and born[0] <= int(match[1]) <= born[1] and int(match[2]) >= alive_after:
            return True
    return False


def select_books(catalog: Path, language: str, born: tuple[int, int], alive_after: int) -> list[dict[str, str]]:
    with catalog.open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    return [{'id': r['Text#'], 'title': r['Title'], 'authors': r['Authors']} for r in rows
            if r['Type'] == 'Text' and r['Language'].strip() == language
            and author_matches(r['Authors'], born, alive_after)]


def body(text: str) -> str:
    """Drop the Gutenberg license header and footer (English)."""
    start = _START.search(text)
    end = _END.search(text)
    return text[start.end() if start else 0:end.start() if end else len(text)]


def count_words(text: str) -> Counter[str]:
    return Counter(_WORD.findall(unicodedata.normalize('NFC', text).lower()))


def bigram_counts(texts: list[Path], words: list[str], min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Counts of adjacent kept-word pairs; a dropped word breaks the chain. Keys are id1 * len(words) + id2."""
    index = {w: i for i, w in enumerate(words)}
    keys = []
    for path in texts:
        ids = np.array([index.get(w, -1) for w in _WORD.findall(unicodedata.normalize(
            'NFC', body(path.read_text(encoding='utf-8', errors='replace'))).lower())], dtype=np.int64)
        valid = (ids[:-1] >= 0) & (ids[1:] >= 0)
        keys.append(ids[:-1][valid] * len(words) + ids[1:][valid])
    unique, counts = np.unique(np.concatenate(keys), return_counts=True)
    kept = counts >= min_count
    return unique[kept], counts[kept].astype(np.int32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('models/lexicon/pt_hist_gutenberg.txt'))
    parser.add_argument('--sources', type=Path, default=Path('models/lexicon/sources/gutenberg'),
                        help='download cache for the catalog and the book texts')
    parser.add_argument('--language', default='pt')
    parser.add_argument('--born-min', type=int, default=1790)
    parser.add_argument('--born-max', type=int, default=1875)
    parser.add_argument('--alive-after', type=int, default=1850)
    parser.add_argument('--min-count', type=int, default=2, help='drop words seen fewer times (typos, OCR noise)')
    parser.add_argument('--delay', type=float, default=1.0, help='seconds between downloads')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    catalog = args.sources / 'pg_catalog.csv'
    fetch(CATALOG_URL, catalog, 0)
    books = select_books(catalog, args.language, (args.born_min, args.born_max), args.alive_after)
    logger.info('%d books match the catalog rule', len(books))
    counts: Counter[str] = Counter()
    for i, book in enumerate(books, 1):
        try:
            raw = fetch(TEXT_URL.format(id=book['id']), args.sources / f'pg{book["id"]}.txt', args.delay)
        except requests.HTTPError as error:  # a few catalog entries have no plain-text file
            logger.warning('skip %s: %s', book['id'], error)
            book['sha256'] = None
            continue
        book['sha256'] = hashlib.sha256(raw).hexdigest()
        counts.update(count_words(body(raw.decode('utf-8', errors='replace'))))
        if i % 50 == 0:
            logger.info('%d/%d books, %d distinct words', i, len(books), len(counts))
    words = [(w, n) for w, n in counts.most_common() if n >= args.min_count]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(f'{w} {n}\n' for w, n in words), encoding='utf-8')
    manifest = {'rule': {k: v for k, v in vars(args).items() if k not in ('out', 'sources')},
                'catalog_url': CATALOG_URL, 'tokens': sum(counts.values()), 'words_kept': len(words),
                'output_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(), 'books': books}
    texts = [args.sources / f'pg{b["id"]}.txt' for b in books if b['sha256']]
    keys, pair_counts = bigram_counts(texts, [w for w, _ in words], args.min_count)
    np.savez_compressed(args.out.with_suffix('.bigrams.npz'), words=np.array([w for w, _ in words]),
                        unigrams=np.array([n for _, n in words], dtype=np.int64), keys=keys, counts=pair_counts)
    logger.info('%d bigrams kept', len(keys))
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                                                      encoding='utf-8')
    logger.info('%d tokens, %d words kept -> %s', manifest['tokens'], len(words), args.out)


if __name__ == '__main__':
    main()
