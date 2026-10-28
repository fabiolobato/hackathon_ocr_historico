"""Drop probable typos from a period lexicon: rare words one edit away from a much more frequent word.

The period corpora are transcriptions with their own typos (tilulo 7x next to titulo, precidente, minicipal);
in the lexicon they count as known words, so a misread can be "corrected" into them or left as read. Rule
(in the spirit of Fink et al. 2017's typo-free modern lexicon, without needing one): drop a word if its count is
at most --max-count and some word within one edit (ignoring accents) is at least --ratio times more frequent.
Uses no GT. The n-gram LM is left as is; only the candidate list changes.

  python tools/clean_lexicon.py models/lexicon/pt_hist_legal_w4.txt --ratio 50 --max-count 40
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from ocr_pipeline.postprocess import fold  # noqa: E402

logger = logging.getLogger(__name__)


def read_counts(path: Path) -> list[tuple[str, int]]:
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        word, _, count = line.rpartition(' ')
        rows.append((word, int(count)))
    return rows


def _deletions(word: str) -> set[str]:
    return {word} | {word[:i] + word[i + 1:] for i in range(len(word))}


def typos(rows: list[tuple[str, int]], ratio: float, max_count: int) -> dict[str, str]:
    """word -> the frequent word it is taken to be a typo of."""
    best: dict[str, tuple[str, int]] = {}  # accent-free form -> (most frequent spelling, total count)
    for word, count in rows:
        key = fold(word)
        spelling, total = best.get(key, (word, 0))
        best[key] = (spelling, total + count)
    buckets: dict[str, list[str]] = defaultdict(list)  # single-deletion key -> accent-free forms
    for key in best:
        for d in _deletions(key):
            buckets[d].append(key)
    dropped = {}
    for key, (spelling, count) in best.items():
        if count > max_count or len(key) < 4:
            continue
        near = {k for d in _deletions(key) for k in buckets[d] if k != key}
        for other in sorted(near, key=lambda k: -best[k][1]):
            if best[other][1] < ratio * count:
                break
            if Levenshtein.distance(key, other) == 1:
                dropped[key] = best[other][0]
                break
    return dropped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('lexicon', type=Path)
    parser.add_argument('--ratio', type=float, default=50.0)
    parser.add_argument('--max-count', type=int, default=40)
    parser.add_argument('--out', type=Path, help='default: <lexicon stem>_clean_r<ratio>_m<max>.txt')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    rows = read_counts(args.lexicon)
    dropped = typos(rows, args.ratio, args.max_count)
    out = args.out or args.lexicon.with_name(f'{args.lexicon.stem}_clean_r{args.ratio:g}_m{args.max_count}.txt')
    kept = [(w, c) for w, c in rows if fold(w) not in dropped]
    out.write_text(''.join(f'{w} {c}\n' for w, c in kept), encoding='utf-8')
    manifest = {'source': str(args.lexicon), 'ratio': args.ratio, 'max_count': args.max_count,
                'words': len(rows), 'kept': len(kept), 'dropped_examples': dict(list(dropped.items())[:200])}
    out.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    logger.info('%d -> %d words (%d typo forms dropped) -> %s', len(rows), len(kept), len(dropped), out)


if __name__ == '__main__':
    main()
