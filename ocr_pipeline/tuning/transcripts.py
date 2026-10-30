"""Attach page transcriptions (one text line per manuscript line, e.g. an LLM's diplomatic transcription of pages
outside the evaluation set) to the segmented lines of a pseudo-label run.

The transcription's lines are aligned to the recognizer's greedy readings of the segmented lines by dynamic
programming (match cost: normalized edit distance; a line on either side may stay unmatched), so lines the
segmenter missed (running headers, signatures) or split (table columns) do not shift the rest of the page. Each row of
a transcribed page gets `transcript` (the aligned text, or null) and `transcript_cer` (its distance to the reading);
`pseudo_labels select --max-transcript-cer` then trains on the transcript instead of the pipeline's label.

  python -m ocr_pipeline.tuning.transcripts outputs/pseudo/ldl1 --transcripts imagens_extra/transcriptions_txt
"""
import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)
SKIP = 0.7  # cost of leaving a line unmatched; a pair costs at most 1


def clean(line: str) -> str | None:
    """Collapse column spacing and drop decorative tildes; editorial notes ([...]) make the line unusable."""
    if '[' in line or ']' in line:
        return None
    text = ' '.join(t for t in line.split() if t.strip('~'))
    return text or None


def read_transcript(path: Path) -> list[str | None]:
    return [clean(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _norm(text: str) -> str:
    return ' '.join(re.sub(r'[^\w\s]', '', text.lower()).split())


def distance(reading: str, text: str) -> float:
    a, b = _norm(reading), _norm(text)
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)


def align(readings: list[str], texts: list[str | None]) -> list[tuple[int | None, int | None]]:
    """Monotone alignment of segmented lines to transcription lines (None = unmatched)."""
    n, m = len(readings), len(texts)
    cost = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[''] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            steps = []
            if i and j:
                pair = 1.0 if texts[j - 1] is None else min(distance(readings[i - 1], texts[j - 1]), 1.0)
                steps.append((cost[i - 1][j - 1] + pair, 'M'))
            if i:
                steps.append((cost[i - 1][j] + SKIP, 'R'))
            if j:
                steps.append((cost[i][j - 1] + SKIP, 'T'))
            if steps:
                cost[i][j], back[i][j] = min(steps)
    pairs, i, j = [], n, m
    while i or j:
        step = back[i][j]
        pairs.append((i - 1 if step in 'MR' else None, j - 1 if step in 'MT' else None))
        i, j = i - (step in 'MR'), j - (step in 'MT')
    return pairs[::-1]


def attach_page(rows: list[dict], texts: list[str | None]) -> None:
    rows.sort(key=lambda r: r['line'])
    for row in rows:
        row['transcript'], row['transcript_cer'] = None, 1.0
    for i, j in align([r['greedy'] for r in rows], texts):
        if i is not None and j is not None and texts[j] is not None:
            rows[i]['transcript'] = texts[j]
            rows[i]['transcript_cer'] = distance(rows[i]['greedy'], texts[j])


def attach(run: Path, transcripts: Path) -> dict:
    lines = run / 'lines.jsonl'
    pages: dict[str, list[dict]] = defaultdict(list)
    for line in lines.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        pages[row['page']].append(row)
    found = [name for name in pages if (transcripts / f'{name}.txt').is_file()]
    for name in found:
        attach_page(pages[name], read_transcript(transcripts / f'{name}.txt'))
    lines.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for rows in pages.values() for r in rows),
                     encoding='utf-8')
    cers = [r['transcript_cer'] for name in found for r in pages[name] if r['transcript'] is not None]
    summary = {'transcripts': str(transcripts), 'pages': len(found), 'lines': sum(len(pages[n]) for n in found),
               'matched': len(cers), 'median_cer': round(median(cers), 4) if cers else None}
    (run / 'transcripts.json').write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path, help='pseudo-label run (lines.jsonl from `pseudo_labels label`)')
    parser.add_argument('--transcripts', type=Path, required=True, help='<page>.txt per page image')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger.info('%s', attach(args.run, args.transcripts))


if __name__ == '__main__':
    main()
