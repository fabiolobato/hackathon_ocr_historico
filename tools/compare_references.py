"""Word-align the official GT with another transcription of the 12 pages (e.g. Gemini) and classify each disagreement:
accents, period spelling, word split/merge, abbreviation/number, missing words, or a different word.

  python tools/compare_references.py DIR_WITH_SAME_FILE_NAMES OUT.json
"""
import json, re, sys, unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from ocr_pipeline.metrics import normalize

GT, GEM, OUT = Path('test_kraken/ground_truth'), Path(sys.argv[1]), Path(sys.argv[2])


def fold(w: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn')


def period(w: str) -> str:
    w = fold(w)
    for a, b in (('ph', 'f'), ('th', 't'), ('y', 'i'), ('ch', 'c'), ('mm', 'm'), ('nn', 'n'), ('ll', 'l'), ('tt', 't'),
                 ('cc', 'c'), ('pp', 'p'), ('ff', 'f'), ('ss', 's'), ('z', 's'), ('sc', 'c'), ('ct', 't'), ('pt', 't'),
                 ('gu', 'g'), ('qu', 'c'), ('k', 'c'), ('ao', 'am'), ('h', '')):
        w = w.replace(a, b)
    return w


def kind(r: list[str], h: list[str]) -> str:
    a, b = ' '.join(r), ' '.join(h)
    if not r or not h:
        return 'só num dos dois (palavra ou linha a mais)'
    if ''.join(r) == ''.join(h):
        return 'junção/separação'
    if fold(a) == fold(b):
        return 'acento'
    if period(''.join(r)) == period(''.join(h)):
        return 'grafia (modernização)'
    if len(r) == len(h) == 1 and (r[0].startswith(h[0]) or h[0].startswith(r[0]) or re.search(r'\d', a + b)):
        return 'abreviatura/número'
    return 'palavra diferente'


rows, counts, words = [], Counter(), 0
for ref in sorted(GT.glob('*.txt'), key=lambda p: int(p.name.split('.')[0])):
    gem = GEM / ref.name
    r = normalize(ref.read_text(encoding='utf-8'), False).split()
    h = normalize(gem.read_text(encoding='utf-8'), False).split()
    words += len(r)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, r, h, autojunk=False).get_opcodes():
        if op == 'equal':
            continue
        k = kind(r[i1:i2], h[j1:j2])
        counts[k] += max(i2 - i1, j2 - j1)
        rows.append({'page': int(ref.name.split('.')[0]), 'kind': k, 'oficial': ' '.join(r[i1:i2]),
                     'gemini': ' '.join(h[j1:j2]), 'ctx': ' '.join(r[max(0, i1 - 4):i2 + 4])})
OUT.write_text(json.dumps({'words': words, 'counts': counts, 'rows': rows}, ensure_ascii=False, indent=1))
print('palavras no GT oficial:', words)
for k, v in counts.most_common():
    print(f'{k:45s} {v:5d}  ({100 * v / words:.1f}% das palavras)')
