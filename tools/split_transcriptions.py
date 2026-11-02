"""Sort page transcriptions of the book (Gemini, E15) into train / test / quarantine and lay out the test set.

  python tools/split_transcriptions.py imagens_extra/transcriptions_txt imagens_extra/transcriptions_allothers_txt \\
      --new imagens_extra/transcriptions_allothers_txt --test-count 10 --seed 42

Rules (by the leading sequence number of the file name, as in imagens_extra/ldl1):
- evaluated pages (the numbers of imagens/*.jpg, 5-16) -> quarantine: never train, test or lexicon text; kept only to
  measure the transcriptions against the human GT;
- covers and labels (no cropped page in --crops, or fewer than --min-lines lines) -> covers;
- test: --test-count pages drawn with --seed from the pages of --new (transcribed after the first training rounds,
  so no earlier model or LM has seen their text) -> never trained on, never in a lexicon/LM;
- the rest -> train.
Output under --out: transcricoes/{treino,teste,avaliadas,capas}/, treino_crop/ (cropped training images, the image
folder for self_train), teste/imagens/<n>.jpg + teste/referencias/<n>.txt (lines joined, line-end hyphens removed;
the layout ocr_pipeline.data.discover reads), and split.json. The source folders are moved to transcricoes/brutas/.
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path


def number(path: Path) -> int:
    return int(path.name.split('br-')[0])


def reference_text(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip() and '[' not in line]
    return re.sub(r'[-=¬]\s*\n\s*', '', '\n'.join(lines)) + '\n'


def classify(paths: list[Path], crops: Path, evaluated: set[int], min_lines: int) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {'avaliadas': [], 'capas': [], 'paginas': []}
    for path in sorted(paths, key=number):
        lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if number(path) in evaluated:
            groups['avaliadas'].append(path)
        elif not (crops / f'{path.stem}.jpg').is_file() or len(lines) < min_lines:
            groups['capas'].append(path)
        else:
            groups['paginas'].append(path)
    return groups


def lay_out(groups: dict[str, list[Path]], test: list[Path], crops: Path, out: Path) -> None:
    root = out / 'transcricoes'
    targets = {**{k: v for k, v in groups.items() if k != 'paginas'}, 'teste': test,
               'treino': [p for p in groups['paginas'] if p not in test]}
    for folder, paths in targets.items():
        (root / folder).mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, root / folder / path.name)
    (out / 'treino_crop').mkdir(exist_ok=True)
    for path in targets['treino']:
        shutil.copy2(crops / f'{path.stem}.jpg', out / 'treino_crop' / f'{path.stem}.jpg')
    for sub in ('imagens', 'referencias'):
        (out / 'teste' / sub).mkdir(parents=True, exist_ok=True)
    for path in test:
        shutil.copy2(crops / f'{path.stem}.jpg', out / 'teste' / 'imagens' / f'{number(path)}.jpg')
        (out / 'teste' / 'referencias' / f'{number(path)}.txt').write_text(reference_text(path), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sources', type=Path, nargs='+', help='folders of <page>.txt transcriptions')
    parser.add_argument('--new', type=Path, required=True, help='source folder the test pages are drawn from')
    parser.add_argument('--crops', type=Path, default=Path('imagens_extra/ldl1_crop'))
    parser.add_argument('--evaluated', type=Path, default=Path('imagens'), help='the evaluated page images')
    parser.add_argument('--out', type=Path, default=Path('imagens_extra'))
    parser.add_argument('--test-count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-lines', type=int, default=5)
    parser.add_argument('--split', type=Path, default=Path('configs/gemini_split.json'))
    args = parser.parse_args()
    evaluated = {int(p.name.split('.')[0]) for p in args.evaluated.glob('*.jpg')}
    groups = classify([p for s in args.sources for p in s.glob('*.txt')], args.crops, evaluated, args.min_lines)
    pool = [p for p in groups['paginas'] if p.parent == args.new]
    test = sorted(random.Random(args.seed).sample(pool, args.test_count), key=number)
    lay_out(groups, test, args.crops, args.out)
    split = {'rule': f'test: random.Random({args.seed}).sample of the {len(pool)} pages in {args.new.name}; '
                     'evaluated pages quarantined; covers excluded',
             **{k: [number(p) for p in v] for k, v in groups.items() if k != 'paginas'},
             'teste': [number(p) for p in test],
             'treino': [number(p) for p in groups['paginas'] if p not in test]}
    args.split.write_text(json.dumps(split, indent=1) + '\n')
    brutas = args.out / 'transcricoes' / 'brutas'
    brutas.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        shutil.move(str(source), str(brutas / source.name))
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in split.items()}))


if __name__ == '__main__':
    main()
