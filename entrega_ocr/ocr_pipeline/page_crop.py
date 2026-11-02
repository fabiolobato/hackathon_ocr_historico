"""Crop uncropped FPMSC scans the way the 12 project images were cropped (drops the watermark band and borders).

The project images are hand crops of the FPMSC Flickr 'large' scans (~694x1024). Matching them back (template
matching, scale 1.0) puts the page at left 0.02-0.05, top 0.09-0.10, right 0.96-0.98, bottom 0.95-0.96 of the
scan; the top band holds the Pró-Memória watermark. One fixed relative box (the median of the 11 full-page
matches) overlaps the hand crops with IoU 0.96-0.99. Automatic page detection (brightness or edge profiles) only
reached IoU ~0.86 and sometimes kept the watermark band, so scans from another source should pass their own
--box (fractions of width/height) instead.

  python -m ocr_pipeline.page_crop novas_imagens_brutas novas_imagens
  python -m ocr_pipeline.page_crop imagens_extra/ldl1 imagens_extra/ldl1_crop --skip 1,2,5-16,106
"""
import argparse
import re
from pathlib import Path

from PIL import Image

FPMSC_BOX = (0.035, 0.100, 0.970, 0.953)  # left, top, right, bottom as fractions of the scan
SUFFIXES = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


def crop_page(image: Image.Image, box: tuple[float, float, float, float] = FPMSC_BOX) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop((round(left * width), round(top * height), round(right * width), round(bottom * height)))


def parse_numbers(text: str) -> set[int]:
    """'1,2,5-16' -> {1, 2, 5, ..., 16}"""
    numbers: set[int] = set()
    for part in filter(None, text.split(',')):
        low, _, high = part.partition('-')
        numbers.update(range(int(low), int(high or low) + 1))
    return numbers


def crop_folder(source: Path, target: Path, box: tuple[float, float, float, float] = FPMSC_BOX,
                skip: set[int] = frozenset()) -> list[Path]:
    """Crops every image of source into target under the same name; skip: leading photo numbers to leave out."""
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(p for p in source.iterdir() if p.suffix.lower() in SUFFIXES):
        number = re.match(r'(\d+)', path.name)
        if number and int(number.group(1)) in skip:
            continue
        with Image.open(path) as image:
            crop_page(image.convert('RGB'), box).save(target / path.name, quality=95)
        written.append(target / path.name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--box', default=','.join(map(str, FPMSC_BOX)), help='left,top,right,bottom fractions')
    parser.add_argument('--skip', default='', help='leading photo numbers to leave out, e.g. covers or eval pages')
    args = parser.parse_args()
    box = tuple(float(v) for v in args.box.split(','))
    written = crop_folder(args.source, args.target, box, parse_numbers(args.skip))
    print(f'{len(written)} images cropped into {args.target}')


if __name__ == '__main__':
    main()
