"""Page discovery and the fixed tuning/held-out split inherited from test_kraken (seed 42 -> held out 7, 10, 12)."""
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Page:
    number: int
    image: Path
    reference: Path

    def reference_text(self) -> str:
        return self.reference.read_text(encoding='utf-8')


def _nfc(name: str) -> str:
    return unicodedata.normalize('NFC', name)


def discover(image_dir: Path, reference_dir: Path) -> list[Page]:
    references = {_nfc(p.stem): p for p in reference_dir.glob('*.txt')}
    pages = []
    for image in image_dir.glob('*.jpg'):
        reference = references.get(_nfc(image.stem))
        if reference is None:
            raise FileNotFoundError(f'No reference for {image.name}')
        pages.append(Page(int(image.name.split('.')[0]), image, reference))
    if not pages:
        raise FileNotFoundError(f'No JPEGs in {image_dir}')
    return sorted(pages, key=lambda p: p.number)


def split(pages: list[Page], heldout_count: int = 3, seed: int = 42) -> tuple[list[Page], list[Page]]:
    shuffled = pages.copy()
    random.Random(seed).shuffle(shuffled)
    tuning, heldout = shuffled[heldout_count:], shuffled[:heldout_count]
    return sorted(tuning, key=lambda p: p.number), sorted(heldout, key=lambda p: p.number)


def select(pages: list[Page], subset: str) -> list[Page]:
    tuning, heldout = split(pages)
    if subset == 'tuning':
        return tuning
    if subset == 'heldout':
        return heldout
    if subset == 'all':
        return pages
    wanted = {int(n) for n in subset.split(',')}
    return [p for p in pages if p.number in wanted]
