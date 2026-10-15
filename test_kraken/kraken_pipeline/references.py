import argparse
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from .artifacts import file_hash, write_json

logger = logging.getLogger(__name__)
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
HEADER = re.compile(r'^(\d+)\.BR FPMSC FUN CAM LDL.*Página')


def docx_paragraphs(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read('word/document.xml'))
    return [''.join(node.text or '' for node in paragraph.findall('.//w:t', NS)).strip()
            for paragraph in root.findall('.//w:p', NS)]


def split_sections(paragraphs: list[str]) -> dict[int, list[tuple[int, str]]]:
    sections: dict[int, list[tuple[int, str]]] = {}
    current = None
    for index, paragraph in enumerate(paragraphs):
        match = HEADER.match(paragraph)
        if match:
            current = int(match[1])
            if current in sections:
                raise ValueError(f'Duplicate page section: {current}')
            sections[current] = []
        elif current is not None and paragraph and not re.fullmatch(r'[_\s]+', paragraph):
            sections[current].append((index, paragraph))
    return sections


def organize_references(source: Path, image_dir: Path, output_dir: Path) -> list[Path]:
    sections = split_sections(docx_paragraphs(source))
    images = sorted(image_dir.glob('*.jpg'), key=lambda p: int(p.name.split('.')[0]))
    numbers = {int(image.name.split('.')[0]) for image in images}
    if not images or numbers != set(sections):
        raise ValueError(f'Image/GT mismatch: images={sorted(numbers)}, GT={sorted(sections)}')
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, pages = [], {}
    for image in images:
        section = sections[int(image.name.split('.')[0])]
        text = '\n'.join(re.sub(r'\s+', ' ', text) for _, text in section) + '\n'
        if not text.strip():
            raise ValueError(f'Empty reference: {image.name}')
        path = output_dir / f'{image.stem}.txt'
        if path.exists() and path.read_text(encoding='utf-8') != text:
            raise FileExistsError(f'Refusing to overwrite edited reference: {path}')
        path.write_text(text, encoding='utf-8')
        pages[image.stem] = {'status': 'supplied_unverified', 'source': source.name,
                             'source_paragraph_indices': [i for i, _ in section],
                             'image_sha256': file_hash(image), 'reference_sha256': file_hash(path)}
        paths.append(path)
    review = output_dir / 'review.json'
    if not review.exists():
        write_json(review, {'source_sha256': file_hash(source), 'pages': pages})
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Split supplied DOCX GT into one UTF-8 reference per image.')
    parser.add_argument('source', type=Path)
    parser.add_argument('--images', type=Path, default=Path('Imagens'))
    parser.add_argument('--output', type=Path, default=Path('ground_truth'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    paths = organize_references(args.source, args.images, args.output)
    logger.info('Organized %s page references in %s', len(paths), args.output)


if __name__ == '__main__':
    main()
