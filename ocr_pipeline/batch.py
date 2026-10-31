"""Run a chosen configuration on new page images: one .txt per image and, when references are given, WER/CER.

Nothing is tuned here. The cache lives inside --out by default: cache keys use the image file name, so a new
image named like one of the 12 project pages must not reuse that page's cached segmentation.

  python -m ocr_pipeline.batch --config configs/final_v5.json --images novas_imagens --out outputs/batch
  python -m ocr_pipeline.batch --config configs/final_v5.json --images novas_imagens --references novas_gt \\
      --out outputs/batch
"""
import argparse
import html
import json
import logging
import unicodedata
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig, load_config
from .engine import Engine
from .metrics import corpus, score
from .pipeline import PageTrace, page_executor, trace_page, worker_engine
from .postprocess import correct
from .presentation import images_html, predicted_html, steps_table, text_steps, text_versions
from .visualize import line_table, word_diff

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchPage:
    number: int  # position in the batch, from 1
    image: Path
    reference: Path | None  # <image stem>.txt in the references folder, if there is one

    def reference_text(self) -> str:
        return self.reference.read_text(encoding='utf-8') if self.reference else ''


def discover_batch(images: Path, references: Path | None = None) -> list[BatchPage]:
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f'no images ({", ".join(IMAGE_SUFFIXES)}) in {images}')
    known = {unicodedata.normalize('NFC', p.stem): p for p in references.glob('*.txt')} if references else {}
    return [BatchPage(i, path, known.get(unicodedata.normalize('NFC', path.stem)))
            for i, path in enumerate(files, 1)]


def check_config(config: PipelineConfig) -> None:
    if config.post.domain_lexicon:
        raise ValueError('domain_lexicon builds a lexicon from project GT; not available for new images')
    for path in (config.post.lexicon, config.post.period_lexicon, config.post.context_lm):
        if path and not Path(path).is_file():
            raise FileNotFoundError(f'{path} (required by the config) is missing')


def transcribe(engine: Engine, page: BatchPage, config: PipelineConfig) -> tuple[PageTrace, str]:
    trace = trace_page(engine, page, config)  # type: ignore[arg-type]  # duck-typed page: image + number
    return trace, '\n'.join(correct(trace.lines, config.post))


def _transcribe_worker(args: tuple[BatchPage, PipelineConfig]) -> tuple[PageTrace, str]:
    return transcribe(worker_engine(), *args)


def run_batch(engine: Engine, pages: list[BatchPage], config: PipelineConfig, out: Path,
              executor: Executor | None = None) -> tuple[list[dict[str, Any]], dict[int, PageTrace]]:
    """Writes out/texts/<image stem>.txt per page and out/summary.json; returns the rows and the traces.
    executor: pipeline.page_executor, to transcribe pages in parallel."""
    (out / 'texts').mkdir(parents=True, exist_ok=True)
    rows, traces = [], {}
    results = (executor.map(_transcribe_worker, [(page, config) for page in pages]) if executor
               else (transcribe(engine, page, config) for page in pages))
    for page, (trace, text) in zip(pages, results):
        target = out / 'texts' / f'{page.image.stem}.txt'
        target.write_text(text + '\n', encoding='utf-8')
        row = {'image': page.image.name, 'text_file': str(target), 'lines': len(trace.segmentation.lines),
               'words': len(text.split())}
        if page.reference:
            row.update(score(page.reference_text(), text))
        rows.append(row)
        traces[page.number] = trace
        logger.info('%s: %d lines, %d words%s', page.image.name, row['lines'], row['words'],
                    f", WER {100 * row['wer_norm']:.2f}" if page.reference else '')
    scored = [r for r in rows if 'wer_norm' in r]
    summary = {'pages': rows, 'scored_pages': len(scored), 'corpus': corpus(scored) if scored else None}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return rows, traces


def card_html(page: BatchPage, trace: PageTrace, config: PipelineConfig) -> str:
    """Every image step on both paths, the segmentation, the text after each step and, with a reference, the
    WER/CER of each step and the final text diffed against it."""
    parts = [f'<div class="card"><h3>{html.escape(page.image.name)}</h3>', images_html(page, trace, config)]
    if page.reference:
        steps = text_steps(page, trace, config, ())  # type: ignore[arg-type]  # duck-typed page
        reference = page.reference_text()
        parts += [f'<h4>texto: cada etapa e seu WER contra a referência</h4>{steps_table(steps)}',
                  '<h4>texto final contra a referência</h4>',
                  f'<div class="pair"><div>{word_diff(reference, steps[-1].text)}</div><div><div class="cap">'
                  f'referência</div><div class="text">{html.escape(reference).replace(chr(10), "<br>")}</div></div>'
                  '</div>', predicted_html(steps[-1].text, reference, 'referência')]
    else:
        for label, text in text_versions(trace, config, ()):
            parts.append(f'<h4>{html.escape(label)}</h4>'
                         f'<div class="text">{html.escape(text).replace(chr(10), "<br>")}</div>')
    parts.append(f'<details><summary>linha a linha: recorte da imagem ↔ texto lido</summary>'
                 f'{line_table(trace, config.post.decoder != "edit")}</details></div>')
    return ''.join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--references', type=Path, help='optional: <image stem>.txt transcriptions for WER/CER')
    parser.add_argument('--out', type=Path, default=Path('outputs/batch'))
    parser.add_argument('--cache', type=Path, help='default: <out>/cache')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=1, help='processes transcribing pages in parallel')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = load_config(args.config)
    check_config(config)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'run_config.json').write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()},
                                                         'config': config.to_dict()}, indent=2), encoding='utf-8')
    engine = Engine(args.cache or args.out / 'cache', args.device, args.seed)
    pages = discover_batch(args.images, args.references)
    if args.workers > 1:
        with page_executor(engine, args.workers) as executor:
            rows, _ = run_batch(engine, pages, config, args.out, executor)
    else:
        rows, _ = run_batch(engine, pages, config, args.out)
    scored = [r for r in rows if 'wer_norm' in r]
    if scored:
        total = corpus(scored)
        logger.info('%d/%d pages scored: WER %.2f  CER %.2f', len(scored), len(rows), 100 * total['wer_norm'],
                    100 * total['cer_norm'])


if __name__ == '__main__':
    main()
