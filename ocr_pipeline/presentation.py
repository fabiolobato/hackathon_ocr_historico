"""Step-by-step presentation of a configuration (used by the delivery notebook, labels in Portuguese).

Per page: every image operation applied cumulatively on top of the original, the segmentation drawn on its
input, then every text step (recognizer -> CTC rescoring -> conventions -> dehyphenation -> lexicon) with its
WER/CER against the GT, the final text diffed against the GT and, line by line, image crop <-> text.
Nothing here re-tunes anything: it only runs the given configuration.
"""
import html
from dataclasses import dataclass, replace
from typing import Any

from PIL import Image

from .config import PipelineConfig
from .data import Page
from .engine import Engine
from .metrics import corpus, score
from .pipeline import OP_PARAMS, PageTrace, domain_for, trace_page
from .postprocess import correct
from .preprocessing import apply_chain
from .visualize import CSS, img, line_table, overlay, pct, word_diff

OP_NAMES = {
    'upscale': 'ampliação', 'deskew': 'correção de inclinação', 'illumination': 'normalização de iluminação',
    'clahe': 'CLAHE (contraste local)', 'gamma': 'gama', 'kraken_nlbin': 'binarização nlbin',
    'contrast': 'contraste', 'grayscale': 'tons de cinza', 'sharpen': 'nitidez', 'otsu': 'binarização Otsu',
    'denoise': 'remoção de ruído', 'remove_rules': 'remoção de pauta',
}
EXTRA_CSS = '.steps { display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:8px; }' \
            '.flag { color:var(--sub-fg); font-weight:600; }'


@dataclass(frozen=True)
class TextStep:
    label: str
    text: str
    scores: dict[str, float | int]


def op_label(op: str, config: PipelineConfig) -> str:
    values = ', '.join(f'{name}={getattr(config.params, name)}' for name in OP_PARAMS.get(op, ()))
    return f'+ {OP_NAMES.get(op, op)}' + (f' ({values})' if values else '')


def cumulative(image: Image.Image, ops: tuple[str, ...], config: PipelineConfig,
               geometric: bool) -> list[tuple[str, Image.Image]]:
    """One (label, image) per op, each applied on top of the previous result."""
    steps = []
    for op in ops:
        image = apply_chain(image, (op,), config.params, geometric)
        steps.append((op_label(op, config), image))
    return steps


def image_steps(page: Page, config: PipelineConfig) -> dict[str, list[tuple[str, Image.Image]]]:
    with Image.open(page.image) as original:
        rgb = original.convert('RGB')
    geometry = [('original', rgb)] + cumulative(rgb, config.geometric_ops, config, geometric=True)
    base = geometry[-1][1]
    return {'geometria (vale para os dois fluxos)': geometry,
            'imagem de segmentação': cumulative(base, config.seg_ops, config, geometric=False),
            'imagem de reconhecimento': cumulative(base, config.rec_ops, config, geometric=False)}


def text_steps(page: Page, trace: PageTrace, config: PipelineConfig, domain: tuple[str, ...]) -> list[TextStep]:
    """Each step adds one text operation; the last one equals the pipeline's final output."""
    post = config.post
    texts = [('reconhecedor (guloso)', '\n'.join(trace.greedy))]
    if post.decoder in ('ctc', 'ctc+edit'):
        texts.append((f'+ reescore CTC (α={post.ctc_alpha})', '\n'.join(trace.lines)))
    no_edit = replace(post, decoder='ctc' if post.decoder != 'edit' else 'none')
    if post.conventions:
        texts.append(('+ convenções (nº, §)', '\n'.join(correct(trace.lines, replace(no_edit, dehyphenate=False)))))
    if post.dehyphenate:
        texts.append(('+ hifenização', '\n'.join(correct(trace.lines, no_edit))))
    if post.decoder in ('edit', 'ctc+edit'):
        texts.append((f'+ correção léxica ({post.lexicon_max_ratio})',
                      '\n'.join(correct(trace.lines, post, domain))))
    reference = page.reference_text()
    return [TextStep(label, text, score(reference, text)) for label, text in texts]


def images_html(page: Page, trace: PageTrace, config: PipelineConfig) -> str:
    blocks = []
    for title, steps in image_steps(page, config).items():
        figures = [img(image, label, 500) for label, image in steps] or \
                  ['<div class="cap">nenhuma operação: usa a imagem da geometria</div>']
        if title == 'imagem de segmentação':
            figures.append(img(overlay(trace.seg_image, trace.segmentation),
                               f'= segmentação: {len(trace.segmentation.lines)} linhas (polígono + linha de base)', 500))
        blocks.append(f'<h4>{title}</h4><div class="steps">{"".join(figures)}</div>')
    return ''.join(blocks)


def steps_table(steps: list[TextStep]) -> str:
    rows, previous = [], None
    for step in steps:
        wer = 100 * step.scores['wer_norm']
        delta = '' if previous is None else f'{wer - previous:+.2f}'
        rows.append(f'<tr><td>{html.escape(step.label)}</td><td class="num"><b>{wer:.2f}</b></td>'
                    f'<td class="num">{delta}</td><td class="num">{pct(step.scores["cer_norm"])}</td>'
                    f'<td class="num">{pct(step.scores["wer_strict"])}</td></tr>')
        previous = wer
    head = ('<tr><th>etapa de texto (cumulativa)</th><th class="num">WER norm. %</th><th class="num">Δ</th>'
            '<th class="num">CER norm. %</th><th class="num">WER estrito %</th></tr>')
    return f'<table>{head}{"".join(rows)}</table>'


def page_html(engine: Engine, page: Page, config: PipelineConfig, pool: list[Page],
              title: str, note: str = '') -> tuple[str, list[TextStep]]:
    domain = domain_for(page, pool) if config.post.domain_lexicon else ()
    trace = trace_page(engine, page, config, domain)
    steps = text_steps(page, trace, config, domain)
    final, reference = steps[-1], page.reference_text()
    body = (f'<div class="card"><h3>{html.escape(title)}</h3>'
            + (f'<div class="flag">{html.escape(note)}</div>' if note else '')
            + images_html(page, trace, config)
            + f'<h4>texto: cada etapa e seu WER contra o GT</h4>{steps_table(steps)}'
            + '<h4>texto final contra o GT</h4><div class="cap"><span class="sub">riscado</span> = palavra errada, '
              '<span class="miss">verde</span> = o que o GT diz</div>'
            + f'<div class="pair"><div>{word_diff(reference, final.text)}</div>'
              f'<div><div class="cap">GT</div><div class="text">{html.escape(reference).replace(chr(10), "<br>")}'
              '</div></div></div>'
            + f'<details><summary>linha a linha: recorte da imagem ↔ texto lido</summary>'
              f'{line_table(trace, config.post.decoder != "edit")}</details></div>')
    return body, steps


def style() -> str:
    return f'<style>{CSS.replace("body {", ".ocr-body {")}{EXTRA_CSS}</style>'


def stage_corpus(results: dict[int, list[TextStep]], pages: list[int]) -> list[tuple[str, dict[str, float]]]:
    """Corpus scores of each text step over the given pages."""
    labels = [step.label for step in next(iter(results.values()))]
    return [(label, corpus([results[p][i].scores for p in pages])) for i, label in enumerate(labels)]


def summary_html(results: dict[str, dict[int, list[TextStep]]], heldout: set[int], in_sample: set[str]) -> str:
    """Final WER per page and corpus totals (all / tuning / held-out) for each configuration."""
    pages = sorted(next(iter(results.values())))
    head = '<tr><th>página</th>' + ''.join(f'<th class="num">{html.escape(n)}</th>' for n in results) + '</tr>'
    rows = []
    for p in pages:
        kind = 'reservada' if p in heldout else 'ajuste'
        cells = ''.join(f'<td class="num">{pct(results[n][p][-1].scores["wer_norm"])}'
                        + ('*' if n in in_sample and p not in heldout else '') + '</td>' for n in results)
        rows.append(f'<tr><td>{p} <span class="cap">({kind})</span></td>{cells}</tr>')
    groups = (('todas', pages), ('ajuste', [p for p in pages if p not in heldout]),
              ('reservadas', [p for p in pages if p in heldout]))
    for name, group in groups:
        cells = ''.join(f'<td class="num"><b>{pct(corpus([results[n][p][-1].scores for p in group])["wer_norm"])}'
                        f'</b></td>' for n in results)
        rows.append(f'<tr><td><b>corpus: {name}</b></td>{cells}</tr>')
    return f'<table>{head}{"".join(rows)}</table>'
