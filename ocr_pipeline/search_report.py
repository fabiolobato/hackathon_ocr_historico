"""HTML report of a search run. Every evaluated variant is shown on one probe page as image ↔ text pairs:
the segmentation overlay it produced (image stages), each line crop beside the text read from it, and the
words it changed relative to the stage's incumbent; corpus Δ and decision on top."""
import html
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from .config import from_dict
from .data import Page
from .engine import Engine
from .pipeline import PageTrace, domain_for, score_trace, trace_page
from .visualize import bar_svg, describe, img, line_box, line_table, overlay, pct, render, word_diff

if TYPE_CHECKING:
    from .search import Candidate, Search

logger = logging.getLogger(__name__)
IMAGE_KEYS = ('geometric_ops', 'seg_ops', 'seg.', 'params.')
STATUS = {'accepted': ('var(--miss-fg)', 'accepted'), 'improver': ('var(--acc)', 'improved alone'),
          'rejected': ('var(--sub-fg)', 'no gain: rejected'), 'incumbent': ('var(--muted)', 'reference: current choices minus this stage')}


def status(record: dict[str, Any], min_gain: float) -> str:
    if record['kind'] == 'incumbent':
        return 'incumbent'
    if record['accepted']:
        return 'accepted'
    return 'improver' if record['delta'] < -min_gain and record['kind'] == 'single' else 'rejected'


def delta_svg(items: list[tuple[str, float, str]]) -> str:
    """Diverging bars of Δ metric (points) around zero; left = better."""
    width, row_h, label_w = 680, 22, 250
    span = max([abs(d) for _, d, _ in items] + [0.5])
    zero, half = label_w + (width - label_w) / 2, (width - label_w) / 2 - 50
    bars = [f'<line x1="{zero}" x2="{zero}" y1="0" y2="{row_h * len(items)}" stroke="currentColor" opacity=".4"/>']
    for i, (label, delta, state) in enumerate(items):
        y, w = i * row_h, half * abs(delta) / span
        x = zero - w if delta < 0 else zero
        color = STATUS[state][0]
        tx = x - 4 if delta < 0 else x + w + 4
        bars.append(f'<text x="{label_w - 6}" y="{y + 15}" text-anchor="end" fill="currentColor">{html.escape(label)}</text>'
                    f'<rect x="{x:.1f}" y="{y + 4}" width="{max(w, 1):.1f}" height="{row_h - 8}" fill="{color}"/>'
                    f'<text x="{tx:.1f}" y="{y + 15}" text-anchor="{"end" if delta < 0 else "start"}" '
                    f'fill="currentColor">{delta:+.2f}</text>')
    return (f'<svg viewBox="0 0 {width} {row_h * len(items)}" style="max-width:{width}px;width:100%;font-size:12px">'
            f'{"".join(bars)}</svg>')


def overlay_excerpt(trace: PageTrace, n_lines: int) -> Image.Image:
    """Segmentation overlay cropped to the first n lines, so it sits beside their text."""
    lines = trace.segmentation.lines[:n_lines]
    canvas = overlay(trace.seg_image, trace.segmentation)
    if not lines:
        return canvas
    boxes = [line_box(line, canvas.size, pad=10) for line in lines]
    return canvas.crop((0, min(b[1] for b in boxes), canvas.width, max(b[3] for b in boxes)))


def variant_card(record: dict[str, Any], state: str, trace: PageTrace, row: dict[str, Any], incumbent_text: str,
                 page: Page, show_overlay: bool, n_lines: int) -> str:
    config = from_dict(record['config'])
    label = '+'.join(record['choices']) or 'incumbent'
    color, verdict = STATUS[state]
    head = (f'<h3>{html.escape(label)} <span style="color:{color}">· {verdict}</span></h3>'
            f'<div>corpus {record["metric"] * 100:.2f} (Δ {record["delta"] * 100:+.2f}) · page {page.number}: '
            f'WER norm {pct(row["wer_norm"])}, CER norm {pct(row["cer_norm"])}</div><div class="cap">{describe(config)}</div>')
    left = img(overlay_excerpt(trace, n_lines), f'segmentation of the first {n_lines} lines', 600) if show_overlay else ''
    left += line_table(trace, config.post.decoder != 'edit', n_lines)
    changes = (word_diff(row['text'], incumbent_text) if row['text'] != incumbent_text
               else '<div class="cap">same text as the incumbent</div>')
    right = (f'<div class="cap">page {page.number}: <span class="sub">incumbent</span> → '
             f'<span class="miss">this variant</span></div>{changes}<details><summary>full text vs GT</summary>'
             f'{word_diff(page.reference_text(), row["text"])}</details>')
    return f'<div class="card">{head}<div class="pair"><div>{left}</div><div>{right}</div></div></div>'


def path_section(search: 'Search', final: list['Candidate']) -> str:
    metric = search.spec.metric
    rows = [f'<tr><td>{t["pass"]}</td><td>{t["stage"]}</td><td>{"+".join(t["picks"]) or "<span class=rej>none</span>"}'
            f'</td><td class="num">{pct(t["before"])}</td><td class="num">{pct(t["after"])}</td><td class="num">'
            f'{100 * (t["after"] - t["before"]):+.2f}</td></tr>' for t in search.trail]
    values = {f'p{t["pass"]} after {t["stage"]}': 100 * t['after'] for t in search.trail if t['pass'] == 1}
    values.update({f'end of pass {t["pass"]}': 100 * t['after'] for t in search.trail
                   if t['stage'] == search.spec.stages[-1][0] and t['pass'] > 1})
    dropped = [f'pass {r["pass"]}: {r["choices"][0][1:]}' for r in search.records
               if r['kind'] == 'backward' and r['accepted']]
    final_value = search.score(search.build(final))
    return (f'<h2>Decision path ({metric}, lower is better)</h2>{bar_svg(values)}<table><tr><th>pass</th><th>stage</th>'
            f'<th>picks after stage</th><th class="num">before</th><th class="num">after</th><th class="num">Δ</th></tr>'
            f'{"".join(rows)}</table><p>Backward pass dropped: {"; ".join(dropped) or "nothing"}.</p>'
            f'<p>Final {metric}: <b>{pct(final_value)}</b> after {search.pass_no} pass(es).</p>'
            f'<div class="cap">{describe(search.build(final))}</div>')


def stage_records(search: 'Search', pass_no: int, stage: str) -> list[dict[str, Any]]:
    return [r for r in search.records if r['pass'] == pass_no and r['stage'] == stage]


def stage_section(search: 'Search', pass_no: int, stage: str, cards: list[str]) -> str:
    items = [('+'.join(r['choices']) or 'incumbent', 100 * r['delta'], status(r, search.spec.min_gain))
             for r in stage_records(search, pass_no, stage) if r['kind'] != 'incumbent']
    chart = delta_svg(items) if items else '<p class="cap">no candidate applicable</p>'
    note = '' if cards else '<p class="cap">picks unchanged from the previous pass; cards omitted</p>'
    return (f'<h2>Pass {pass_no} · {stage}</h2><div class="cap">Δ {search.spec.metric} vs this stage\'s incumbent '
            f'(points; incumbent = all current choices except this stage)</div>{chart}{note}{"".join(cards)}')


def stage_cards(search: 'Search', records: list[dict[str, Any]], show_overlay: bool, engine: Engine, probe: Page,
                pool: list[Page], n_lines: int) -> list[str]:
    cards, incumbent_text = [], ''
    for record in records:
        config = from_dict(record['config'])
        domain = domain_for(probe, pool) if config.post.domain_lexicon else ()
        trace = trace_page(engine, probe, config, domain)
        row = score_trace(probe, trace, config, domain)
        if record['kind'] == 'incumbent':
            incumbent_text = row['text']
        cards.append(variant_card(record, status(record, search.spec.min_gain), trace, row, incumbent_text, probe,
                                  show_overlay, n_lines))
    return cards


def write_report(out: Path, search: 'Search', final: list['Candidate'], engine: Engine, probe: Page,
                 pool: list[Page], n_lines: int = 6) -> None:
    body, last_picks = path_section(search, final), {}
    for entry in search.trail:
        stage, pass_no = entry['stage'], entry['pass']
        changed = last_picks.get(stage) != entry['picks'] or pass_no == 1
        last_picks[stage] = entry['picks']
        cands = next(c for name, c in search.spec.stages if name == stage)
        show_overlay = any(k.startswith(IMAGE_KEYS) for c in cands for k in c.overrides)
        cards = stage_cards(search, stage_records(search, pass_no, stage), show_overlay, engine, probe, pool,
                            n_lines) if changed else []
        body += stage_section(search, pass_no, stage, cards)
        logger.info('report: pass %d stage %s rendered', pass_no, stage)
    backward = [r for r in search.records if r['kind'] == 'backward']
    body += '<h2>Backward checks</h2>' + delta_svg([(f'p{r["pass"]} {r["choices"][0]}', 100 * r['delta'],
                                                     'accepted' if r['accepted'] else 'rejected') for r in backward])
    (out / 'report.html').write_text(render('Pipeline search', body), encoding='utf-8')
    logger.info('report -> %s', out / 'report.html')
