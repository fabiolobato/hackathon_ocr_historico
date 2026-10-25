"""Visual report: each method's images always sit next to the text that method produced.

Per (config, page) card:
  segmentation image + numbered line polygons | recognition image   <->  final text, word errors vs GT
  per line: crop of the recognition image  <->  greedy text | decoded text (CTC rescoring)
  post-processing: decoded page text -> final page text, changed words marked
Self-contained HTML (images embedded). Scores use the normalized policy of metrics.py.

  python -m ocr_pipeline.visualize --config configs/best_v2.json --config configs/best_v3_ft.json --pages heldout
"""
import argparse
import base64
import html
import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from rapidfuzz.distance import Levenshtein

from .config import PipelineConfig, load_config
from .data import Page, discover, select, split
from .engine import Engine
from .metrics import normalize
from .pipeline import PageTrace, domain_for, score_trace, trace_page

logger = logging.getLogger(__name__)
PALETTE = [(230, 25, 75), (60, 140, 75), (0, 110, 200), (245, 130, 48), (145, 30, 180), (0, 150, 150)]

CSS = '''
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#e3e1da; --ok:#1d1d1b;
        --sub-bg:#fde2e1; --sub-fg:#a4161a; --miss-bg:#dff3e3; --miss-fg:#1b6e33; --acc:#2b6cb0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#161614; --fg:#ecebe6; --muted:#9a9993; --card:#1f1f1c; --line:#34332f; --ok:#ecebe6;
  --sub-bg:#4a1d1d; --sub-fg:#ffb3ae; --miss-bg:#173a22; --miss-fg:#9be0ad; --acc:#7fb2f0; } }
body { background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, "Segoe UI", sans-serif; margin:0 auto;
       padding:16px; max-width:1500px; }
h1 { font-size:22px; } h2 { font-size:18px; margin-top:32px; border-bottom:1px solid var(--line); } h3 { font-size:15px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin:12px 0; }
.pair { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; align-items:start; }
.imgs { display:grid; grid-template-columns:1fr 1fr; gap:8px; } img { max-width:100%; height:auto; display:block; }
.cap { color:var(--muted); font-size:12px; } .text { font-family:ui-monospace, Menlo, monospace; font-size:13px; }
.sub { background:var(--sub-bg); color:var(--sub-fg); text-decoration:line-through; }
.miss { background:var(--miss-bg); color:var(--miss-fg); }
table { border-collapse:collapse; width:100%; } td, th { border-bottom:1px solid var(--line); padding:4px 6px;
        vertical-align:top; text-align:left; } th { color:var(--muted); font-weight:600; }
.num { text-align:right; font-variant-numeric:tabular-nums; } .lines td:first-child { width:45%; }
details { margin-top:8px; } summary { cursor:pointer; color:var(--acc); }
.acc { color:var(--miss-fg); font-weight:600; } .rej { color:var(--muted); }
@media (max-width:800px) { .pair, .imgs { grid-template-columns:1fr; } }
'''


def data_uri(image: Image.Image, max_width: int = 900) -> str:
    image = image.convert('RGB')
    if image.width > max_width:
        image = image.resize((max_width, round(image.height * max_width / image.width)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=80)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def img(image: Image.Image, caption: str, max_width: int = 900) -> str:
    return f'<figure style="margin:0"><img src="{data_uri(image, max_width)}"><figcaption class="cap">' \
           f'{html.escape(caption)}</figcaption></figure>'


def line_box(line: Any, size: tuple[int, int], pad: int = 4) -> tuple[int, int, int, int]:
    points = line.boundary or [(x, y + d) for x, y in line.baseline for d in (-25, 8)]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (max(0, min(xs) - pad), max(0, min(ys) - pad), min(size[0], max(xs) + pad), min(size[1], max(ys) + pad))


def overlay(image: Image.Image, segmentation: Any) -> Image.Image:
    """Line polygons and baselines, numbered from 1 in reading order."""
    canvas = image.convert('RGB')
    draw = ImageDraw.Draw(canvas, 'RGBA')
    for i, line in enumerate(segmentation.lines):
        color = PALETTE[i % len(PALETTE)]
        if line.boundary:
            draw.polygon([tuple(p) for p in line.boundary], fill=color + (45,), outline=color + (200,))
        draw.line([tuple(p) for p in line.baseline], fill=color + (255,), width=2)
        x, y = line.baseline[0]
        draw.text((max(0, x - 18), y - 12), str(i + 1), fill=color + (255,))
    return canvas


def _token_lines(text: str) -> list[list[str]]:
    return [normalize(line, strict=False).split() for line in text.splitlines()]


def word_diff(reference: str, hypothesis: str) -> str:
    """Hypothesis words in their own line layout; wrong words struck through, followed by the reference
    words in green. With reference=after, hypothesis=before it shows what a text step changed."""
    hyp_lines = _token_lines(hypothesis)
    hyp = [w for line in hyp_lines for w in line]
    starts = {sum(len(l) for l in hyp_lines[:k]) for k in range(1, len(hyp_lines))}
    ref = normalize(reference, strict=False).split()
    vocab: dict[str, int] = {}
    ids = ([vocab.setdefault(w, len(vocab)) for w in hyp], [vocab.setdefault(w, len(vocab)) for w in ref])
    out: list[str] = []
    for op in Levenshtein.opcodes(*ids):
        for i in range(op.src_start, op.src_end):
            out.append(('<br>' if i in starts else '') +
                       (html.escape(hyp[i]) if op.tag == 'equal' else f'<span class="sub">{html.escape(hyp[i])}</span>'))
        if op.tag in ('replace', 'insert'):
            out.append(f'<span class="miss">{html.escape(" ".join(ref[op.dest_start:op.dest_end]))}</span>')
    return f'<div class="text">{" ".join(out)}</div>'


def pct(value: float) -> str:
    return f'{100 * value:.2f}'


def describe(config: PipelineConfig) -> str:
    post = config.post
    lexicon = '+'.join(n for n, on in (('domain', post.domain_lexicon), ('generic', post.lexicon)) if on) or 'none'
    text = [n for n, on in (('conventions', post.conventions), ('dehyphenate', post.dehyphenate)) if on]
    return (f"geometry: {','.join(config.geometric_ops) or '–'} | seg image: {','.join(config.seg_ops) or 'original'}"
            f" | rec image: {','.join(config.rec_ops) or 'original'} | model: {config.rec.model}"
            f" (pad {config.rec.padding}) | text: {','.join(text) or '–'} | lexicon: {lexicon}"
            f" (ratio {post.lexicon_max_ratio}, min {post.lexicon_min_len}) | decoder: {post.decoder}"
            + (f' α={post.ctc_alpha}' if post.decoder != 'edit' else ''))


def line_table(trace: PageTrace, rescored: bool, limit: int | None = None) -> str:
    """Each line crop of the recognition image beside the text read from it."""
    head = '<th>line image (recognizer input)</th><th>recognizer output</th>'
    head += '<th>after CTC rescoring (changes vs output)</th>' if rescored else ''
    rows = []
    for i, line in enumerate(trace.segmentation.lines[:limit]):
        crop = trace.rec_image.crop(line_box(line, trace.rec_image.size))
        cells = f'<td><img src="{data_uri(crop, 700)}"></td><td class="text">{i + 1}. {html.escape(trace.greedy[i])}</td>'
        if rescored:
            same = trace.lines[i] == trace.greedy[i]
            cells += '<td class="cap">unchanged</td>' if same else f'<td>{word_diff(trace.lines[i], trace.greedy[i])}</td>'
        rows.append(f'<tr>{cells}</tr>')
    return f'<table class="lines"><tr>{head}</tr>{"".join(rows)}</table>'


def page_card(name: str, page: Page, trace: PageTrace, row: dict[str, Any], config: PipelineConfig) -> str:
    seg_caption = f"segmenter input ({','.join(config.seg_ops) or 'original'}) + {len(trace.segmentation.lines)} lines"
    rec_caption = f"recognizer input ({','.join(config.rec_ops) or 'original'})"
    images = img(overlay(trace.seg_image, trace.segmentation), seg_caption, 700) + img(trace.rec_image, rec_caption, 700)
    stats = (f"WER norm <b>{pct(row['wer_norm'])}</b> · CER norm {pct(row['cer_norm'])} · WER strict "
             f"{pct(row['wer_strict'])} · words {row['pred_words']}/{row['norm_rw']} (pred/GT)")
    rescored = config.post.decoder != 'edit'
    decoded = '\n'.join(trace.lines)
    return (f'<div class="card"><h3>{html.escape(name)} — page {page.number}</h3><div class="cap">{describe(config)}</div>'
            f'<div class="pair"><div class="imgs">{images}</div><div><div>{stats}</div>'
            f'<div class="cap">final text; <span class="sub">struck</span> = wrong, <span class="miss">green</span> = GT</div>'
            f'{word_diff(page.reference_text(), row["text"])}</div></div>'
            f'<details><summary>line by line: image ↔ text</summary>{line_table(trace, rescored)}</details>'
            f'<details><summary>text post-processing: before ↔ after</summary><div class="cap">'
            f'<span class="sub">struck</span> = decoder output, <span class="miss">green</span> = after conventions/'
            f'dehyphenation/lexicon</div>{word_diff(row["text"], decoded)}</details></div>')


def bar_svg(values: dict[str, float], unit: str = '%') -> str:
    """Horizontal bars, one per label; values already in display units."""
    width, row_h, label_w = 640, 22, 260
    top = max(values.values()) or 1
    bars = []
    for i, (label, value) in enumerate(values.items()):
        y, w = i * row_h, (width - label_w - 70) * value / top
        bars.append(f'<text x="{label_w - 6}" y="{y + 15}" text-anchor="end" fill="currentColor">{html.escape(label)}</text>'
                    f'<rect x="{label_w}" y="{y + 3}" width="{w:.1f}" height="{row_h - 6}" fill="var(--acc)" rx="2"/>'
                    f'<text x="{label_w + w + 6:.1f}" y="{y + 15}" fill="currentColor">{value:.2f}{unit}</text>')
    return (f'<svg viewBox="0 0 {width} {row_h * len(values)}" style="max-width:{width}px;width:100%;font-size:12px">'
            f'{"".join(bars)}</svg>')


def summary(results: dict[str, list[dict[str, Any]]], configs: dict[str, PipelineConfig]) -> str:
    from .metrics import corpus

    totals = {name: corpus(rows) for name, rows in results.items()}
    pages = [row['page'] for row in next(iter(results.values()))]
    head = '<tr><th>config</th><th class="num">WER norm</th><th class="num">CER norm</th><th class="num">WER strict</th>'
    head += ''.join(f'<th class="num">p{p}</th>' for p in pages) + '</tr>'
    body = ''.join(f'<tr><td>{html.escape(n)}<div class="cap">{describe(configs[n])}</div></td>'
                   + ''.join(f'<td class="num">{pct(t[k])}</td>' for k in ('wer_norm', 'cer_norm', 'wer_strict'))
                   + ''.join(f'<td class="num">{pct(r["wer_norm"])}</td>' for r in results[n]) + '</tr>'
                   for n, t in totals.items())
    chart = bar_svg({n: 100 * t['wer_norm'] for n, t in totals.items()})
    return f'<h2>Corpus (WER norm, lower is better)</h2>{chart}<table>{head}{body}</table>'


def render(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style>'
            f'</head><body><h1>{html.escape(title)}</h1>{body}</body></html>')


def build_report(engine: Engine, pages: list[Page], configs: dict[str, PipelineConfig], pool: list[Page],
                 title: str = 'OCR pipeline report') -> str:
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in configs}
    sections = []
    for page in pages:
        cards = []
        for name, config in configs.items():
            domain = domain_for(page, pool) if config.post.domain_lexicon else ()
            trace = trace_page(engine, page, config, domain)
            row = score_trace(page, trace, config, domain)
            results[name].append(row)
            cards.append(page_card(name, page, trace, row, config))
        sections.append(f'<h2>Page {page.number}</h2>{"".join(cards)}')
        logger.info('rendered page %d', page.number)
    return render(title, summary(results, configs) + ''.join(sections))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, action='append', required=True, help='repeat to compare configs')
    parser.add_argument('--pages', default='heldout')
    parser.add_argument('--images', type=Path, default=Path('imagens'))
    parser.add_argument('--references', type=Path, default=Path('test_kraken/ground_truth'))
    parser.add_argument('--out', type=Path, default=Path('outputs/viz/report.html'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    all_pages = discover(args.images, args.references)
    configs = {path.stem: load_config(path) for path in args.config}
    engine = Engine(args.cache, args.device, args.seed)
    report = build_report(engine, select(all_pages, args.pages), configs, split(all_pages)[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding='utf-8')
    logger.info('report -> %s', args.out)


if __name__ == '__main__':
    main()
