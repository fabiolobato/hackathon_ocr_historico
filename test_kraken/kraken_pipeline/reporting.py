import json
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import matplotlib.pyplot as plt
from IPython.display import HTML, Markdown, display


def trial_description(row: dict[str, Any]) -> str:
    """Explica os parâmetros ativos de um candidato, omitindo os não usados."""
    trial = row.get('trial')
    if trial is None:
        return 'Controle / configuração do run'
    parameters = trial['preprocessing']
    keys = {'contrast': ('contrast_factor',), 'clahe': ('clahe_clip', 'clahe_grid'),
            'denoise': ('denoise_strength',), 'illumination': ('background_sigma',),
            'upscale': ('upscale_factor',), 'deskew': ('deskew_limit', 'deskew_step'),
            'dewarp': ('dewarp_limit',), 'kraken_nlbin': ('nlbin_threshold',)}
    variant = trial['variant']
    details = [f"segmenter={variant['segmenter']}"]
    for operation in variant['operations']:
        details.extend(f'{key}={parameters[key]}' for key in keys.get(operation, ()))
    return '; '.join(details)


def show_comparison(run_dir: Path) -> list[dict[str, Any]]:
    rows = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    rows.sort(key=lambda row: row['wer'] if row['wer'] is not None else float('inf'))
    baseline = rows[0].get('baseline_variant', 'raw') if rows else 'raw'
    table = [f'| Variante | WER (erro) | Diferença vs {baseline} | Páginas | Com texto | Referência | Parâmetros ativos |',
             '|---|---:|---:|---:|---:|---|---|']
    for row in rows:
        wer = f"{row['wer']:.2%}" if row['wer'] is not None else 'indisponível'
        delta = f"{row['delta_wer'] * 100:+.2f} pp" if row['delta_wer'] is not None else '-'
        table.append(f"| {row['variant']} | {wer} | {delta} | {row['pages_scored']}/{row['pages_expected']} | {row.get('nonempty_pages', '?')} | {row['reference_status']} | {trial_description(row)} |")
    display(Markdown('\n'.join(table)))
    scored = [row for row in rows if row['wer'] is not None]
    if scored:
        fig, ax = plt.subplots(figsize=(9, max(3, len(scored) * 0.4)))
        ax.barh([row['variant'] for row in scored], [100 * row['wer'] for row in scored])
        ax.invert_yaxis()
        ax.set_xlabel('WER (%) - taxa de erro; menor é melhor')
        ax.set_title('WER provisório' if any(row['reference_status'] != 'reviewed' for row in scored) else 'WER')
        fig.tight_layout()
        fig.savefig(run_dir / 'wer_comparison.png', dpi=160)
        plt.show()
        plt.close(fig)
    return rows


def compare_runs(run_dirs: tuple[Path, ...]) -> list[dict[str, Any]]:
    """Compara etapas sem repetir inferências nem misturar conjuntos distintos."""
    from .artifacts import write_json
    from .tuning import _verify_comparison

    if not run_dirs:
        raise ValueError('Informe as execuções que deseja comparar.')
    reference = json.loads((run_dirs[0] / 'config.json').read_text(encoding='utf-8'))
    rows = []
    for directory in run_dirs:
        _verify_comparison(run_dirs[0], directory)
        current = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        if any(reference['config'][key] != current['config'][key] for key in ('scoring', 'seed')):
            raise ValueError('Política de WER ou semente diferente entre execuções.')
        rows.extend(json.loads((directory / 'summary.json').read_text(encoding='utf-8')))
    names = [row['variant'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Variantes repetidas: não misture repetições na mesma comparação.')
    baseline = next((name for name in ('kraken_neural', 'raw') if name in names), None)
    score = next((row['wer'] for row in rows if row['variant'] == baseline), None)
    for row in rows:
        row['baseline_variant'] = baseline
        row['delta_wer'] = row['wer'] - score if row['wer'] is not None and score is not None else None
    destination = run_dirs[-1] / 'comparison'
    destination.mkdir(exist_ok=True)
    write_json(destination / 'summary.json', rows)
    write_json(destination / 'sources.json', [str(path) for path in run_dirs])
    return show_comparison(destination)


def show_page(run_dir: Path, page_stem: str, variant: str = 'raw') -> None:
    page_dir = run_dir / variant / page_stem
    results_path = run_dir / 'results.json'
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding='utf-8'))
        failure = next((row for row in results if row['variant'] == variant
                        and Path(row['page']).stem == page_stem
                        and row['status'] == 'failed'), None)
        if failure is not None:
            raise RuntimeError(f"{variant} failed on {page_stem}: {failure['error']}")
    required = ('input.png', 'segmentation_overlay.png', 'prediction.txt')
    missing = [name for name in required if not (page_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Incomplete page output at {page_dir}: {missing}. Check {results_path} for the original error.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    for axis, filename in zip(axes, ('input.png', 'segmentation_overlay.png')):
        axis.imshow(plt.imread(page_dir / filename))
        axis.set_title(filename)
        axis.axis('off')
    plt.show()
    display(Markdown('**Texto reconhecido**'))
    display((page_dir / 'prediction.txt').read_text(encoding='utf-8'))


def review_reference(image_path: Path, reference_dir: Path) -> None:
    from PIL import Image

    display(Markdown('**GT fornecido: confira com a imagem original**'))
    with Image.open(image_path) as image:
        display(image.copy())
    display((reference_dir / f'{image_path.stem}.txt').read_text(encoding='utf-8'))
    metadata = json.loads((reference_dir / 'review.json').read_text(encoding='utf-8'))
    display(metadata['pages'][image_path.stem])


def show_transcriptions(run_dir: Path, reference_dir: Path) -> None:
    """Mostra GT e predição de todas as páginas, sem ocultar saídas vazias."""
    rows = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    sections = []
    for row in rows:
        stem = Path(row['page']).stem
        reference = reference_dir / f'{stem}.txt'
        prediction = run_dir / row['variant'] / stem / 'prediction.txt'
        gt = reference.read_text(encoding='utf-8') if reference.is_file() else '[GT ausente]'
        text = prediction.read_text(encoding='utf-8') if prediction.is_file() else '[Saída ausente]'
        wer = f"{row['wer']:.2%}" if row.get('wer') is not None else 'indisponível'
        label = escape(f"{row['variant']} | {row['page']} | WER: {wer} | linhas: {row.get('detected_lines', '?')}")
        sections.append(f'<details><summary>{label}</summary><b>GT</b>'
                        f'<pre style="white-space:pre-wrap">{escape(gt)}</pre>'
                        f'<b>Reconhecimento</b><pre style="white-space:pre-wrap">'
                        f'{escape(text) if text.strip() else "[TEXTO VAZIO]"}</pre></details>')
    display(HTML(''.join(sections)))


def _display_prediction(directory: Path) -> None:
    prediction = directory / 'prediction.txt'
    text = prediction.read_text(encoding='utf-8') if prediction.is_file() else '[Saída ausente]'
    content = escape(text) if text.strip() else '[TEXTO VAZIO]'
    display(HTML(f'<b>Texto predito</b><pre style="white-space:pre-wrap">{content}</pre>'))


def show_stages(run_dir: Path, page_stem: str, variants: tuple[str, ...] | None = None) -> None:
    """Exibe cada operação, segmentação final e o texto predito correspondente."""
    rows = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    for row in rows:
        if Path(row['page']).stem != page_stem or variants is not None and row['variant'] not in variants:
            continue
        directory = run_dir / row['variant'] / page_stem
        stages = sorted((directory / 'stages').glob('*.png'))
        paths = stages or [directory / 'input.png']
        paths += [directory / 'segmentation_overlay.png']
        paths = [path for path in paths if path.is_file()]
        display(Markdown(f"**{row['variant']}** — {row['status']} — {row.get('diagnostic', row.get('error', ''))}"))
        if not paths:
            _display_prediction(directory)
            continue
        fig, axes = plt.subplots(1, len(paths), figsize=(4 * len(paths), 6), squeeze=False)
        for axis, path in zip(axes[0], paths):
            axis.imshow(plt.imread(path), cmap='gray')
            axis.set_title(path.stem)
            axis.axis('off')
        fig.tight_layout()
        plt.show()
        plt.close(fig)
        _display_prediction(directory)


def export_report(run_dir: Path, reference_dir: Path) -> Path:
    """Salva comparação textual e visual navegável sem executar novamente o OCR."""
    rows = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    summary = show_comparison(run_dir)
    sections = ['<!doctype html><meta charset="utf-8"><title>Comparação Kraken</title>',
                '<style>body{font:16px sans-serif;max-width:1400px;margin:2em auto} '
                '.texts{display:grid;grid-template-columns:1fr 1fr;gap:1em} '
                'pre{white-space:pre-wrap}img{max-width:100%;max-height:700px} '
                'table{border-collapse:collapse}td,th{padding:.4em;border:1px solid #ccc}</style>',
                '<h1>Kraken: resultados provisórios</h1><p>WER é taxa de erro; menor é melhor. '
                'Saída vazia permanece na avaliação. GT fornecido ainda não integralmente revisado.</p>',
                '<img src="wer_comparison.png" alt="WER por variante"><table><tr><th>Variante</th>'
                '<th>WER</th><th>Páginas com texto</th></tr>']
    for row in summary:
        score = f"{row['wer']:.2%}" if row['wer'] is not None else 'indisponível'
        sections.append(f"<tr><td>{escape(row['variant'])}</td><td>{score}</td>"
                        f"<td>{row.get('nonempty_pages', '?')}/{row['pages_expected']}</td></tr>")
    sections.append('</table>')
    for row in rows:
        stem = Path(row['page']).stem
        directory = run_dir / row['variant'] / stem
        prediction = directory / 'prediction.txt'
        reference = reference_dir / f'{stem}.txt'
        text = prediction.read_text(encoding='utf-8') if prediction.is_file() else '[Saída ausente]'
        gt = reference.read_text(encoding='utf-8') if reference.is_file() else '[GT ausente]'
        sections.append(f"<details><summary>{escape(row['variant'] + ' | ' + row['page'])} "
                        f"| WER={row.get('wer')} | {escape(row.get('diagnostic', row['status']))}</summary>")
        paths = sorted((directory / 'stages').glob('*.png')) or [directory / 'input.png']
        for path in paths + [directory / 'segmentation_overlay.png']:
            if path.is_file():
                sections.append(f'<figure><figcaption>{escape(path.stem)}</figcaption>'
                                f'<img loading="lazy" src="{quote(path.relative_to(run_dir).as_posix())}"></figure>')
        sections.append(f'<div class="texts"><section><h3>GT</h3><pre>{escape(gt)}</pre></section>'
                        f'<section><h3>OCR</h3><pre>{escape(text) if text.strip() else "[TEXTO VAZIO]"}'
                        '</pre></section></div></details>')
    destination = run_dir / 'report.html'
    destination.write_text('\n'.join(sections), encoding='utf-8')
    return destination
