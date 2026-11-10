"""Demonstração: uma folha escrita à caneta, fotografada com o celular, lida pelos dois OCRs do projeto.

A foto, copiada do celular para dados/demo/, é lida pelos dois OCRs, sempre os mesmos, sem opções. A única
preparação é a redução à escala das 12 páginas do livro (a folha deve preencher o quadro):

  pipeline  o pipeline final (configs/final_v6.json), local, sem LLM: a folha reduzida à escala das 12 páginas do
            livro entra, sem nenhum ajuste, na mesma configuração que transcreve o livro.
  llm       o melhor LLM da avaliação do projeto (Transcription_Evaluation.ipynb): Gemini 3.8 Flash com esforço de
            raciocínio low, sobre a foto, pelo motor ocr_historico.motor_llm (OpenRouter). É pago:
            nada é enviado sem --permitir-gasto; sem ele a requisição é montada e gravada, mas não sai da máquina.

O cartão abre com uma página do livro (BOOK_PAGE, uma das páginas reservadas) e a pergunta "você consegue ler?";
dois botões revelam o que o pipeline final leu nela e a transcrição de referência, e um terceiro destaca as
palavras que diferem. Depois vem a foto passando por cada etapa (a redução, depois as operações de imagem do
pipeline e a segmentação) e as duas leituras lado a lado. A interface principal é o caderno demo_ao_vivo.ipynb;
pela linha de comando:

  python -m ocr_pipeline.demo --vigiar --abrir                                        # dados/demo/: cada foto nova abre um cartão
  python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg
  python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg --permitir-gasto          # envia de verdade ao LLM (pago)
  python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg --exemplo-llm             # ensaio: o cartão do LLM com um texto de exemplo, nada enviado

Saída em --out (padrão outputs/demo): entrada/ (o que cada OCR recebeu), textos/ (<id>.pipeline.txt, <id>.llm.txt),
cartoes/ (<id>.html), llm/ (requisição e resposta, sem a imagem) e cache/ (segmentação e reconhecimento do pipeline,
com chaves pelo nome único de cada foto).
"""
import argparse
import copy
import html
import itertools
import json
import logging
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .presentation import style
from .visualize import data_uri

logger = logging.getLogger(__name__)

PIPELINE_CONFIG = Path('configs/final_v6.json')  # the final pipeline, as it transcribes the book
LLM_MODEL = 'google/gemini-3.8-flash'  # best run of Transcription_Evaluation.ipynb: WER 11.8 / CER 4.0 (pré-processadas),
LLM_EFFORT = 'low'  # 12.3 / 4.3 on the cropped pages; here it reads the photo itself, reduced to LLM_LONG_SIDE
LLM_EXAMPLE_TEXT = ('olá, teste teste. texto em cursivo que, com sorte, [ilegível] será lido pelo pipeline. '
                    'TAMOIO® Boas Impressões')  # what Gemini 3.8 Flash really read on dados/demo/IMG20261105115033.jpg
LLM_EXAMPLE_COST = 0.0033  # (2026-11-05, US$ 0.0033): the stand-in text and cost of the example mode, which
# rehearses the card without sending anything
LLM_EXAMPLE_DATE = '05/11/2026'
PHOTOS = Path('dados/demo')  # where the photos land (in the project, but git-ignored: only the folder's README is versioned)
OUT = Path('outputs/demo')
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.heic', '.heif', '.webp', '.bmp')
BOOK_LONG_SIDE = 880  # the 12 project pages are ~650 x 880 px: the pipeline gets the sheet at that scale
LLM_LONG_SIDE = 1600  # the LLM is not tied to the book's scale: it gets the sheet larger
PREPARACAO = (f'só a redução de escala: lado maior {BOOK_LONG_SIDE} px para o pipeline, {LLM_LONG_SIDE} px para o '
              'LLM; nada é recortado nem endireitado')  # shown by the notebook, so a stale kernel is easy to spot
BOOK_PAGE = 10  # opens the card: a held-out page (7, 10 and 12 took part in no choice of the pipeline)
BOOK_IMAGES = Path('dados/Imagens Originais')  # the photos of the book as they were received
BOOK_TEXTS = Path('entrega_ocr/resultados/textos')  # the final pipeline's text of each page (pagina_NN.txt)
BOOK_REFERENCES = Path('entrega_ocr/ground_truth')  # the reference transcription of each page, named like its image
BOOK_LLM = Path('dados/LLM/imagens-pré-processadas-gemini-3.8-flash.json')  # the best LLM run of the evaluation
# (Transcription_Evaluation.ipynb: Gemini 3.8 Flash, effort low, on the preprocessed pages): text and cost per page

try:  # iPhone photos: HEIC opens like any other format when pillow-heif is installed (optional)
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - optional dependency
    pass


# --------------------------------------------------------------------------------------------------------------
# Preparing the photo: only the reduction to the book's scale
# --------------------------------------------------------------------------------------------------------------

@dataclass
class Prepared:
    original: Image.Image  # the photo as it came (its orientation tag applied, as any viewer does)
    full: Image.Image  # what the LLM gets: the photo itself (the LLM engine reduces it to LLM_LONG_SIDE)
    sheet: Image.Image  # what the pipeline gets: the photo reduced to the book's scale
    steps: list[tuple[str, Image.Image]]  # (label, image) of each preparation step, for the card


def prepare_photo(source: Path | Image.Image, long_side: int = BOOK_LONG_SIDE) -> Prepared:
    """The only preparation: reduce the photo to the book's scale (the pipeline's input). Nothing is cropped or
    straightened, so the sheet should fill the frame. The LLM gets the photo before this step."""
    image = Image.open(source) if isinstance(source, Path) else source
    original = ImageOps.exif_transpose(image).convert('RGB')
    steps: list[tuple[str, Image.Image]] = [('foto', original)]
    sheet = original
    if long_side and max(sheet.size) > long_side:
        factor = long_side / max(sheet.size)
        sheet = sheet.resize((round(sheet.width * factor), round(sheet.height * factor)), Image.Resampling.LANCZOS)
        steps.append((f'+ redução à escala do pipeline ({sheet.width}×{sheet.height} px): entrada do pipeline', sheet))
    return Prepared(original, original, sheet, steps)


# --------------------------------------------------------------------------------------------------------------
# The two OCRs
# --------------------------------------------------------------------------------------------------------------

@dataclass
class BookPage:
    """A page of the book for the opening of the card: its photo, what the final pipeline read, the reference."""
    number: int
    image: Image.Image
    pipeline_text: str
    reference: str
    llm_text: str | None = None  # what the best LLM run of the evaluation read on this page, if recorded
    llm_cost: float | None = None  # in US$, as the run recorded it
    llm_label: str = ''  # model and effort of that run


def load_book_page(number: int = BOOK_PAGE, images: Path = BOOK_IMAGES, texts: Path = BOOK_TEXTS,
                   references: Path = BOOK_REFERENCES, llm: Path | None = BOOK_LLM) -> BookPage:
    """The page's photo from the data folder, its two texts from the delivery (the final pipeline's result,
    resultados/textos/pagina_NN.txt, and the reference, ground_truth/<image name>.txt) and, when the evaluation's
    record is there, the LLM's reading of the page with its cost."""
    image_file = next((f for f in sorted(images.iterdir()) if is_image(f) and f.stem.split('.')[0] == str(number)),
                      None)
    if image_file is None:
        raise FileNotFoundError(f'página {number} do livro não encontrada em {images}')
    pipeline_text = (texts / f'pagina_{number:02d}.txt').read_text(encoding='utf-8').strip()
    reference = (references / f'{image_file.stem}.txt').read_text(encoding='utf-8').strip()
    page = BookPage(number, Image.open(image_file).convert('RGB'), pipeline_text, reference)
    if llm is not None and llm.is_file():
        run = json.loads(llm.read_text(encoding='utf-8'))
        item = next((i for i in run.get('imagens', []) if i.get('arquivo', '').split('.')[0] == str(number)), None)
        if item and item.get('texto_transcrito'):
            page.llm_text, page.llm_cost = item['texto_transcrito'].strip(), item.get('custo')
            page.llm_label = f"{run.get('modelo', LLM_MODEL)}, esforço {run.get('esforco_raciocinio', LLM_EFFORT)}"
    return page


def money(cost: float) -> str:
    return f'US$ {cost:.4f}'


@dataclass
class Reading:
    """What one OCR made of one photo."""
    engine: str  # 'pipeline' or 'llm'
    title: str
    text: str
    seconds: float
    details: str  # one line for the card: lines, words, time, cost...
    section: str  # HTML fragment with everything the OCR can show (needs presentation.style())
    text_file: Path | None
    error: str | None = None
    input_image: Image.Image | None = None  # what this OCR received, when it is not already among the steps
    input_label: str = ''
    example: bool = False  # the text is a stand-in (example mode of the LLM), not a reading
    cost: float | None = None  # in US$ (the LLM: what OpenRouter charged, or the reference cost of the example)


class PipelineOCR:
    """The project's final pipeline, run exactly as it reads a page of the book."""

    name = 'pipeline'
    remote = False

    def __init__(self, out: Path, device: str = 'cpu', seed: int = 42, config_path: Path = PIPELINE_CONFIG) -> None:
        from .batch import check_config
        from .config import load_config
        from .engine import Engine

        self.config_path = config_path
        self.config = load_config(config_path)
        check_config(self.config)
        self.out = out
        self.engine = Engine(out / 'cache', device, seed)
        self.counter = 0

    def warm_up(self) -> None:
        """Load the segmenter, the recognizer, the lexicon and the context model before the first photo arrives."""
        from .postprocess import lexicon_choices

        started = time.perf_counter()
        self.engine._segmenter(self.config.seg.model)
        self.engine._recognizer(self.config.rec.model)
        lexicon_choices(self.config.post, ())
        if self.config.post.context_lm:
            from .context_lm import load_lm

            load_lm(self.config.post.context_lm)
        logger.info('pipeline: modelos carregados em %.1f s', time.perf_counter() - started)

    def read(self, job_id: str, prepared: Prepared) -> Reading:
        from .batch import BatchPage, transcribe

        entrada = self.out / 'entrada' / f'{job_id}.pipeline.jpg'  # unique name: unique cache keys
        prepared.sheet.save(entrada, quality=95)
        self.counter += 1
        page = BatchPage(self.counter, entrada, None)
        started = time.perf_counter()
        trace, text = transcribe(self.engine, page, self.config)
        seconds = time.perf_counter() - started
        text_file = self.out / 'textos' / f'{job_id}.pipeline.txt'
        text_file.write_text(text + '\n', encoding='utf-8')
        details = (f'{len(trace.segmentation.lines)} linhas · {len(text.split())} palavras · {seconds:.1f} s · '
                   f'{prepared.sheet.width}×{prepared.sheet.height} px')
        return Reading(self.name, f'pipeline local ({self.config_path.name}), sem LLM', text, seconds, details,
                       pipeline_section(page, trace, self.config), text_file)


def pipeline_section(page: Any, trace: Any, config: Any, width: int = 1200) -> str:
    """The pipeline's own card, with the same content as batch.card_html (every image operation, the segmentation,
    the text after each stage, line by line) but larger figures, so they can be enlarged and compared on the page."""
    from .presentation import image_steps, text_versions
    from .visualize import img, line_table, overlay

    parts = ['<div class="card">']
    for title, steps in image_steps(page, config).items():
        figures = [img(image, label, width) for label, image in steps] or \
                  ['<div class="cap">nenhuma operação: usa a imagem da geometria</div>']
        if title == 'imagem de segmentação':
            figures.append(img(overlay(trace.seg_image, trace.segmentation),
                               f'= segmentação: {len(trace.segmentation.lines)} linhas (polígono + linha de base)', width))
        parts.append(f'<h4>{html.escape(title)}</h4><div class="steps">{"".join(figures)}</div>')
    for label, text in text_versions(trace, config, ()):
        parts.append(f'<h4>{html.escape(label)}</h4><div class="text">{html.escape(text).replace(chr(10), "<br>")}</div>')
    parts.append('<details><summary>linha a linha: recorte da imagem ↔ texto lido</summary>'
                 f'{line_table(trace, config.post.decoder != "edit")}</details></div>')
    return ''.join(parts)


def without_image(body: dict[str, Any]) -> dict[str, Any]:
    """A copy of an OpenRouter request with the image data URL replaced by its size (for the record on disk)."""
    body = copy.deepcopy(body)
    for message in body.get('messages', ()):
        for part in message.get('content', ()) if isinstance(message.get('content'), list) else ():
            if part.get('type') == 'image_url':
                url = part['image_url']['url']
                part['image_url']['url'] = f'<data URL PNG: {len(url)} caracteres>'
    return body


class LLMOCR:
    """The project's LLM engine (ocr_historico.motor_llm, OpenRouter) with the model and effort that scored best in
    Transcription_Evaluation.ipynb, on the photo itself. Paid: it only sends with allow_spend=True; otherwise it
    builds and records the request without sending it (simulation)."""

    name = 'llm'
    remote = True

    def __init__(self, out: Path, allow_spend: bool = False, long_side: int = LLM_LONG_SIDE,
                 example: str | None = None) -> None:
        from ocr_historico import motor_llm

        self.motor_llm = motor_llm
        self.motor = motor_llm.MotorLLM(LLM_MODEL, LLM_EFFORT, permitir_gasto=allow_spend, simular=not allow_spend)
        self.out = out
        self.long_side = long_side
        self.example = None if allow_spend else (example or None)  # simulation only: a stand-in text on the card

    @property
    def simulate(self) -> bool:
        return self.motor.simular

    def check(self) -> None:
        """Fail before the first photo: no sending without a key."""
        if not self.simulate and self.motor_llm.obter_chave() is None:
            raise self.motor_llm.GastoNaoAutorizado(
                'Chave não encontrada: defina OPENROUTER_API_KEY (ou API_KEY) no ambiente, o segredo API_KEY do '
                'Colab ou um .env na raiz do repositório.')

    def image_for(self, prepared: Prepared) -> Image.Image:
        image = prepared.full
        if self.long_side and max(image.size) > self.long_side:
            factor = self.long_side / max(image.size)
            image = image.resize((round(image.width * factor), round(image.height * factor)),
                                 Image.Resampling.LANCZOS)
        return image

    def read(self, job_id: str, prepared: Prepared) -> Reading:
        image = self.image_for(prepared)
        entrada = self.out / 'entrada' / f'{job_id}.llm.png'  # PNG: exactly what motor_llm encodes and sends
        image.save(entrada)
        started = time.perf_counter()
        text, cost = self.motor.transcrever(image)
        seconds = time.perf_counter() - started
        if self.example:  # simulating: the request was built and recorded, the card shows the stand-in text and cost
            text, cost = self.example, LLM_EXAMPLE_COST
        elif self.simulate:
            cost = None  # nothing was sent, so nothing was charged: no cost on the card
        text_file = self.out / 'textos' / f'{job_id}.llm.txt'
        text_file.write_text(text + '\n', encoding='utf-8')
        record: dict[str, Any] = {
            'id': job_id, 'modelo': self.motor.modelo, 'esforco': self.motor.esforco, 'simulado': self.simulate,
            'exemplo': bool(self.example),
            'imagem': entrada.name, 'largura': image.width, 'altura': image.height,
            'prompt': self.motor_llm.PROMPT_TRANSCRICAO, 'segundos': round(seconds, 2), 'custo_usd': cost,
            'texto': text}
        if self.simulate:
            record['requisicao'] = without_image(self.motor.requisicoes_simuladas[-1])
        (self.out / 'llm' / f'{job_id}.json').write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                          encoding='utf-8')
        size = f'{image.width}×{image.height} px'
        if self.example:
            details = (f'exemplo: texto ilustrativo no lugar da leitura, nada enviado · {size} · {money(cost)} é o custo '
                       f'da leitura real de {LLM_EXAMPLE_DATE}')
        elif self.simulate:
            details = f'simulação: requisição montada e gravada, nada enviado · {size}'
        else:
            details = f'{len(text.split())} palavras · {seconds:.1f} s · {size}'
            if cost is not None:
                details += f' · {money(cost)}'
        title = f'LLM: {self.motor.modelo}, esforço {self.motor.esforco} (OpenRouter)'
        return Reading(self.name, title, text, seconds, details, llm_section(title, record), text_file,
                       input_image=image, input_label='imagem enviada ao LLM', example=bool(self.example), cost=cost)


def llm_section(title: str, record: dict[str, Any]) -> str:
    """The prompt and the answer."""
    status = ('exemplo: texto ilustrativo, nada foi enviado' if record.get('exemplo')
              else 'simulação: nada foi enviado' if record['simulado'] else 'resposta do modelo')
    parts = [f'<div class="card"><h3>{html.escape(title)}</h3>',
             f'<div class="cap">{record["largura"]}×{record["altura"]} px enviados · {status}</div>',
             f'<h4>prompt</h4><div class="text">{html.escape(record["prompt"])}</div>',
             f'<h4>{status}</h4><div class="text">{html.escape(record["texto"])}</div>']
    if record.get('custo_usd') is not None and record.get('exemplo'):
        parts.append(f'<div class="cap">custo de referência, o da leitura real de {LLM_EXAMPLE_DATE}: '
                     f'{money(record["custo_usd"])}</div>')
    elif record.get('custo_usd') is not None and not record['simulado']:
        parts.append(f'<div class="cap">custo informado pelo OpenRouter: {money(record["custo_usd"])}</div>')
    return ''.join(parts) + '</div>'


def failed_reading(engine: Any, error: BaseException) -> Reading:
    message = f'{type(error).__name__}: {error}'
    title = {'pipeline': 'pipeline local', 'llm': 'LLM'}.get(engine.name, engine.name)
    section = f'<div class="card"><h3>{html.escape(title)}</h3><div class="text">falhou: {html.escape(message)}</div></div>'
    return Reading(engine.name, title, '', 0.0, 'falhou', section, None, error=message)


# --------------------------------------------------------------------------------------------------------------
# One photo, both OCRs, one card
# --------------------------------------------------------------------------------------------------------------

@dataclass
class DemoResult:
    job_id: str
    prepared: Prepared
    readings: list[Reading]
    card: str  # HTML fragment (needs presentation.style() and DEMO_CSS): what the notebook displays inline
    card_file: Path  # the same card as a standalone page


class Demo:
    """Prepares each photo once and hands it to every OCR; one card per photo."""

    FOLDERS = ('entrada', 'textos', 'cartoes', 'llm')

    def __init__(self, out: Path, engines: list[Any], open_cards: bool = False, book: BookPage | None = None) -> None:
        self.out = out.resolve()
        self.engines = list(engines)
        self.open_cards = open_cards
        self.book = book  # the page of the book that opens every card (None: the card starts at the photo)
        self.counter = 0
        self._folders()

    def _folders(self) -> None:
        for name in self.FOLDERS:
            (self.out / name).mkdir(parents=True, exist_ok=True)

    def warm_up(self) -> None:
        for engine in self.engines:
            warm = getattr(engine, 'warm_up', None)
            if warm:
                warm()

    def new_job_id(self) -> str:
        self.counter += 1
        return f'{datetime.now():%Y%m%d_%H%M%S}_{self.counter:03d}_{uuid.uuid4().hex[:6]}'

    def run(self, photo: Path | Image.Image) -> DemoResult:
        """photo -> prepared sheet -> every OCR (the remote one in parallel with the local one) -> card."""
        job_id = self.new_job_id()
        started = time.perf_counter()
        self._folders()  # the output folder may have been cleaned while the demo was running
        prepared = prepare_photo(photo)
        if not isinstance(photo, Path):  # a photo given in memory: keep it, the others stay where they landed
            prepared.original.save(self.out / 'entrada' / f'{job_id}.foto.jpg', quality=92)

        readings: dict[int, Reading] = {}

        def work(index: int, engine: Any) -> None:
            try:
                readings[index] = engine.read(job_id, prepared)
            except Exception as error:  # noqa: BLE001 - one OCR failing must not hide the other's result
                logger.exception('%s: %s falhou', job_id, engine.name)
                readings[index] = failed_reading(engine, error)

        threads = [threading.Thread(target=work, args=(i, e), daemon=True)
                   for i, e in enumerate(self.engines) if getattr(e, 'remote', False)]
        for thread in threads:
            thread.start()
        for index, engine in enumerate(self.engines):
            if not getattr(engine, 'remote', False):
                work(index, engine)
        for thread in threads:
            thread.join()
        ordered = [readings[i] for i in range(len(self.engines))]
        name = photo.name if isinstance(photo, Path) else job_id
        card = demo_card(name, prepared, ordered, time.perf_counter() - started, self.book)
        card_file = self.out / 'cartoes' / f'{job_id}.html'
        card_file.write_text(standalone_html(f'{name} · demonstração', card), encoding='utf-8')
        for reading in ordered:
            logger.info('%s: %s: %s', job_id, reading.engine, reading.details)
        if self.open_cards:
            webbrowser.open(card_file.as_uri())
        return DemoResult(job_id, prepared, ordered, card, card_file)


def book_section(book: BookPage, number: int) -> str:
    """The opening: a page of the book beside two covered cards. One reveals what the final pipeline read in it,
    the other the reference transcription; with both open, a button marks the words that differ."""
    def panel(kind: str, title: str, text: str, button: str, footer: str) -> str:
        text = ' '.join(text.split())  # running text in every card: the three columns then wrap alike
        return (f'<article class="demo-reading demo-covered {kind}" data-panel="{kind}"><header>'
                f'<span class="demo-tag">{"LLM" if kind == "llm" else kind}</span><span class="demo-title">{title}'
                f'</span></header><div class="demo-cover"><button type="button" class="demo-btn {kind}" '
                f'data-reveal="{kind}">{button}</button></div><div class="demo-text" hidden>{html.escape(text)}</div>'
                f'<footer class="demo-dim">{html.escape(footer)}</footer></article>')

    figure = (f'<figure class="demo-step demo-book-page"><img src="{data_uri(book.image, 1400)}" alt="">'
              f'<figcaption><b>página {book.number} do livro</b><br><span>{book.image.width}×{book.image.height} px'
              f' · clique para ampliar</span></figcaption></figure>')
    panels = panel('pipeline', 'o que o pipeline final leu nesta página', book.pipeline_text,
                   'Revelar a leitura do pipeline', 'roda local, sem LLM · custo por página: nenhum')
    if book.llm_text:
        cost = f' · {money(book.llm_cost)} por esta página' if book.llm_cost is not None else ''
        panels += panel('llm', 'o que o LLM leu nesta página', book.llm_text, 'Revelar a leitura do LLM',
                        f'{book.llm_label}, sobre a página pré-processada, na avaliação do projeto{cost}')
    panels += panel('real', 'a transcrição de referência da página', book.reference, 'Revelar a transcrição real',
                    'o gabarito usado para medir os dois OCRs')
    tools = ('<div class="demo-book-tools"><button type="button" class="demo-btn ghost" data-reveal="diff" disabled '
             'aria-pressed="false">Destacar diferenças</button><button type="button" class="demo-btn ghost" '
             'data-reveal="hide">Esconder de novo</button><span class="demo-diff-key demo-dim" hidden></span></div>')
    return (f'<section id="livro" class="demo-section demo-book"><h2><span class="demo-num">{number}</span>'
            f'Você consegue ler esta página?</h2><p class="demo-lead">Página {book.number} do Livro de Registro de '
            f'Leis nº 1 (1892–1894) da Fundação Pró-Memória de São Carlos, uma das páginas que não entraram em '
            f'nenhuma escolha do pipeline. Tente ler; depois revele o que {"cada OCR leu" if book.llm_text else "o pipeline leu"} '
            f'e a transcrição real.</p>'
            f'<div class="demo-body"><div class="demo-book-grid">{figure}<div class="demo-book-panels">{tools}{panels}'
            f'</div></div></div><script>{BOOK_SCRIPT}</script></section>')


def demo_card(name: str, prepared: Prepared, readings: list[Reading], seconds: float,
              book: BookPage | None = None) -> str:
    """The page body: a page of the book to read (if given), the original photo forking into one reduced input
    per OCR, the readings side by side, then each OCR's own section (the pipeline's shows its image operations
    and the segmentation)."""
    def figure(image: Image.Image, caption: str) -> str:
        return (f'<figure class="demo-step"><img src="{data_uri(image, 1400)}" alt=""><figcaption>'
                f'<b>{html.escape(caption)}</b><br><span>{image.width}×{image.height} px</span></figcaption></figure>')

    branches = []  # each OCR gets its own reduction, made from the original photo, never from the other one
    for reading in readings:
        if reading.engine == 'pipeline':
            arrow = ('redução à escala do pipeline' if prepared.sheet.size != prepared.original.size
                     else 'sem redução: já está na escala do pipeline')
            branches.append(('pipeline', arrow, 'entrada do pipeline', prepared.sheet))
        elif reading.input_image is not None:
            branches.append(('llm', 'redução para o LLM', reading.input_label, reading.input_image))
    fork = [f'<div class="demo-origin">{figure(prepared.original, "foto original")}</div>', '<div class="demo-branches">']
    for engine, arrow, caption, image in branches:
        fork.append(f'<div class="demo-branch {engine}"><div class="demo-arrow" aria-hidden="true"><span>→</span>'
                    f'<small>{html.escape(arrow)}</small></div>{figure(image, caption)}</div>')
    fork.append('</div>')
    count = {1: 'uma entrada', 2: 'duas entradas'}.get(len(branches), f'{len(branches)} entradas')
    cards = []
    for reading in readings:
        body = html.escape(reading.text or reading.error or '').replace(chr(10), '<br>')
        tags = f'<span class="demo-tag">{"LLM" if reading.engine == "llm" else "pipeline"}</span>'
        tags += '<span class="demo-tag example">exemplo</span>' if reading.example else ''
        cost = (f'<span class="demo-cost" title="custo desta leitura">{money(reading.cost)}</span>'
                if reading.cost is not None else '')
        cards.append(f'<article class="demo-reading {reading.engine}"><header>{tags}'
                     f'<span class="demo-title">{html.escape(reading.title)}</span>{cost}</header>'
                     f'<div class="demo-text">{body}</div>'
                     f'<footer class="demo-dim">{html.escape(reading.details)}</footer></article>')
    numbers = itertools.count(1)

    def section(anchor: str, classes: str, title: str, lead: str, body: str) -> str:
        """One slide: number and title, an optional lead, and the body, which scrolls inside the slide."""
        lead = f'<p class="demo-lead">{lead}</p>' if lead else ''
        return (f'<section id="{anchor}" class="demo-section {classes}"><h2><span class="demo-num">{next(numbers)}'
                f'</span>{title}</h2>{lead}<div class="demo-body">{body}</div></section>')

    sections = [book_section(book, next(numbers))] if book else []
    sections += [
        section('foto', '', f'A foto original vira {count}',
                'Cada OCR recebe a sua própria redução da foto original; nenhuma é feita a partir da outra, e nada '
                'é recortado.', f'<div class="demo-fork">{"".join(fork)}</div>'),
        section('leituras', '', 'As leituras' if len(readings) > 1 else 'A leitura', '',
                f'<div class="demo-grid cols{len(readings)}">{"".join(cards)}</div>')]
    for reading in readings:
        title = 'Dentro do pipeline' if reading.engine == 'pipeline' else 'O LLM'
        sections.append(section(reading.engine, f'demo-inner {reading.engine}', title, '', reading.section))
    links = ([('livro', 'livro')] if book else []) + [('foto', 'foto'), ('leituras', 'leituras')]
    links += [(r.engine, 'LLM' if r.engine == 'llm' else 'pipeline') for r in readings]
    nav = ''.join(f'<a href="#{anchor}" class="{anchor}">{label}</a>' for anchor, label in links)
    legend = ''.join(f'<span class="demo-key {r.engine}"><i></i>{html.escape(r.title)}</span>' for r in readings)
    header = ('<header class="demo-top"><div><h1>Uma folha escrita à mão, '
              f'{"dois OCRs" if len(readings) > 1 else "um OCR"}</h1>'
              f'<div class="demo-legend">{legend}</div></div><nav class="demo-nav">{nav}'
              f'<button type="button" class="demo-theme" title="alternar entre o tema escuro (tinta) e o claro (papel)">'
              f'tema</button></nav></header>')
    return header + ''.join(sections)


DEMO_CSS = """
:root { --top:100px; }  /* height of the fixed bar; measured by the page's script */
html { scroll-snap-type:y mandatory; scroll-padding-top:var(--top); }
/* the light theme is paper: warm background, the three OCR colours and one colour per section */
.demo { max-width:1700px; font-size:16px; padding:0 12px; min-height:100vh;
        --pipe:#2b6cb0; --llm:#b7791f; --real:#2f855a; --livro:#c05621; --foto:#6b46c1; --acc:#2b6cb0;
        --pipe-soft:rgba(43,108,176,.28); --llm-soft:rgba(183,121,31,.38);
        --bg:#f6efe1; --fg:#1f1d1a; --muted:#6c665a; --card:#fffcf5; --line:#dfd3bb; --ok:#1f1d1a;
        --sub-bg:#fde2e1; --sub-fg:#a4161a; --miss-bg:#dff3e3; --miss-fg:#1b6e33;
        --glow1:rgba(43,108,176,.18); --glow2:rgba(183,121,31,.20); --glow3:rgba(47,133,90,.16);
        color:var(--fg);
        background:radial-gradient(1100px 700px at 0% 0%, var(--glow1), transparent 60%),
                   radial-gradient(1100px 700px at 100% 100%, var(--glow2), transparent 60%),
                   radial-gradient(800px 600px at 100% 0%, var(--glow3), transparent 60%), var(--bg);
        background-attachment:fixed; }
/* the dark theme is ink: the same colours, lighter, over a deep blue */
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .demo { --pipe:#7fb2f0; --llm:#e2b35c; --real:#6fcf97; --livro:#ee9a7a; --foto:#b39df3; --acc:#7fb2f0;
        --pipe-soft:rgba(127,178,240,.32); --llm-soft:rgba(226,179,92,.4);
        --bg:#131826; --fg:#eef0f6; --muted:#a3abbe; --card:#1b2132; --line:#303a52; --ok:#eef0f6;
        --sub-bg:#4a1d1d; --sub-fg:#ffb3ae; --miss-bg:#173a22; --miss-fg:#9be0ad;
        --glow1:rgba(127,178,240,.22); --glow2:rgba(226,179,92,.18); --glow3:rgba(111,207,151,.14); } }
:root[data-theme="dark"] .demo { --pipe:#7fb2f0; --llm:#e2b35c; --real:#6fcf97; --livro:#ee9a7a; --foto:#b39df3; --acc:#7fb2f0;
        --pipe-soft:rgba(127,178,240,.32); --llm-soft:rgba(226,179,92,.4);
        --bg:#131826; --fg:#eef0f6; --muted:#a3abbe; --card:#1b2132; --line:#303a52; --ok:#eef0f6;
        --sub-bg:#4a1d1d; --sub-fg:#ffb3ae; --miss-bg:#173a22; --miss-fg:#9be0ad;
        --glow1:rgba(127,178,240,.22); --glow2:rgba(226,179,92,.18); --glow3:rgba(111,207,151,.14); }
.demo-top { display:flex; justify-content:space-between; align-items:center; gap:24px; flex-wrap:wrap; padding:18px 12px 14px;
            position:sticky; top:0; z-index:2; margin:0 -12px;
            background:color-mix(in srgb, var(--bg) 84%, transparent); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
.demo-top::after { content:""; position:absolute; left:0; right:0; bottom:0; height:4px;
                   background:linear-gradient(90deg, var(--livro), var(--foto), var(--pipe), var(--real), var(--llm)); }
.demo h1 { font-size:1.5rem; margin:0; font-weight:600; }
.demo-legend { display:flex; gap:12px; flex-wrap:wrap; margin-top:8px; font-size:.92rem; }
.demo-key { display:inline-flex; align-items:center; padding:3px 11px 3px 7px; border-radius:999px; --c:var(--muted);
            background:color-mix(in srgb, var(--c) 16%, transparent); color:var(--fg); }
.demo-key.pipeline { --c:var(--pipe); } .demo-key.llm { --c:var(--llm); }
.demo-key i { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:8px; background:var(--c); }
.demo-nav { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.demo-nav a { --c:var(--fg); color:var(--c); background:color-mix(in srgb, var(--c) 12%, transparent); padding:5px 13px;
              border-radius:999px; text-decoration:none; font-size:.95rem; font-weight:600; }
.demo-nav a:hover { background:color-mix(in srgb, var(--c) 26%, transparent); }
.demo-nav a.livro { --c:var(--livro); } .demo-nav a.foto { --c:var(--foto); }
.demo-nav a.pipeline { --c:var(--pipe); } .demo-nav a.llm { --c:var(--llm); }
.demo-theme { font:inherit; font-size:.9rem; padding:5px 13px; border-radius:999px; border:1px solid var(--line); margin-left:10px;
              background:var(--card); color:var(--fg); cursor:pointer; }
.demo-theme:hover { border-color:var(--fg); }
.demo-dim { color:var(--muted); font-size:.9rem; }
/* every section is a slide with its own colour: the number, a band at the top and the frames inside */
.demo-section { margin:0; height:calc(100vh - var(--top)); box-sizing:border-box; padding:16px 20px 12px; display:flex;
                flex-direction:column; scroll-snap-align:start; scroll-snap-stop:always; --sec:var(--fg);
                border-top:3px solid color-mix(in srgb, var(--sec) 65%, transparent);
                background:linear-gradient(180deg, color-mix(in srgb, var(--sec) 14%, transparent), transparent 46%); }
#livro { --sec:var(--livro); } #foto { --sec:var(--foto); }
#leituras { --sec:var(--pipe); border-image:linear-gradient(90deg, var(--pipe), var(--llm)) 1;
            background:radial-gradient(55% 60% at 0% 0%, color-mix(in srgb, var(--pipe) 18%, transparent), transparent 70%),
                       radial-gradient(55% 60% at 100% 0%, color-mix(in srgb, var(--llm) 18%, transparent), transparent 70%); }
.demo-inner.pipeline { --sec:var(--pipe); } .demo-inner.llm { --sec:var(--llm); }
.demo h2, .demo-lead { flex:none; }
.demo-body { flex:1; min-height:0; overflow:auto; }
.demo h2 { font-size:1.4rem; border:0; margin:0 0 16px; display:flex; align-items:center; gap:12px; }
.demo-num { display:inline-flex; width:32px; height:32px; border-radius:50%; background:var(--sec); color:var(--bg);
            align-items:center; justify-content:center; font-size:1rem; font-weight:700; }
.demo-lead { color:var(--muted); margin:-6px 0 14px; font-size:1rem; }
.demo-fork { display:flex; align-items:center; gap:22px; overflow-x:auto; padding:8px 2px 12px; }
.demo-origin, .demo-branches, .demo-branch { flex:0 0 auto; }
.demo-branches { display:flex; flex-direction:column; gap:18px; }
.demo-branch { display:flex; align-items:center; gap:12px; }
.demo-step { margin:0; background:var(--card); border:3px solid var(--line); border-radius:12px; padding:12px; --c:var(--line); }
.demo-origin .demo-step { --c:var(--foto); } .demo-branch.pipeline .demo-step { --c:var(--pipe); } .demo-branch.llm .demo-step { --c:var(--llm); }
.demo-step { border-color:var(--c); background:linear-gradient(180deg, color-mix(in srgb, var(--c) 12%, var(--card)), var(--card) 70%); }
.demo-step figcaption b { color:var(--c); }
.demo-origin img { max-height:calc(100vh - 330px); }
.demo-branch img { max-height:calc(50vh - 205px); }
.demo-step img { width:auto; max-width:min(36vw, 520px); display:block; margin:0 auto; border-radius:4px; }
.demo-step figcaption { margin-top:10px; font-size:.95rem; line-height:1.4; text-align:center; }
.demo-step figcaption span { color:var(--muted); }
.demo-arrow { display:flex; flex-direction:column; align-items:center; color:var(--muted); min-width:170px; text-align:center; }
.demo-arrow span { font-size:2.6rem; line-height:1; }
.demo-arrow small { font-size:.85rem; margin-top:2px; }
.demo-branch.pipeline .demo-arrow { color:var(--pipe); } .demo-branch.llm .demo-arrow { color:var(--llm); }
.demo-inner .demo-body > .card { margin-top:0; padding:16px 20px; border-left:6px solid var(--sec);
                                 border-color:color-mix(in srgb, var(--sec) 45%, var(--line)); border-left-color:var(--sec);
                                 background:linear-gradient(180deg, color-mix(in srgb, var(--sec) 9%, var(--card)), var(--card) 40%); }
.demo-inner .demo-body > .card > h3:first-child { display:none; }
.demo-inner.pipeline .steps img, .demo-inner.pipeline .lines img { border:2px solid var(--pipe); border-radius:4px; }
.demo-inner h4 { font-size:.85rem; margin:24px 0 10px; color:var(--sec); text-transform:uppercase; letter-spacing:.08em; }
.demo-inner .text { font-size:1.05rem; line-height:1.5; }
.demo-inner summary { font-size:1rem; padding:6px 0; }
.demo-grid { display:grid; gap:22px; } .demo-grid.cols2 { grid-template-columns:1fr 1fr; }
/* reading cards, each shaded in its OCR's colour */
.demo-reading { --c:var(--muted); background:linear-gradient(180deg, color-mix(in srgb, var(--c) 18%, var(--card)), var(--card) 65%);
                border:1px solid color-mix(in srgb, var(--c) 45%, var(--line)); border-top:6px solid var(--c); border-radius:12px;
                padding:18px 24px 14px; min-width:0; }
.demo-reading.pipeline { --c:var(--pipe); } .demo-reading.llm { --c:var(--llm); } .demo-reading.real { --c:var(--real); }
.demo-reading header { display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; margin-bottom:14px; }
.demo-tag { font-size:.75rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; padding:3px 9px; border-radius:999px;
            background:var(--c, var(--muted)); color:var(--bg); }
.demo-reading .demo-tag.example { background:#c53030; color:#fff; }
.demo-cost { margin-left:auto; font-weight:700; font-size:1.05rem; color:var(--llm); font-variant-numeric:tabular-nums; }
.demo-title { color:var(--muted); font-size:.9rem; }
.demo-text { font-family:Georgia, "Times New Roman", serif; font-size:1.6rem; line-height:1.55; min-height:3em; }
.demo-reading footer { margin-top:16px; border-top:1px solid color-mix(in srgb, var(--c) 30%, var(--line)); padding-top:8px; }
/* the opening: the page of the book and the three covered cards */
.demo-book-grid { display:grid; grid-template-columns:minmax(240px, 25%) 1fr; grid-template-rows:minmax(0, 1fr); gap:28px; height:100%; }
.demo-book-page { margin:0; display:flex; flex-direction:column; min-height:0; --c:var(--livro); }
.demo-book-page img { flex:1; min-height:0; width:100%; max-width:100%; object-fit:contain; }  /* the whole page in view */
.demo-book-panels { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:18px; align-content:start;
                    min-height:0; overflow:auto; }
.demo-book-tools { grid-column:1 / -1; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.demo-diff-key { font-size:.9rem; } .demo-diff-key[hidden] { display:none; }
.demo-diff-key i { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin:0 4px 0 10px; }
.demo-diff-key i.k-pipeline { background:var(--pipe-soft); } .demo-diff-key i.k-llm { background:var(--llm-soft); }
.demo-diff-key i.k-wrong { background:rgba(229,62,62,.35); margin-left:0; }
.demo-book .demo-reading footer { margin-top:12px; padding-top:8px; font-size:.85rem; }
.demo-btn { font:inherit; font-size:1.02rem; font-weight:600; padding:11px 22px; border-radius:999px; cursor:pointer;
            border:2px solid transparent; background:var(--fg); color:var(--bg); }
.demo-btn.pipeline { background:var(--pipe); } .demo-btn.real { background:var(--real); } .demo-btn.llm { background:var(--llm); }
.demo-btn.ghost { background:color-mix(in srgb, var(--card) 75%, transparent); color:var(--fg); font-weight:500;
                  border-color:color-mix(in srgb, var(--sec) 55%, var(--line)); }
.demo-btn.ghost[aria-pressed="true"] { border-color:var(--sec); background:color-mix(in srgb, var(--sec) 16%, var(--card)); }
.demo-btn:hover:not(:disabled) { filter:brightness(1.08); } .demo-btn:disabled { opacity:.4; cursor:default; }
.demo-cover { min-height:240px; display:flex; align-items:center; justify-content:center; border-radius:8px;
              background:repeating-linear-gradient(135deg, color-mix(in srgb, var(--c) 28%, transparent) 0 5px, transparent 5px 17px); }
.demo-cover[hidden], .demo-book .demo-text[hidden] { display:none; }
.demo-book .demo-text { font-size:1.2rem; line-height:1.5; }
.demo-book mark { background:transparent; color:inherit; border-radius:3px; padding:0 2px; }
.demo-book .pipeline mark, .demo-book .llm mark { background:rgba(229,62,62,.22); text-decoration:underline wavy #e53e3e; }
.demo-book .real mark.m-pipeline { background:var(--pipe-soft); } .demo-book .real mark.m-llm { background:var(--llm-soft); }
.demo-book .real mark.m-pipeline.m-llm { background:linear-gradient(to bottom, var(--llm-soft) 50%, var(--pipe-soft) 50%); }
@media (max-width:1000px) { .demo-grid.cols2 { grid-template-columns:1fr; } .demo-step img { max-width:80vw; }
                            html { scroll-snap-type:none; } .demo-section { height:auto; display:block; }
                            .demo-body { overflow:visible; } .demo-origin img, .demo-branch img { max-height:none; }
                            .demo-book-grid { grid-template-columns:1fr; height:auto; grid-template-rows:none; }
                            .demo-book-page img { height:auto; max-height:80vh; }
                            .demo-fork { flex-direction:column; align-items:flex-start; } .demo-arrow { min-width:0; }
                            .demo-top { position:static; } .demo-text { font-size:1.3rem; } }
@media print { .demo { background:#fff; } .demo-top { position:static; } .viewer { display:none; } html { scroll-snap-type:none; }
               .demo-section { height:auto; display:block; } .demo-body { overflow:visible; } }
.zoomable { cursor:zoom-in; }
.viewer { position:fixed; inset:0; z-index:50; display:flex; flex-direction:column; background:rgba(12,12,12,.97); color:#eee; }
.viewer[hidden] { display:none; }
.v-bar { display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; padding:10px 16px;
         background:#1b1b1b; border-bottom:1px solid #333; }
.v-captions { display:flex; gap:24px; font-size:.95rem; min-width:0; } .v-captions span:empty { display:none; }
.v-captions b { color:#9ec5ff; font-weight:600; margin-right:6px; }
.v-actions { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.v-actions button { background:#2b2b2b; color:#eee; border:1px solid #444; border-radius:6px; padding:6px 11px; font:inherit;
                    font-size:.9rem; cursor:pointer; }
.v-actions button:hover { background:#383838; } .v-actions button[aria-pressed="true"] { background:#3b5bdb; border-color:#3b5bdb; }
.v-actions button:disabled { opacity:.35; cursor:default; }
#v-zoom { min-width:52px; text-align:center; font-variant-numeric:tabular-nums; font-size:.9rem; }
.v-stage { flex:1; display:flex; gap:4px; min-height:0; }
.v-pane { flex:1; position:relative; overflow:hidden; cursor:grab; background:#101010; }
.v-pane.dragging { cursor:grabbing; }
.v-pane img { position:absolute; left:0; top:0; transform-origin:0 0; max-width:none; user-select:none; -webkit-user-drag:none; }
.v-pane .v-label { position:absolute; left:10px; top:8px; background:rgba(0,0,0,.65); padding:4px 9px; border-radius:6px;
                   font-size:.85rem; pointer-events:none; max-width:90%; }
.v-help { padding:6px 16px; font-size:.8rem; color:#aaa; background:#1b1b1b; border-top:1px solid #333; }
"""


LAYOUT_SCRIPT = '''
(() => {  // every section is one screen high below the fixed bar, whatever the bar's height turns out to be
  const root = document.documentElement, bar = document.querySelector('.demo-top'); if (!bar) return;
  const set = () => root.style.setProperty('--top', bar.offsetHeight + 'px');
  set(); addEventListener('resize', set); if (document.fonts && document.fonts.ready) document.fonts.ready.then(set);
  // the theme follows the system until the button is used; the choice stays in this browser
  const button = document.querySelector('.demo-theme');
  const system = () => matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  const current = () => root.dataset.theme || system();
  let saved = null; try { saved = localStorage.getItem('demo-theme'); } catch (e) {}
  if (!root.dataset.theme && (saved === 'dark' || saved === 'light')) root.dataset.theme = saved;
  const label = () => { if (button) button.textContent = current() === 'dark' ? 'tema claro' : 'tema escuro'; };
  label();
  if (button) button.addEventListener('click', () => {
    root.dataset.theme = current() === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('demo-theme', root.dataset.theme); } catch (e) {}
    label(); set();
  });
})();
'''


def standalone_html(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>{style()}'
            f'<style>{DEMO_CSS}</style></head><body class="ocr-body demo">{body}{VIEWER}'
            f'<script>{LAYOUT_SCRIPT}</script></body></html>')


VIEWER = r'''
<div id="viewer" class="viewer" hidden>
  <div class="v-bar">
    <div class="v-captions"><span id="v-capA"></span><span id="v-capB"></span></div>
    <div class="v-actions">
      <button data-act="prev" title="passo anterior (←)">‹ anterior</button>
      <button data-act="next" title="passo seguinte (→)">seguinte ›</button>
      <button data-act="compare" title="comparar com o passo anterior, lado a lado (C)">comparar</button>
      <button data-act="flip" title="alternar entre este passo e o anterior (espaço)">alternar</button>
      <button data-act="zoomout" title="afastar (−)">−</button><span id="v-zoom">100%</span>
      <button data-act="zoomin" title="aproximar (+)">+</button>
      <button data-act="fit" title="ajustar à tela (F)">ajustar</button>
      <button data-act="one" title="tamanho real, 1 pixel = 1 pixel (1)">1:1</button>
      <button data-act="close" title="fechar (Esc)">×</button>
    </div>
  </div>
  <div class="v-stage" id="v-stage">
    <div class="v-pane" id="v-paneA"><img id="v-imgA" alt=""><div class="v-label" id="v-labelA"></div></div>
    <div class="v-pane" id="v-paneB" hidden><img id="v-imgB" alt=""><div class="v-label" id="v-labelB"></div></div>
  </div>
  <div class="v-help">clique numa imagem para ampliar · roda do mouse: zoom · arrastar: mover · ← →: passo anterior / seguinte ·
    C: comparar lado a lado com o passo anterior (mesmo zoom e posição) · espaço: alternar com o anterior · F: ajustar · 1: tamanho real · Esc: fechar</div>
</div>
<script>
(function () {
  const viewer = document.getElementById('viewer');
  const stage = document.getElementById('v-stage');
  const panes = { A: document.getElementById('v-paneA'), B: document.getElementById('v-paneB') };
  const imgs = { A: document.getElementById('v-imgA'), B: document.getElementById('v-imgB') };
  const labels = { A: document.getElementById('v-labelA'), B: document.getElementById('v-labelB') };
  const caps = { A: document.getElementById('v-capA'), B: document.getElementById('v-capB') };
  const zoomOut = document.getElementById('v-zoom');
  const buttons = {};
  viewer.querySelectorAll('button[data-act]').forEach(b => { buttons[b.dataset.act] = b; });

  function captionOf(img) {
    const fig = img.closest('figure');
    if (fig) { const c = fig.querySelector('figcaption'); if (c) return c.textContent.replace(/\s+/g, ' ').trim(); }
    const td = img.closest('td');
    if (td && td.nextElementSibling) return td.nextElementSibling.textContent.replace(/\s+/g, ' ').trim();
    return img.alt || '';
  }
  // groups of consecutive images: the page of the book, the fork, every image step of a pipeline section, each
  // line table
  const groups = [];
  const containers = [...document.querySelectorAll('.demo-book, .demo-fork, .demo-inner, table.lines')];
  containers.forEach(c => {
    let list;
    if (c.classList.contains('demo-inner')) list = [...c.querySelectorAll('.steps img')];
    else list = [...c.querySelectorAll('img')];
    list = list.filter(img => !img.closest('.viewer'));
    if (!list.length) return;
    const group = list.map(img => ({ src: img.getAttribute('src'), caption: captionOf(img) }));
    groups.push(group);
    list.forEach((img, i) => {
      if (img.dataset.viewerBound) return;
      img.dataset.viewerBound = '1'; img.classList.add('zoomable');
      img.addEventListener('click', () => open(group, i));
    });
  });

  let group = null, idx = 0, compare = false, flipped = false;
  const view = { zoom: 1, cx: 0.5, cy: 0.5 };  // zoom relative to "fit"; centre as a fraction of the image
  const natural = { A: null, B: null };

  function open(g, i) { group = g; idx = i; compare = false; flipped = false; view.zoom = 1; view.cx = view.cy = 0.5;
    viewer.hidden = false; document.body.style.overflow = 'hidden'; load(); }
  function close() { viewer.hidden = true; document.body.style.overflow = ''; }
  function current() { return flipped && idx > 0 ? idx - 1 : idx; }
  function load() {
    const a = group[current()], b = idx > 0 ? group[idx - 1] : null;
    setImage('A', a, compare ? 'A · este passo: ' : (flipped ? 'passo anterior: ' : ''));
    panes.B.hidden = !(compare && b);
    if (compare && b) setImage('B', b, 'B · passo anterior: ');
    buttons.prev.disabled = idx === 0; buttons.next.disabled = idx === group.length - 1;
    buttons.compare.disabled = idx === 0; buttons.flip.disabled = idx === 0;
    buttons.compare.setAttribute('aria-pressed', compare); buttons.flip.setAttribute('aria-pressed', flipped);
    caps.A.innerHTML = '<b>' + (compare ? 'A' : (current() + 1) + '/' + group.length) + '</b>' + esc(a.caption);
    caps.B.innerHTML = compare && b ? '<b>B</b>' + esc(b.caption) : '';
  }
  function esc(s) { return s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
  function setImage(k, item, prefix) {
    const img = imgs[k]; natural[k] = null; labels[k].textContent = prefix + item.caption;
    img.onload = () => { natural[k] = { w: img.naturalWidth, h: img.naturalHeight }; render(); };
    img.src = item.src;
    if (img.complete && img.naturalWidth) img.onload();
  }
  function scaleOf(k) {
    const n = natural[k]; if (!n) return 1;
    const r = panes[k].getBoundingClientRect();
    return Math.min(r.width / n.w, r.height / n.h) * view.zoom;
  }
  function render() {
    for (const k of ['A', 'B']) {
      const n = natural[k]; if (!n || panes[k].hidden) continue;
      const r = panes[k].getBoundingClientRect(), s = scaleOf(k);
      const left = r.width / 2 - view.cx * n.w * s, top = r.height / 2 - view.cy * n.h * s;
      imgs[k].style.width = n.w + 'px'; imgs[k].style.height = n.h + 'px';
      imgs[k].style.transform = 'translate(' + left + 'px,' + top + 'px) scale(' + s + ')';
    }
    if (natural.A) zoomOut.textContent = Math.round(scaleOf('A') * 100) + '%';
  }
  function zoomBy(factor, pane, clientX, clientY) {
    const k = pane || 'A', n = natural[k]; if (!n) return;
    const r = panes[k].getBoundingClientRect();
    const px = clientX == null ? r.width / 2 : clientX - r.left, py = clientY == null ? r.height / 2 : clientY - r.top;
    const s0 = scaleOf(k);
    // image point under the cursor stays put
    const ix = view.cx + (px - r.width / 2) / (n.w * s0), iy = view.cy + (py - r.height / 2) / (n.h * s0);
    view.zoom = Math.min(Math.max(view.zoom * factor, 0.2), 40);
    const s1 = scaleOf(k);
    view.cx = ix - (px - r.width / 2) / (n.w * s1); view.cy = iy - (py - r.height / 2) / (n.h * s1);
    render();
  }
  function fit() { view.zoom = 1; view.cx = view.cy = 0.5; render(); }
  function one() { const n = natural.A; if (!n) return; const r = panes.A.getBoundingClientRect();
    view.zoom = 1 / Math.min(r.width / n.w, r.height / n.h); render(); }
  function step(d) { const j = idx + d; if (j < 0 || j >= group.length) return; idx = j; if (idx === 0) { compare = false; flipped = false; } load(); }
  function toggleCompare() { if (idx === 0) return; compare = !compare; flipped = false; load(); render(); }
  function toggleFlip() { if (idx === 0 || compare) return; flipped = !flipped; load(); }

  viewer.addEventListener('click', e => {
    const b = e.target.closest('button[data-act]'); if (!b) return;
    ({ prev: () => step(-1), next: () => step(1), compare: toggleCompare, flip: toggleFlip, zoomin: () => zoomBy(1.25),
       zoomout: () => zoomBy(0.8), fit, one, close })[b.dataset.act]();
  });
  for (const k of ['A', 'B']) {
    const pane = panes[k];
    pane.addEventListener('wheel', e => { e.preventDefault(); zoomBy(e.deltaY < 0 ? 1.15 : 1 / 1.15, k, e.clientX, e.clientY); }, { passive: false });
    let drag = null;
    pane.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY }; pane.classList.add('dragging'); pane.setPointerCapture(e.pointerId); });
    pane.addEventListener('pointermove', e => {
      if (!drag) return; const n = natural[k]; if (!n) return; const s = scaleOf(k);
      view.cx -= (e.clientX - drag.x) / (n.w * s); view.cy -= (e.clientY - drag.y) / (n.h * s);
      drag = { x: e.clientX, y: e.clientY }; render();
    });
    const end = () => { drag = null; pane.classList.remove('dragging'); };
    pane.addEventListener('pointerup', end); pane.addEventListener('pointercancel', end);
    pane.addEventListener('dblclick', e => zoomBy(2, k, e.clientX, e.clientY));
  }
  window.addEventListener('keydown', e => {
    if (viewer.hidden) return;
    const k = e.key;
    if (k === 'Escape') close(); else if (k === 'ArrowLeft') step(-1); else if (k === 'ArrowRight') step(1);
    else if (k === 'c' || k === 'C') toggleCompare(); else if (k === ' ') { e.preventDefault(); toggleFlip(); }
    else if (k === 'f' || k === 'F') fit(); else if (k === '1') one();
    else if (k === '+' || k === '=') zoomBy(1.25); else if (k === '-') zoomBy(0.8); else return;
    e.preventDefault();
  });
  window.addEventListener('resize', render);
})();
</script>
'''


BOOK_SCRIPT = r'''
(() => {
  const section = document.getElementById('livro'); if (!section) return;
  const panels = {}, texts = {}, originals = {}, words = {}, parts = {}, missed = {};
  [...section.querySelectorAll('[data-panel]')].forEach(p => { const k = p.dataset.panel; panels[k] = p;
    texts[k] = p.querySelector('.demo-text'); originals[k] = texts[k].textContent; });
  const kinds = Object.keys(panels), ocrs = kinds.filter(k => k !== 'real');
  const names = { pipeline: 'o pipeline', llm: 'o LLM' };
  const diffButton = section.querySelector('[data-reveal="diff"]'), diffKey = section.querySelector('.demo-diff-key');
  let diff = false, computed = false;
  const esc = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  // the same policy as the project's normalised WER: NFC, lower case, no punctuation
  const key = w => w.normalize('NFC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');

  function compute() {  // for each OCR: its words outside the longest common subsequence with the reference, and
    computed = true;    // the reference words it missed
    kinds.forEach(k => { parts[k] = originals[k].split(/(\s+)/);
      words[k] = parts[k].map((w, i) => ({ i, key: i % 2 === 0 ? key(w) : '' })).filter(x => x.key); });
    const b = words.real, m = b.length;
    ocrs.forEach(k => {
      const a = words[k], n = a.length;
      const L = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
      for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
        L[i][j] = a[i].key === b[j].key ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
      const same = new Set(), hit = new Set();
      for (let i = 0, j = 0; i < n && j < m;) {
        if (a[i].key === b[j].key) { same.add(a[i].i); hit.add(b[j].i); i++; j++; }
        else if (L[i + 1][j] >= L[i][j + 1]) i++; else j++;
      }
      missed[k] = new Set(b.filter(x => !hit.has(x.i)).map(x => x.i));
      texts[k].dataset.marked = parts[k].map((w, i) =>
        i % 2 === 1 || !key(w) || same.has(i) ? esc(w) : '<mark>' + esc(w) + '</mark>').join('');
    });
  }
  function referenceMarked(shownOcrs) {  // words missed by each OCR that is open, in that OCR's colour
    return parts.real.map((w, i) => {
      if (i % 2 === 1 || !key(w)) return esc(w);
      const by = shownOcrs.filter(k => missed[k].has(i));
      return by.length ? '<mark class="' + by.map(k => 'm-' + k).join(' ') + '">' + esc(w) + '</mark>' : esc(w);
    }).join('');
  }
  function shown(k) { return !texts[k].hidden; }
  function reveal(k, on) {
    texts[k].hidden = !on; panels[k].querySelector('.demo-cover').hidden = on;
    panels[k].classList.toggle('demo-covered', !on); update();
  }
  function update() {
    const open = ocrs.filter(shown), ready = shown('real') && open.length > 0;
    if (!ready) diff = false;
    diffButton.disabled = !ready; diffButton.setAttribute('aria-pressed', diff);
    if (diff && !computed) compute();
    kinds.forEach(k => { texts[k].innerHTML = !diff ? esc(originals[k]) : k === 'real' ? referenceMarked(open) : texts[k].dataset.marked; });
    diffKey.hidden = !diff;
    if (diff) diffKey.innerHTML = '<i class="k-wrong"></i>palavra que o OCR errou ou inventou · na referência:'
      + open.map(k => '<i class="k-' + k + '"></i>' + names[k] + ' não leu').join(';');
  }
  section.addEventListener('click', e => {
    const b = e.target.closest('[data-reveal]'); if (!b || b.disabled) return;
    const act = b.dataset.reveal;
    if (act === 'diff') { diff = !diff; update(); }
    else if (act === 'hide') { ocrs.concat(['real']).forEach(k => reveal(k, false)); }
    else reveal(act, !shown(act));
  });
})();
'''


def build_demo(out: Path = OUT, device: str | None = None, allow_spend: bool = False, seed: int = 42,
               open_cards: bool = False, use_llm: bool = True, book_page: int | None = BOOK_PAGE,
               llm_example: str | None = None) -> Demo:
    """The two OCRs, always the same: the final pipeline and the best LLM run (or the pipeline alone), and the
    page of the book that opens every card (none if its files are missing). Fails early if the LLM is allowed to
    spend but there is no key. llm_example (simulation only): the LLM card shows this text instead of the
    simulation mark, to rehearse the page without sending anything."""
    if device is None:
        import torch

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    out = out.resolve()
    engines: list[Any] = [PipelineOCR(out, device, seed)]
    if use_llm:
        llm = LLMOCR(out, allow_spend=allow_spend, example=llm_example)
        llm.check()
        engines.append(llm)
    book = None
    if book_page:
        try:
            book = load_book_page(book_page)
        except OSError as error:  # outside the repository, or the delivery folder is missing: the card starts at the photo
            logger.warning('o cartão abre sem a página do livro: %s', error)
    return Demo(out, engines, open_cards=open_cards, book=book)


# --------------------------------------------------------------------------------------------------------------
# Photos folder, watched folder and command line
# --------------------------------------------------------------------------------------------------------------

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(('.', '~'))


def latest_photo(folder: Path = PHOTOS) -> Path:
    """The most recently modified image in the folder (the one just copied from the phone)."""
    photos = [p for p in Path(folder).iterdir() if is_image(p)] if Path(folder).is_dir() else []
    if not photos:
        raise FileNotFoundError(f'nenhuma foto em {Path(folder).resolve()}')
    return max(photos, key=lambda p: p.stat().st_mtime)


def copied(path: Path, settle: float = 1.0, timeout: float = 60.0) -> bool:
    """True once the file's size has stopped changing: a photo copied from the phone may take a while to land."""
    deadline = time.monotonic() + timeout
    size = -1
    while time.monotonic() < deadline:
        try:
            current = path.stat().st_size
        except OSError:
            current = -1
        if current > 0 and current == size:
            return True
        size = current
        time.sleep(settle)
    return False


def report(result: DemoResult) -> None:
    for reading in result.readings:
        print(f'\n=== {reading.title} · {reading.details} ===\n{reading.text or reading.error}', flush=True)
    print(f'\n-> cartão: {result.card_file}\n', flush=True)


def watch(demo: Demo, folder: Path = PHOTOS, interval: float = 1.0) -> None:
    """Every image that appears in the folder is processed once."""
    folder.mkdir(parents=True, exist_ok=True)
    seen = {p for p in folder.iterdir() if is_image(p)}
    print(f'\nvigiando {folder.resolve()} ({len(seen)} imagens já existentes ignoradas).\n'
          'Copie a foto do celular para essa pasta. Ctrl+C para sair.\n', flush=True)
    while True:
        for path in sorted(p for p in folder.iterdir() if is_image(p) and p not in seen):
            seen.add(path)
            if not copied(path):
                print(f'{path.name}: a cópia não terminou; ignorada', flush=True)
                continue
            print(f'{path.name}: processando...', flush=True)
            try:
                result = demo.run(path)
            except Exception as error:  # noqa: BLE001 - keep watching
                logger.exception('%s falhou', path.name)
                print(f'{path.name}: falhou ({error})', flush=True)
                continue
            report(result)
        time.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--foto', type=Path, nargs='+', help='uma ou mais fotos')
    source.add_argument('--vigiar', type=Path, nargs='?', const=PHOTOS, metavar='PASTA',
                        help=f'pasta vigiada: processa toda imagem nova que cair nela (padrão: {PHOTOS})')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--abrir', action='store_true', help='abrir cada cartão no navegador')
    parser.add_argument('--device', help='cuda:0 ou cpu (padrão: cuda:0 se houver GPU)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--permitir-gasto', action='store_true',
                        help=f'enviar as fotos ao LLM ({LLM_MODEL}, pago); sem isso, só simula a requisição')
    parser.add_argument('--sem-llm', action='store_true', help='só o pipeline, sem a coluna do LLM')
    parser.add_argument('--exemplo-llm', nargs='?', const=LLM_EXAMPLE_TEXT, metavar='TEXTO',
                        help='ensaio sem gasto: o cartão do LLM mostra um texto de exemplo no lugar da leitura '
                             '(o padrão é a leitura real do Gemini na folha de teste); ignorado com --permitir-gasto')
    parser.add_argument('--sem-aquecer', action='store_true', help='não carregar os modelos antes da primeira foto')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for noisy in ('kraken', 'lightning', 'lightning.pytorch', 'shapely', 'urllib3'):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    try:
        demo = build_demo(args.out, args.device, args.permitir_gasto, args.seed, args.abrir, use_llm=not args.sem_llm,
                          llm_example=args.exemplo_llm)
    except Exception as error:  # noqa: BLE001 - GastoNaoAutorizado, reported without a traceback
        parser.error(str(error))
    if not args.sem_aquecer:
        demo.warm_up()
    mode = ('envio autorizado' if args.permitir_gasto else 'exemplo, nada enviado' if args.exemplo_llm else 'simulação')
    llm = '' if args.sem_llm else f' + {LLM_MODEL} ({mode})'
    print(f'\nOCRs: {PIPELINE_CONFIG}{llm}\nsaída: {demo.out}', flush=True)
    if args.foto:
        for path in args.foto:
            report(demo.run(path))
    else:
        watch(demo, args.vigiar)


if __name__ == '__main__':
    sys.exit(main())
