"""ocr_pipeline.demo: preparing a phone photo and the LLM engine. No pipeline model is loaded and no HTTP request
ever leaves the machine: the LLM tests run in simulation or against a fake transport, and every one of them
patches requests.post so that a real call would fail the test instead of costing money."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import ocr_historico.motor_llm as motor_llm
from ocr_pipeline.demo import (BOOK_PAGE, LLM_EFFORT, LLM_EXAMPLE_COST, LLM_EXAMPLE_TEXT, LLM_MODEL, BookPage, Demo,
                               LLMOCR, build_demo, latest_photo, load_book_page, prepare_photo, without_image)


def test_the_notebook_entry_point_exists():
    assert callable(build_demo)  # the notebook and the CLI build the demo through it


def photo(width: int = 1500, height: int = 2000, lines: int = 6) -> Image.Image:
    """A light sheet with dark strokes, like handwriting, filling the frame."""
    paper = Image.new('RGB', (width, height), (250, 248, 240))
    draw = ImageDraw.Draw(paper)
    margin = width // 10
    for i in range(lines):
        y = margin + 40 + (height - 2 * margin - 40) * i // max(1, lines - 1)
        for x in range(margin, width - margin, 14):
            draw.line([(x, y + (x % 3) * 4), (x + 9, y - 12 + (x % 5) * 3)], fill=(30, 30, 60), width=3)
    return paper


# ---- preparing the photo: only the reduction to the book's scale ---------------------------------------------

def test_prepare_reduces_to_the_book_scale_and_records_the_steps(tmp_path: Path):
    path = tmp_path / 'foto.jpg'
    photo().save(path)
    prepared = prepare_photo(path)
    assert prepared.sheet.size == (660, 880)  # same proportions, longer side 880
    labels = [label for label, _ in prepared.steps]
    assert labels == ['foto', '+ redução à escala do pipeline (660×880 px): entrada do pipeline']
    assert prepared.steps[0][1].size == (1500, 2000) and prepared.steps[1][1].size == (660, 880)


def test_the_llm_gets_the_photo_at_its_own_resolution(tmp_path: Path):
    prepared = prepare_photo(photo())
    assert prepared.full.size == (1500, 2000)  # the LLM engine reduces it to 1600 on its own
    sent = LLMOCR(tmp_path).image_for(prepared)
    assert sent.size == (1200, 1600)


def test_exif_orientation_is_applied_when_opening(tmp_path: Path):
    upright = photo()
    sideways = upright.rotate(90, expand=True)  # stored sideways; orientation 6 = rotate 90° clockwise to view
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / 'exif.jpg'
    sideways.save(path, exif=exif.tobytes())
    prepared = prepare_photo(path, long_side=0)
    assert prepared.original.size == upright.size
    assert np.abs(np.asarray(prepared.original).astype(int) - np.asarray(upright).astype(int)).mean() < 3


def test_a_small_image_passes_untouched():
    scan = photo(640, 870)  # already at the book's scale, like the project pages
    prepared = prepare_photo(scan)
    assert prepared.sheet.size == scan.size and [label for label, _ in prepared.steps] == ['foto']


def test_latest_photo_is_the_newest_image(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        latest_photo(tmp_path)
    for name, age in (('antiga.jpg', 100), ('nova.png', 0), ('nota.txt', 0)):
        (tmp_path / name).write_bytes(b'x')
        os.utime(tmp_path / name, (time.time() - age, time.time() - age))
    assert latest_photo(tmp_path).name == 'nova.png'


# ---- the LLM engine, without ever sending anything ---------------------------------------------------------

@pytest.fixture(autouse=True)
def no_real_requests(monkeypatch):
    """Any attempt to reach the network fails the test (and is never a paid call)."""
    def forbidden(*args, **kwargs):
        raise AssertionError('uma requisição HTTP real foi tentada')

    monkeypatch.setattr(motor_llm.requests, 'post', forbidden)
    monkeypatch.setattr(motor_llm, 'obter_chave', lambda: None)  # never pick up a real key from .env/environment


def test_llm_simulation_builds_the_request_and_sends_nothing(tmp_path: Path):
    engine = LLMOCR(tmp_path)  # without allow_spend the engine only simulates
    engine.check()  # simulation never needs a key
    assert engine.simulate
    result = Demo(tmp_path, [engine]).run(photo())
    reading = result.readings[0]
    assert reading.engine == 'llm' and reading.text == motor_llm.TEXTO_SIMULADO
    assert reading.error is None and 'simulação' in reading.details
    record = json.loads((tmp_path / 'llm' / f'{result.job_id}.json').read_text(encoding='utf-8'))
    assert record['simulado'] and record['modelo'] == LLM_MODEL == 'google/gemini-3.8-flash'
    request = record['requisicao']
    assert request['model'] == LLM_MODEL and request['reasoning'] == {'effort': LLM_EFFORT} == {'effort': 'low'}
    content = request['messages'][0]['content']
    assert content[0] == {'type': 'text', 'text': motor_llm.PROMPT_TRANSCRICAO}
    assert content[1]['image_url']['url'].startswith('<data URL PNG')  # the image itself is not in the record
    sent = Image.open(tmp_path / 'entrada' / f'{result.job_id}.llm.png')
    assert (record['largura'], record['altura']) == sent.size == (1200, 1600)
    assert (tmp_path / 'textos' / f'{result.job_id}.llm.txt').is_file()


def test_the_card_shows_the_steps_before_the_readings(tmp_path: Path):
    result = Demo(tmp_path, [LLMOCR(tmp_path)]).run(photo())
    card = result.card
    steps, readings = card.index('A foto original vira uma entrada'), card.index('A leitura')  # one OCR here
    assert 0 < steps < readings
    for label in ('foto original', '1500×2000 px', 'redução para o LLM', 'imagem enviada ao LLM', '1200×1600 px'):
        assert steps < card.index(label) < readings
    assert card.count('class="demo-branch ') == 1 and 'demo-branch llm' in card  # one branch, in the LLM colour
    assert 'entrada do pipeline' not in card and 'demo-inner llm' in card
    assert 'recorte' not in card and 'scanner' not in card  # nothing but the reduction happens to the photo
    page = result.card_file.read_text(encoding='utf-8')
    assert page.startswith('<!doctype html>') and card in page and 'class="ocr-body demo"' in page
    assert 'id="viewer"' in page and 'data-act="compare"' in page  # the image viewer ships with every card
    # slides: one body per section, snapping from one to the next, the bar's height measured by the page
    assert card.count('<section ') == card.count('class="demo-body"') == 3  # photo, reading, the LLM's section
    assert 'scroll-snap-type:y mandatory' in page and "setProperty('--top'" in page
    assert 'class="demo-theme"' in page and ':root[data-theme="dark"] .demo' in page  # the theme button and its CSS


def test_the_example_mode_shows_a_stand_in_text_without_sending(tmp_path: Path):
    engine = LLMOCR(tmp_path, example='texto de exemplo, nada enviado')
    assert engine.simulate and engine.example  # simulation, with a stand-in text instead of the simulation mark
    result = Demo(tmp_path, [engine]).run(photo())
    reading = result.readings[0]
    assert reading.text == 'texto de exemplo, nada enviado' and reading.example and 'exemplo' in reading.details
    assert reading.cost == LLM_EXAMPLE_COST and 'US$ 0.0033' in reading.details  # the cost of the real reading
    record = json.loads((tmp_path / 'llm' / f'{result.job_id}.json').read_text(encoding='utf-8'))
    assert record['simulado'] and record['exemplo'] and record['texto'] == reading.text and 'requisicao' in record
    assert record['custo_usd'] == LLM_EXAMPLE_COST
    assert 'demo-tag example' in result.card and 'texto de exemplo, nada enviado' in result.card
    assert 'exemplo: texto ilustrativo, nada foi enviado' in result.card  # the LLM's section says so too
    assert 'class="demo-cost"' in result.card and 'custo de referência' in result.card
    plain = Demo(tmp_path, [LLMOCR(tmp_path)]).run(photo())  # plain simulation: no text, no cost
    assert plain.readings[0].cost is None and 'demo-cost' not in plain.card
    assert LLMOCR(tmp_path, allow_spend=True, example='x').example is None  # real sending ignores the example
    assert 'Gemini' not in LLM_EXAMPLE_TEXT and len(LLM_EXAMPLE_TEXT.split()) > 10  # the default stand-in is a text


# ---- the opening: a page of the book to read, with the two texts to reveal --------------------------------------

def test_the_book_page_of_the_opening_comes_from_the_delivery():
    book = load_book_page()  # the data folder, the delivery and the evaluation's LLM record, as checked in
    assert book.number == BOOK_PAGE == 10 and book.image.width > 600
    assert 'Artigo' in book.pipeline_text and 'Artigo' in book.reference and book.pipeline_text != book.reference
    assert 'Artigo' in book.llm_text and 0 < book.llm_cost < 0.01  # Gemini 3.8 Flash low on the preprocessed page
    assert book.llm_label == 'google/gemini-3.8-flash, esforço low'
    assert load_book_page(llm=None).llm_text is None  # without the record the opening has two cards


def test_the_card_opens_with_the_book_page_when_given(tmp_path: Path):
    book = BookPage(10, photo(600, 900), 'o que o pipeline\nleu', 'o texto real da página',
                    llm_text='o que o llm leu', llm_cost=0.00197, llm_label='google/gemini-3.8-flash, esforço low')
    card = Demo(tmp_path, [LLMOCR(tmp_path)], book=book).run(photo()).card
    opening, fork = card.index('Você consegue ler esta página?'), card.index('A foto original vira')
    assert 0 < opening < fork and opening < card.index('página 10 do livro') < fork
    assert 'demo-num">1</span>Você consegue' in card and 'demo-num">2</span>A foto' in card  # the rest shifts down
    for piece in ('data-reveal="pipeline"', 'data-reveal="real"', 'data-reveal="diff"', 'o que o pipeline leu',
                  'o texto real da página', 'href="#livro"', 'class="demo-reading demo-covered real"',
                  'data-reveal="llm"', 'o que o llm leu', 'US$ 0.0020 por esta página', 'custo por página: nenhum'):
        assert piece in card
    assert card.index('data-panel="pipeline"') < card.index('data-panel="llm"') < card.index('data-panel="real"')
    assert 'o que o pipeline\nleu' not in card  # running text: the line breaks of each text do not reach the cards
    two = Demo(tmp_path, [LLMOCR(tmp_path)], book=BookPage(10, photo(600, 900), 'p', 'r')).run(photo()).card
    assert 'data-reveal="llm"' not in two and 'data-reveal="real"' in two  # no LLM record: two cards
    without = Demo(tmp_path, [LLMOCR(tmp_path)]).run(photo()).card  # no page given: the card starts at the photo
    assert 'Você consegue ler' not in without and 'demo-num">1</span>A foto' in without


def test_llm_with_permission_but_no_key_fails_before_the_first_photo(tmp_path: Path):
    with pytest.raises(motor_llm.GastoNaoAutorizado, match='Chave'):
        LLMOCR(tmp_path, allow_spend=True).check()


def test_llm_reading_through_a_fake_transport(tmp_path: Path, monkeypatch):
    """The real code path with permission and a key, but requests.post is a stub: nothing leaves the machine."""
    calls = []

    class FakeResponse:
        ok, status_code, text = True, 200, ''

        @staticmethod
        def json():
            return {'choices': [{'message': {'content': 'Lei nº 12 de 3 de Março de 1893 A Camara Municipal decreta'}}],
                    'usage': {'cost': 0.0123}}

    def fake_post(url, json=None, timeout=None, headers=None):
        calls.append((url, json, headers))
        return FakeResponse()

    monkeypatch.setattr(motor_llm.requests, 'post', fake_post)
    monkeypatch.setattr(motor_llm, 'obter_chave', lambda: 'chave-falsa')
    engine = LLMOCR(tmp_path, allow_spend=True, long_side=0)
    engine.check()
    result = Demo(tmp_path, [engine]).run(photo())
    reading = result.readings[0]
    url, body, headers = calls[0]
    assert url == motor_llm.URL_OPENROUTER and headers['Authorization'] == 'Bearer chave-falsa'
    assert body['model'] == 'google/gemini-3.8-flash' and body['reasoning'] == {'effort': 'low'}
    assert body['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
    assert reading.text.startswith('Lei nº 12') and 'US$ 0.0123' in reading.details and reading.error is None
    assert reading.cost == 0.0123 and '<span class="demo-cost" title="custo desta leitura">US$ 0.0123</span>' in result.card
    card = result.card_file.read_text(encoding='utf-8')
    assert 'gemini-3.8-flash' in card and 'resposta do modelo' in card
    assert json.loads((tmp_path / 'llm' / f'{result.job_id}.json').read_text(encoding='utf-8'))['custo_usd'] == 0.0123


def test_without_image_keeps_everything_but_the_data_url():
    body = {'model': 'm', 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'p'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}]}
    stripped = without_image(body)
    assert stripped['messages'][0]['content'][0] == {'type': 'text', 'text': 'p'}
    assert stripped['messages'][0]['content'][1]['image_url']['url'] == '<data URL PNG: 26 caracteres>'
    assert body['messages'][0]['content'][1]['image_url']['url'].startswith('data:')  # the original is untouched
