import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr_historico.combinacoes import PADRAO_KRAKEN, PADRAO_LLM, interpretar
from ocr_historico.dados import listar_paginas, numero_pagina
from ocr_historico.entradas import carregar_entrada
from ocr_historico.executar import avaliar_pasta, executar
from ocr_historico.metricas import calcular, punctuation_error_rate
from ocr_historico.motor_llm import TEXTO_SIMULADO, GastoNaoAutorizado, MotorLLM
from ocr_historico.resultados import carregar, importar_resultado_llm, salvar

NOME = '{n}.BR FPMSC FUN CAM LDL - Livro de Registro de Leis n° 1, 1892 - 1984'


@pytest.fixture
def acervo(tmp_path: Path) -> dict[str, Path]:
    pastas = {nome: tmp_path / nome for nome in ('imagens', 'binarizadas', 'gabaritos', 'resultados')}
    for pasta in pastas.values():
        pasta.mkdir()
    for n in (5, 10):
        Image.new('RGB', (40, 20), 'white').save(pastas['imagens'] / f'{NOME.format(n=n)}.jpg')
        cinza = np.full((20, 40), 250, 'uint8')
        cinza[5:15, 5:35] = 20
        Image.fromarray(cinza).save(pastas['binarizadas'] / f'binaria_{NOME.format(n=n)}.jpg')
        (pastas['gabaritos'] / f'{NOME.format(n=n)}.txt').write_text(f'Lei nº {n}, de Janeiro.', encoding='utf-8')
    return pastas


def paginas_de(acervo):
    return listar_paginas(imagens=acervo['imagens'], binarizadas=acervo['binarizadas'],
                          gabaritos=acervo['gabaritos'])


class KrakenFalso:
    nome = 'kraken/falso'

    def __init__(self):
        self.modos = []

    def transcrever(self, imagem):
        self.modos.append(imagem.mode)
        return 'Lei nº 5, de Janeiro.'


def test_paginas_ligam_original_binarizada_e_gabarito_pelo_numero(acervo) -> None:
    paginas = paginas_de(acervo)
    assert [p.numero for p in paginas] == [5, 10]
    assert paginas[1].binarizada.name.startswith('binaria_10.')
    assert paginas[0].texto_gabarito() == 'Lei nº 5, de Janeiro.'
    assert numero_pagina('binaria_12.BR x.jpg') == 12
    with pytest.raises(ValueError):
        listar_paginas([99], imagens=acervo['imagens'])


def test_binarizada_do_time_volta_a_ser_binaria_e_ampliada(acervo) -> None:
    pagina = paginas_de(acervo)[0]
    imagem = carregar_entrada(pagina, 'binarizada')
    assert set(np.unique(np.asarray(imagem))) <= {0, 255}
    assert carregar_entrada(pagina, 'binarizada', escala=2).size == (80, 40)
    assert carregar_entrada(pagina, 'original').mode == 'RGB'
    with pytest.raises(ValueError):
        carregar_entrada(pagina, 'sepia')


def test_nomes_das_combinacoes() -> None:
    assert interpretar('kraken_binarizada_2x').escala == 2
    correcao = interpretar('llm_corrige_kraken_binarizada')
    assert (correcao.motor, correcao.base, correcao.gasta_dinheiro) == ('correcao', 'kraken_binarizada', True)
    assert not interpretar('kraken_original').gasta_dinheiro
    for nome in PADRAO_KRAKEN + PADRAO_LLM:
        interpretar(nome)
    for invalido in ('llm_corrige_llm_original', 'tesseract_original', 'kraken_sepia'):
        with pytest.raises(ValueError):
            interpretar(invalido)


def test_metricas_do_notebook_e_estrita() -> None:
    perfeito = calcular(['Lei nº 5, de Janeiro.'], ['Lei nº 5, de Janeiro.'])
    assert all(valor == 0 for valor in perfeito.values())
    caixa = calcular(['São Carlos.'], ['são carlos'])
    assert caixa['wer_estrito'] == 1 and caixa['wer_sem_pontuacao'] == 0
    assert punctuation_error_rate('a, b.', 'a b.') == pytest.approx(0.5)
    vazio = calcular(['a b'], [''])
    assert vazio['wer_sem_pontuacao'] == 1


def test_kraken_gera_json_retoma_e_avalia(acervo) -> None:
    paginas, kraken = paginas_de(acervo), KrakenFalso()
    destino = executar(interpretar('kraken_binarizada'), paginas[:1], acervo['resultados'], kraken)
    executar(interpretar('kraken_binarizada'), paginas, acervo['resultados'], kraken)
    assert kraken.modos == ['L', 'L']  # a página 5 não foi refeita
    resultado = carregar(destino)
    assert [i['arquivo'][:3] for i in resultado['imagens']] == ['5.B', '10.']
    assert resultado['entrada'] == 'binarizada'
    resumo = avaliar_pasta(acervo['resultados'], paginas)
    assert resumo[0]['wer_sem_pontuacao'] == pytest.approx(1 / 10)  # só "5" ≠ "10" na página 10
    assert (acervo['resultados'] / 'comparacao.csv').exists()


def test_llm_bloqueia_gasto_sem_autorizacao(acervo, monkeypatch) -> None:
    def proibido(*args, **kwargs):
        raise AssertionError('nenhuma requisição deveria sair')

    monkeypatch.setattr('ocr_historico.motor_llm.requests.post', proibido)
    paginas = paginas_de(acervo)
    with pytest.raises(GastoNaoAutorizado):
        executar(interpretar('llm_original'), paginas, acervo['resultados'], motor_llm=MotorLLM())


def test_correcao_simulada_usa_texto_do_kraken_e_a_imagem(acervo, monkeypatch) -> None:
    monkeypatch.setattr('ocr_historico.motor_llm.requests.post', None)
    paginas, llm = paginas_de(acervo), MotorLLM(simular=True)
    executar(interpretar('kraken_binarizada'), paginas, acervo['resultados'], KrakenFalso())
    simulacao = acervo['resultados'] / 'simulacao'
    destino = executar(interpretar('llm_corrige_kraken_binarizada'), paginas, simulacao, motor_llm=llm)
    assert [i['texto_transcrito'] for i in carregar(destino)['imagens']] == [TEXTO_SIMULADO] * 2
    corpo = llm.requisicoes_simuladas[0]
    texto, imagem = corpo['messages'][0]['content']
    assert 'Lei nº 5, de Janeiro.' in texto['text']
    assert imagem['image_url']['url'].startswith('data:image/png;base64,')
    assert corpo['model'] == 'google/gemini-3.1-pro-preview' and corpo['reasoning'] == {'effort': 'high'}


def test_llm_envia_e_registra_custo(acervo, monkeypatch) -> None:
    class Resposta:
        ok = True

        def json(self):
            return {'choices': [{'message': {'content': 'Lei nº 10, de Janeiro.'}}], 'usage': {'cost': 0.05}}

    enviados = []
    monkeypatch.setattr('ocr_historico.motor_llm.requests.post', lambda *a, **k: enviados.append(k) or Resposta())
    monkeypatch.setenv('OPENROUTER_API_KEY', 'chave-de-teste')
    paginas = paginas_de(acervo)
    destino = executar(interpretar('llm_binarizada'), paginas[1:], acervo['resultados'],
                       motor_llm=MotorLLM(permitir_gasto=True))
    assert enviados[0]['headers']['Authorization'] == 'Bearer chave-de-teste'
    assert carregar(destino)['imagens'][0]['custo'] == 0.05


def test_importa_json_do_notebook_llm(acervo, tmp_path) -> None:
    origem = tmp_path / 'resultados_transcricao (2).json'
    origem.write_text(json.dumps({'modelo': 'google/gemini-3.1-pro-preview', 'esforco_raciocinio': 'high',
                                  'imagens': [{'arquivo': f'/content/x/Imagens/{NOME.format(n=5)}.jpg',
                                               'texto_transcrito': 'Lei nº 5, de Janeiro.', 'custo': 0.06}]}),
                      encoding='utf-8')
    importar_resultado_llm(origem, acervo['resultados'])
    linhas = avaliar_pasta(acervo['resultados'], paginas_de(acervo)[:1])
    assert linhas[0]['combinacao'] == 'llm_original' and linhas[0]['wer_estrito'] == 0


def test_resultado_de_outro_modelo_nao_e_misturado(acervo) -> None:
    salvar({'combinacao': 'kraken_original', 'modelo': 'kraken/outro', 'imagens': []},
           acervo['resultados'] / 'kraken_original.json')
    with pytest.raises(ValueError):
        executar(interpretar('kraken_original'), paginas_de(acervo), acervo['resultados'], KrakenFalso())


class KrakenFtFalso(KrakenFalso):
    nome = 'kraken/best_0.8182'
    paginas_treino = [5]


def test_combinacoes_do_modelo_finetunado() -> None:
    ft = interpretar('kraken_ft_original_2x')
    assert (ft.motor, ft.entrada, ft.escala, ft.finetunado) == ('kraken', 'original', 2.0, True)
    assert not interpretar('kraken_original_2x').finetunado
    assert interpretar('llm_corrige_kraken_ft_original_2x').base == 'kraken_ft_original_2x'
    with pytest.raises(ValueError):
        interpretar('llm_ft_original')


def test_finetunado_marca_paginas_de_treino_ate_na_correcao(acervo) -> None:
    paginas = paginas_de(acervo)
    with pytest.raises(ValueError, match='fine-tunado'):
        executar(interpretar('kraken_ft_original'), paginas, acervo['resultados'], motor_kraken=KrakenFalso())
    executar(interpretar('kraken_ft_original'), paginas, acervo['resultados'], motor_kraken_ft=KrakenFtFalso())
    assert carregar(acervo['resultados'] / 'kraken_ft_original.json')['paginas_treino'] == [5]
    simulacao = acervo['resultados'] / 'simulacao'
    executar(interpretar('llm_corrige_kraken_ft_original'), paginas, simulacao, motor_llm=MotorLLM(simular=True))
    assert carregar(simulacao / 'llm_corrige_kraken_ft_original.json')['paginas_treino'] == [5]

    resumo = {l['combinacao']: l for l in avaliar_pasta(acervo['resultados'], paginas)}
    assert 'inclui páginas de treino [5]' in resumo['kraken_ft_original']['status']
    so_reservada = avaliar_pasta(acervo['resultados'], paginas[1:], sufixo='_reservadas')
    assert so_reservada[0]['status'] == 'ok'
    assert (acervo['resultados'] / 'comparacao_reservadas.csv').exists()
