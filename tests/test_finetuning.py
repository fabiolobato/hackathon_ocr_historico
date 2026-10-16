from pathlib import Path

import pytest
from PIL import Image

from ocr_historico.alinhamento import alinhar_posicoes, distancia, dividir_gabarito
from ocr_historico.dados import listar_paginas
from ocr_historico.finetuning import comando_treino, dividir_linhas, dividir_paginas, melhor_modelo


def test_alinhamento_atravessa_edicoes() -> None:
    assert alinhar_posicoes('abc', 'abc') == [0, 1, 2, 3]
    assert alinhar_posicoes('axc', 'abc') == [0, 1, 2, 3]
    assert alinhar_posicoes('ac', 'abc')[1] == 2
    assert distancia('abc', 'abd') == pytest.approx(1 / 3)
    assert distancia('', '') == 0 and distancia('abc', '') == 1


def test_gabarito_e_cortado_nas_quebras_do_ocr_ruidoso() -> None:
    ocr = ["Os casos omissos n'ute rymicuto serao regulado",
           'pelo disporto no Dea de 15 d Janeiio de 1870, na',
           'Lei no 16 a 13 de Novunbro de 1891']
    gabarito = ('Os casos omissos neste regimento serão regulados pelo disposto no Dec '
                'de 15 de janeiro de 1890, na Lei nº 16 de 13 de Novembro de 1891')
    assert dividir_gabarito(ocr, gabarito) == [
        'Os casos omissos neste regimento serão regulados',
        'pelo disposto no Dec de 15 de janeiro de 1890, na',
        'Lei nº 16 de 13 de Novembro de 1891']


def test_linha_fora_do_gabarito_fica_vazia() -> None:
    assert dividir_gabarito(['Artigo 1º', '7', 'A Camara resolve:'], 'Artigo 1º A Camara resolve:') == [
        'Artigo 1º', '', 'A Camara resolve:']


def test_paginas_reservadas_iguais_ao_test_kraken() -> None:
    treino, reserva = dividir_paginas(listar_paginas())
    assert [p.numero for p in reserva] == [7, 10, 12]
    assert [p.numero for p in treino] == [5, 6, 8, 9, 11, 13, 14, 15, 16]


def test_listas_de_treino_e_validacao(tmp_path: Path) -> None:
    for i in range(20):
        Image.new('L', (30, 10)).save(tmp_path / f'p05_{i:03d}.png')
    treino, validacao = dividir_linhas(tmp_path)
    linhas_treino = treino.read_text().split()
    linhas_validacao = validacao.read_text().split()
    assert len(linhas_treino) == 18 and len(linhas_validacao) == 2
    assert not set(linhas_treino) & set(linhas_validacao)
    assert all(Path(p).is_absolute() for p in linhas_treino)
    with pytest.raises(ValueError):
        dividir_linhas(tmp_path / 'vazia')


def test_comando_e_melhor_modelo(tmp_path: Path) -> None:
    comando = comando_treino(Path('m.safetensors'), Path('t.txt'), Path('v.txt'), tmp_path, 'cuda:0', epocas=5)
    assert comando[comando.index('-d') + 1] == 'cuda:0'
    assert comando[comando.index('-N') + 1] == comando[comando.index('--cos-max') + 1] == '5'
    assert ['-q', 'fixed'] == comando[comando.index('-q'):comando.index('-q') + 2]
    with pytest.raises(FileNotFoundError):
        melhor_modelo(tmp_path)
    for score in ('0.7012', '0.8123', '0.7999'):
        (tmp_path / f'best_{score}.safetensors').touch()
    assert melhor_modelo(tmp_path).name == 'best_0.8123.safetensors'
