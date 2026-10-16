import hashlib
from pathlib import Path

import pytest
import requests

import ocr_historico.motor_kraken as motor_kraken
from kraken_pipeline import model_management
from kraken_pipeline.model_management import ZenodoModel

CONTEUDO = bytes(range(256)) * 40  # 10 KiB de "pesos"


@pytest.fixture
def modelo_falso(monkeypatch):
    falso = ZenodoModel('10.5281/zenodo.1', 'medium.safetensors', hashlib.md5(CONTEUDO).hexdigest())
    monkeypatch.setattr(model_management, 'MULTILINGUAL_MEDIUM_MODEL', falso)
    monkeypatch.setattr(motor_kraken.time, 'sleep', lambda _: None)
    return falso


class Resposta:
    def __init__(self, status, dados=b'', cair_depois=None):
        self.status_code, self.dados, self.cair_depois = status, dados, cair_depois

    def __enter__(self):
        return self

    def __exit__(self, *erro):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)

    def iter_content(self, chunk_size):
        for inicio in range(0, len(self.dados), 1000):
            if self.cair_depois is not None and inicio >= self.cair_depois:
                raise requests.ConnectionError('conexão derrubada')
            yield self.dados[inicio:inicio + 1000]


def servidor(monkeypatch, respostas):
    pedidos = []

    def get(url, stream, timeout, headers):
        pedidos.append(headers.get('Range'))
        resposta = respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta

    monkeypatch.setattr(motor_kraken.requests, 'get', get)
    return pedidos


def test_retoma_de_onde_parou(tmp_path, monkeypatch, modelo_falso) -> None:
    pedidos = servidor(monkeypatch, [
        requests.ConnectTimeout('handshake'),
        Resposta(200, CONTEUDO, cair_depois=4000),
        Resposta(206, CONTEUDO[4000:]),
    ])
    pesos = motor_kraken.baixar_modelo('medium', tmp_path)
    assert pesos.read_bytes() == CONTEUDO
    assert pedidos == [None, None, 'bytes=4000-']
    assert (pesos.parent / 'download.json').exists()
    assert motor_kraken.baixar_modelo('medium', tmp_path) == pesos  # depois usa o cache, sem rede


def test_servidor_sem_range_recomeca_do_zero(tmp_path, monkeypatch, modelo_falso) -> None:
    servidor(monkeypatch, [Resposta(200, CONTEUDO, cair_depois=3000), Resposta(200, CONTEUDO)])
    assert motor_kraken.baixar_modelo('medium', tmp_path).read_bytes() == CONTEUDO


def test_desiste_depois_das_tentativas(tmp_path, monkeypatch, modelo_falso) -> None:
    servidor(monkeypatch, [requests.ReadTimeout('lento')] * 3)
    with pytest.raises(RuntimeError, match='copia_local'):
        motor_kraken.baixar_modelo('medium', tmp_path, tentativas=3)


def test_copia_local_e_conferida(tmp_path, monkeypatch, modelo_falso) -> None:
    servidor(monkeypatch, [])  # nenhuma requisição deveria sair
    corrompida = tmp_path / 'drive.safetensors'
    corrompida.write_bytes(CONTEUDO[:-1] + b'x')
    with pytest.raises(ValueError, match='checksum'):
        motor_kraken.baixar_modelo('medium', tmp_path / 'models', copia_local=corrompida)
    assert not list((tmp_path / 'models').rglob('*.part'))
    boa = tmp_path / 'boa.safetensors'
    boa.write_bytes(CONTEUDO)
    assert motor_kraken.baixar_modelo('medium', tmp_path / 'models', copia_local=boa).read_bytes() == CONTEUDO
