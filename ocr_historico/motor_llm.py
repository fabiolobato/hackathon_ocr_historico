"""Transcrição e correção por LLM via OpenRouter (llm_implementation).

Nenhuma requisição paga é feita sem `permitir_gasto=True`. Com `simular=True` a
requisição é montada e validada, mas não é enviada: serve para testar o fluxo.
"""
import base64
import io
import os
from dataclasses import dataclass, field
from typing import Any

import requests
from PIL import Image

from .dados import RAIZ

URL_OPENROUTER = 'https://openrouter.ai/api/v1/chat/completions'
MODELO_PADRAO = 'google/gemini-3.1-pro-preview'
ESFORCO_PADRAO = 'high'
TEXTO_SIMULADO = '[simulação: nenhuma requisição enviada]'

PROMPT_TRANSCRICAO = (
    'Transcreva todo o texto presente nesta imagem, preservando a grafia original. '
    'Marque trechos ilegíveis como [ilegível] e não utilize quebras de linha como \\n. '
    'Retorne apenas a transcrição, sem introdução, explicações ou avaliação.'
)
PROMPT_CORRECAO = (
    'A imagem é uma página manuscrita do Livro de Registro de Leis nº 1 (São Carlos do Pinhal, '
    '1892-1894). Abaixo está a saída de um OCR (Kraken) da mesma página, linha a linha, com erros '
    'de reconhecimento. Corrija a transcrição comparando-a com a imagem, preservando a grafia '
    'original da época (não modernize nem expanda abreviações). Use o OCR como apoio para a '
    'ordem e o conteúdo das linhas, mas a imagem prevalece em caso de conflito. Marque trechos '
    'ilegíveis como [ilegível] e não utilize quebras de linha como \\n. Retorne apenas a '
    'transcrição, sem introdução, explicações ou avaliação.\n\nOCR:\n{ocr}'
)


class GastoNaoAutorizado(RuntimeError):
    pass


def obter_chave() -> str | None:
    """OPENROUTER_API_KEY/API_KEY do ambiente, segredo API_KEY do Colab ou arquivo .env na raiz."""
    for nome in ('OPENROUTER_API_KEY', 'API_KEY'):
        if os.environ.get(nome):
            return os.environ[nome]
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        return userdata.get('API_KEY')
    except Exception:  # noqa: BLE001 - fora do Colab ou segredo não liberado
        pass
    env = RAIZ / '.env'
    if env.is_file():
        for linha in env.read_text(encoding='utf-8').splitlines():
            nome, _, valor = linha.partition('=')
            if nome.strip() in ('OPENROUTER_API_KEY', 'API_KEY') and valor.strip():
                return valor.strip().strip('"\'')
    return None


def imagem_para_data_url(imagem: Image.Image) -> str:
    # PNG é sem perdas: mantém a binarização exatamente como foi gerada.
    buffer = io.BytesIO()
    imagem.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


@dataclass
class MotorLLM:
    modelo: str = MODELO_PADRAO
    esforco: str = ESFORCO_PADRAO
    permitir_gasto: bool = False
    simular: bool = False
    max_tokens: int = 8192
    requisicoes_simuladas: list[dict[str, Any]] = field(default_factory=list)

    @property
    def nome(self) -> str:
        return self.modelo

    def montar_requisicao(self, conteudo: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            'model': self.modelo,
            'messages': [{'role': 'user', 'content': conteudo}],
            'reasoning': {'effort': self.esforco},
            'max_tokens': self.max_tokens,
        }

    def conversar(self, conteudo: list[dict[str, Any]]) -> tuple[str, float | None]:
        corpo = self.montar_requisicao(conteudo)
        if self.simular:
            self.requisicoes_simuladas.append(corpo)
            return TEXTO_SIMULADO, 0.0
        if not self.permitir_gasto:
            raise GastoNaoAutorizado('Requisição paga bloqueada. Use permitir_gasto=True '
                                     '(ou --permitir-gasto) depois de aprovar o custo, ou simular=True.')
        chave = obter_chave()
        if not chave:
            raise GastoNaoAutorizado('Chave não encontrada: defina OPENROUTER_API_KEY, o segredo API_KEY '
                                     'do Colab ou um arquivo .env na raiz do repositório.')
        resposta = requests.post(URL_OPENROUTER, json=corpo, timeout=(30, 300),
                                 headers={'Authorization': f'Bearer {chave}',
                                          'Content-Type': 'application/json'})
        if not resposta.ok:
            raise RuntimeError(f'OpenRouter: HTTP {resposta.status_code} — {resposta.text}')
        dados = resposta.json()
        if 'error' in dados:
            raise RuntimeError(f"OpenRouter: {dados['error']}")
        texto = dados['choices'][0]['message'].get('content')
        if not isinstance(texto, str) or not texto.strip():
            raise RuntimeError('O modelo não retornou texto. Verifique a resposta e o limite de tokens.')
        return texto, dados.get('usage', {}).get('cost')

    def transcrever(self, imagem: Image.Image) -> tuple[str, float | None]:
        return self.conversar([
            {'type': 'text', 'text': PROMPT_TRANSCRICAO},
            {'type': 'image_url', 'image_url': {'url': imagem_para_data_url(imagem)}},
        ])

    def corrigir(self, imagem: Image.Image, texto_ocr: str) -> tuple[str, float | None]:
        return self.conversar([
            {'type': 'text', 'text': PROMPT_CORRECAO.format(ocr=texto_ocr)},
            {'type': 'image_url', 'image_url': {'url': imagem_para_data_url(imagem)}},
        ])
