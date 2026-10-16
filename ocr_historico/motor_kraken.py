"""Kraken do test_kraken (feat/OCR) como motor: imagem PIL entra, texto sai."""
import logging
import shutil
import sys
import time
from pathlib import Path

import requests
from PIL import Image

from .dados import PASTA_GABARITOS, PASTA_IMAGENS, RAIZ

PASTA_TEST_KRAKEN = RAIZ / 'test_kraken'
if str(PASTA_TEST_KRAKEN) not in sys.path:
    sys.path.insert(0, str(PASTA_TEST_KRAKEN))

from kraken_pipeline.config import PipelineConfig  # noqa: E402

logger = logging.getLogger(__name__)


def _baixar_retomando(url: str, parcial: Path, tentativas: int, espera: float) -> None:
    """Baixa para `parcial`; a cada falha de rede, retoma do byte em que parou (HTTP Range)."""
    for tentativa in range(1, tentativas + 1):
        inicio = parcial.stat().st_size if parcial.exists() else 0
        try:
            cabecalhos = {'Range': f'bytes={inicio}-'} if inicio else {}
            with requests.get(url, stream=True, timeout=(60, 300), headers=cabecalhos) as resposta:
                if resposta.status_code == 416:  # o arquivo parcial já está completo
                    return
                resposta.raise_for_status()
                # 206: o servidor continuou de onde parou; 200: mandou o arquivo inteiro de novo
                with parcial.open('ab' if resposta.status_code == 206 else 'wb') as arquivo:
                    for bloco in resposta.iter_content(chunk_size=1024 * 1024):
                        arquivo.write(bloco)
            return
        except requests.RequestException as erro:
            if tentativa == tentativas:
                raise RuntimeError(f'Zenodo indisponível após {tentativas} tentativas ({erro}). Tente mais '
                                   'tarde ou passe uma cópia do modelo em copia_local.') from erro
            logger.warning('Download interrompido (%s); tentativa %d de %d em %.0f s',
                           erro, tentativa + 1, tentativas, espera * tentativa)
            time.sleep(espera * tentativa)


def baixar_modelo(variante: str = 'medium', pasta: Path = RAIZ / 'models', copia_local: Path | None = None,
                  tentativas: int = 8, espera: float = 10.0) -> Path:
    """PP-OCRv6 do Zenodo (mesmo modelo do test_kraken), com MD5 verificado.

    O Zenodo às vezes derruba a conexão: o download é retomado de onde parou. Com
    `copia_local` (por exemplo, um arquivo no Google Drive), o Zenodo não é usado,
    mas o MD5 da cópia é conferido do mesmo jeito.
    """
    from kraken_pipeline.model_management import (MULTILINGUAL_MEDIUM_MODEL, MULTILINGUAL_MODEL,
                                                  _verify_model, download_multilingual_model)

    modelo = {'small': MULTILINGUAL_MODEL, 'medium': MULTILINGUAL_MEDIUM_MODEL}[variante]
    pesos = pasta / modelo.doi.replace('/', '_') / modelo.filename
    if not pesos.exists():
        pesos.parent.mkdir(parents=True, exist_ok=True)
        parcial = pesos.with_suffix(pesos.suffix + '.part')
        if copia_local is not None:
            shutil.copyfile(copia_local, parcial)
        else:
            _baixar_retomando(modelo.url, parcial, tentativas, espera)
        try:
            _verify_model(parcial, modelo)
        except ValueError:
            parcial.unlink()  # corrompido: a próxima chamada recomeça do zero
            raise
        parcial.replace(pesos)
    # com os pesos no lugar, o test_kraken só confere o MD5 de novo e grava a procedência
    return download_multilingual_model(pasta, variant=variante)


class MotorKraken:
    def __init__(self, modelo: Path, dispositivo: str = 'cpu', paginas_treino: list[int] | None = None) -> None:
        from kraken_pipeline.engine import KrakenEngine

        self.modelo = Path(modelo)
        # páginas vistas no fine-tuning: avaliá-lo nelas infla o resultado
        self.paginas_treino = sorted(paginas_treino or [])
        self.engine = KrakenEngine(PipelineConfig(
            image_dir=PASTA_IMAGENS, reference_dir=PASTA_GABARITOS,
            recognition_model=self.modelo, device=dispositivo))

    @property
    def nome(self) -> str:
        return f'kraken/{self.modelo.stem}'

    def transcrever(self, imagem: Image.Image) -> str:
        segmentacao = self.engine.segment(imagem)
        linhas = self.engine.recognize(imagem, segmentacao)
        return '\n'.join(linha.prediction for linha in linhas)
