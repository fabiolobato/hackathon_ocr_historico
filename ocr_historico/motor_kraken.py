"""Kraken do test_kraken (feat/OCR) como motor: imagem PIL entra, texto sai."""
import sys
from pathlib import Path

from PIL import Image

from .dados import PASTA_GABARITOS, PASTA_IMAGENS, RAIZ

PASTA_TEST_KRAKEN = RAIZ / 'test_kraken'
if str(PASTA_TEST_KRAKEN) not in sys.path:
    sys.path.insert(0, str(PASTA_TEST_KRAKEN))

from kraken_pipeline.config import PipelineConfig  # noqa: E402


def baixar_modelo(variante: str = 'medium', pasta: Path = RAIZ / 'models') -> Path:
    """PP-OCRv6 do Zenodo, com MD5 verificado (mesmo modelo usado pelo test_kraken)."""
    from kraken_pipeline.model_management import download_multilingual_model

    return download_multilingual_model(pasta, variant=variante)


class MotorKraken:
    def __init__(self, modelo: Path, dispositivo: str = 'cpu') -> None:
        from kraken_pipeline.engine import KrakenEngine

        self.modelo = Path(modelo)
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
