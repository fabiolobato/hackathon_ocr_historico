"""Imagens de entrada que qualquer motor (Kraken ou LLM) pode receber."""
import numpy as np
from PIL import Image

from .dados import Pagina

ENTRADAS = ('original', 'binarizada', 'nlbin')


def _ampliar(imagem: Image.Image, escala: float) -> Image.Image:
    if escala == 1:
        return imagem
    tamanho = (round(imagem.width * escala), round(imagem.height * escala))
    return imagem.resize(tamanho, Image.Resampling.LANCZOS)


def _limiarizar(imagem: Image.Image) -> Image.Image:
    # As binárias do time foram salvas em JPEG: a compressão cria tons de cinza nas bordas.
    cinza = np.asarray(imagem.convert('L'))
    return Image.fromarray(np.where(cinza >= 128, 255, 0).astype('uint8'))


def carregar_entrada(pagina: Pagina, entrada: str, escala: float = 1.0,
                     limiar_nlbin: float = 0.5) -> Image.Image:
    """`original` (RGB), `binarizada` (binarização do time) ou `nlbin` (binarização do Kraken)."""
    if entrada not in ENTRADAS:
        raise ValueError(f'Entrada desconhecida: {entrada}. Use uma de {ENTRADAS}.')
    if entrada == 'binarizada':
        if pagina.binarizada is None:
            raise FileNotFoundError(f'Página {pagina.numero} sem imagem em imagens_preprocessadas/')
        with Image.open(pagina.binarizada) as binaria:
            return _limiarizar(_ampliar(binaria.convert('L'), escala))
    with Image.open(pagina.imagem) as original:
        imagem = _ampliar(original.convert('RGB'), escala)
    if entrada == 'nlbin':
        from kraken.binarization import nlbin

        return nlbin(imagem, threshold=limiar_nlbin)
    return imagem
