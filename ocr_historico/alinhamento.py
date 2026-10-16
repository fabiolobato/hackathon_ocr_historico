"""Corta o gabarito de uma página (texto corrido) em linhas, seguindo as quebras do OCR.

O gabarito não tem as quebras de linha do manuscrito. O texto do OCR é alinhado ao
gabarito caractere a caractere, e o gabarito é cortado no espaço mais próximo de cada
quebra de linha do OCR.
"""
import unicodedata

import numpy as np


def _custo_substituicao(caractere: str, outro: str) -> int:
    # quebras de linha do OCR não existem no gabarito: nunca alinhá-las a um caractere
    return 2 if caractere == '\n' else int(caractere != outro)


def _levenshtein(a: str, b: str) -> np.ndarray:
    tabela = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    tabela[:, 0] = np.arange(len(a) + 1)
    tabela[0, :] = np.arange(len(b) + 1)
    codigos = np.frombuffer(b.encode('utf-32-le'), dtype=np.uint32)
    indices = np.arange(len(b) + 1)
    for i, caractere in enumerate(a, 1):
        substituicao = tabela[i - 1, :-1] + (2 if caractere == '\n' else (codigos != ord(caractere)))
        linha = np.minimum(substituicao, tabela[i - 1, 1:] + 1)
        # inserções dependem da célula à esquerda: mínimo acumulado resolve a linha inteira
        tabela[i] = np.minimum.accumulate(np.concatenate(([i], linha)) - indices) + indices
    return tabela


def alinhar_posicoes(origem: str, destino: str) -> list[int]:
    """Para cada índice de `origem` (e o fim), o índice alinhado em `destino`."""
    tabela = _levenshtein(origem, destino)
    mapa = [0] * (len(origem) + 1)
    mapa[len(origem)] = len(destino)
    i, j = len(origem), len(destino)
    while i > 0:
        if j > 0 and tabela[i, j] == tabela[i - 1, j - 1] + _custo_substituicao(origem[i - 1], destino[j - 1]):
            i, j = i - 1, j - 1
        elif tabela[i, j] == tabela[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
            continue
        mapa[i] = j
    return mapa


def distancia(a: str, b: str) -> float:
    """Distância de edição normalizada, de 0 (iguais) a 1."""
    if not a and not b:
        return 0.0
    return float(_levenshtein(a, b)[-1, -1]) / max(len(a), len(b))


def dividir_gabarito(linhas_ocr: list[str], gabarito: str, folga: int = 10) -> list[str]:
    """Uma linha do gabarito para cada linha do OCR (vazia quando não há correspondente)."""
    texto = ' '.join(unicodedata.normalize('NFC', gabarito).split())
    mapa = alinhar_posicoes('\n'.join(linhas_ocr).casefold(), texto.casefold())
    cortes, posicao = [0], 0
    for linha in linhas_ocr[:-1]:
        posicao += len(linha)
        alvo = mapa[posicao]
        candidatos = [k for k in range(max(cortes[-1], alvo - folga), min(len(texto), alvo + folga) + 1)
                      if k in (0, len(texto)) or texto[k] == ' ' or texto[k - 1] == ' ']
        cortes.append(min(candidatos, key=lambda k: abs(k - alvo)) if candidatos else max(alvo, cortes[-1]))
        posicao += 1
    cortes.append(len(texto))
    return [texto[a:b].strip() for a, b in zip(cortes, cortes[1:])]
