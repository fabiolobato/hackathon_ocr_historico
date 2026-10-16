"""Avaliação única para todos os motores.

As métricas sem/com pontuação e a PER vêm do LLM_Transcription.ipynb. `wer_estrito`
é a métrica do test_kraken (caixa, acentos e pontuação preservados, NFC).
Os valores de corpus somam erros de todas as páginas; não são médias de porcentagens.
"""
import csv
import re
import unicodedata
from pathlib import Path
from typing import Any

import jiwer

from .dados import Pagina
from .resultados import textos_por_arquivo

_PUNCT_CHAR = r"(?!(?<=\w)['’-](?=\w))[^\w\s]"
_PUNCT_SUB = re.compile(f"({_PUNCT_CHAR})")
_IS_PUNCT = re.compile(r"^[^\w\s]$")
PUNCT = "[PUNCT]"
METRICAS = ('wer_estrito', 'wer_sem_pontuacao', 'cer_sem_pontuacao',
            'wer_com_pontuacao', 'cer_com_pontuacao', 'per')


def normalize_text(text):
    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(_PUNCT_CHAR, "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_separate_punctuation(text):
    text = unicodedata.normalize('NFC', text).lower()
    text = _PUNCT_SUB.sub(r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_strict(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def _mask(tokens):
    return [PUNCT if _IS_PUNCT.match(t) else t for t in tokens]


def _align_punctuation(ref, hyp):
    """Alinha as sequências mascaradas e devolve os pares (i, j) de pontuações alinhadas."""
    ref_m, hyp_m = _mask(ref), _mask(hyp)
    n, m = len(ref), len(hyp)
    big = n + m + 1

    def step(i, j):
        if ref_m[i] != hyp_m[j]:
            return big
        if ref_m[i] == PUNCT and ref[i] != hyp[j]:
            return 1
        return 0

    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = i * big
    for j in range(1, m + 1):
        dist[0][j] = j * big
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(
                dist[i - 1][j - 1] + step(i - 1, j - 1),
                dist[i - 1][j] + big,
                dist[i][j - 1] + big,
            )

    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if dist[i][j] == dist[i - 1][j - 1] + step(i - 1, j - 1):
            if ref_m[i - 1] == PUNCT and hyp_m[j - 1] == PUNCT:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dist[i][j] == dist[i - 1][j] + big:
            i -= 1
        else:
            j -= 1
    return pairs


def punctuation_counts(reference, hypothesis):
    """Espera textos já passados por normalize_separate_punctuation."""
    ref, hyp = reference.split(), hypothesis.split()
    pairs = _align_punctuation(ref, hyp)
    n_ref = sum(1 for t in ref if _IS_PUNCT.match(t))
    n_hyp = sum(1 for t in hyp if _IS_PUNCT.match(t))
    correct = sum(ref[i] == hyp[j] for i, j in pairs)
    return {"C": correct, "S": len(pairs) - correct, "D": n_ref - len(pairs), "I": n_hyp - len(pairs)}


def _per(c):
    errors = c["S"] + c["D"] + c["I"]
    total = errors + c["C"]
    return errors / total if total else 0.0


def punctuation_error_rate(reference, hypothesis):
    """PER = (Ip + Dp + Sp) / (Ip + Dp + Sp + Cp), conforme https://arxiv.org/abs/2310.02943"""
    return _per(punctuation_counts(normalize_separate_punctuation(reference),
                                   normalize_separate_punctuation(hypothesis)))


def corpus_punctuation_error_rate(references, hypotheses):
    total = {"C": 0, "S": 0, "D": 0, "I": 0}
    for r, h in zip(references, hypotheses):
        for k, v in punctuation_counts(normalize_separate_punctuation(r),
                                       normalize_separate_punctuation(h)).items():
            total[k] += v
    return _per(total)


def calcular(referencias: list[str], hipoteses: list[str]) -> dict[str, float]:
    """Métricas para uma página (listas de um item) ou para o corpus inteiro."""
    def taxa(funcao, normalizar):
        return funcao([normalizar(r) for r in referencias], [normalizar(h) for h in hipoteses])

    return {
        'wer_estrito': taxa(jiwer.wer, normalize_strict),
        'wer_sem_pontuacao': taxa(jiwer.wer, normalize_text),
        'cer_sem_pontuacao': taxa(jiwer.cer, normalize_text),
        'wer_com_pontuacao': taxa(jiwer.wer, normalize_separate_punctuation),
        'cer_com_pontuacao': taxa(jiwer.cer, normalize_separate_punctuation),
        'per': corpus_punctuation_error_rate(referencias, hipoteses),
    }


def avaliar(resultado: dict[str, Any], paginas: list[Pagina]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Linhas por página e uma linha de corpus. Páginas sem transcrição contam como texto vazio."""
    textos = textos_por_arquivo(resultado)
    base = {'combinacao': resultado.get('combinacao', ''), 'modelo': resultado.get('modelo', ''),
            'entrada': resultado.get('entrada', ''), 'escala': resultado.get('escala', '')}
    linhas, referencias, hipoteses, faltando = [], [], [], []
    custo_total = 0.0
    for pagina in paginas:
        if pagina.gabarito is None:
            continue
        referencia = pagina.texto_gabarito()
        hipotese = textos.get(pagina.arquivo)
        if hipotese is None:
            faltando.append(pagina.numero)
            hipotese = ''
        custo = next((item.get('custo') for item in resultado['imagens']
                      if item['arquivo'] == pagina.arquivo), None)
        custo_total += custo or 0.0
        referencias.append(referencia)
        hipoteses.append(hipotese)
        linhas.append({**base, 'pagina': pagina.numero, 'custo': custo,
                       'status': 'sem transcrição' if pagina.numero in faltando else 'ok',
                       **calcular([referencia], [hipotese])})
    if not referencias:
        raise ValueError('Nenhuma página selecionada tem gabarito.')
    corpus = {**base, 'pagina': 'corpus', 'custo': custo_total,
              'status': f'faltando {faltando}' if faltando else 'ok', **calcular(referencias, hipoteses)}
    return linhas, corpus


def salvar_csv(linhas: list[dict[str, Any]], caminho: Path) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    colunas = ['combinacao', 'modelo', 'entrada', 'escala', 'pagina', 'custo', 'status', *METRICAS]
    with caminho.open('w', newline='', encoding='utf-8-sig') as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas, extrasaction='ignore')
        escritor.writeheader()
        escritor.writerows(linhas)
    return caminho
