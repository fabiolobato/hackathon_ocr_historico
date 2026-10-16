"""Formato comum de saída: o mesmo JSON do LLM_Transcription.ipynb, para todos os motores.

{"combinacao": ..., "modelo": ..., "esforco_raciocinio": ..., "entrada": ..., "escala": ...,
 "imagens": [{"arquivo": ..., "texto_transcrito": ..., "custo": ...}]}

Um JSON gerado pelo notebook do LLM pode ser importado sem conversão.
"""
import json
from pathlib import Path
from typing import Any

from .dados import normalizar_nome


def novo_resultado(combinacao: str, modelo: str, **detalhes: Any) -> dict[str, Any]:
    return {'combinacao': combinacao, 'modelo': modelo, **detalhes, 'imagens': []}


def carregar(caminho: Path) -> dict[str, Any]:
    return json.loads(caminho.read_text(encoding='utf-8'))


def salvar(resultado: dict[str, Any], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix('.tmp')
    temporario.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    temporario.replace(caminho)


def textos_por_arquivo(resultado: dict[str, Any]) -> dict[str, str]:
    return {normalizar_nome(item['arquivo']): item['texto_transcrito'] for item in resultado['imagens']}


def registrar_pagina(resultado: dict[str, Any], arquivo: str, texto: str,
                     custo: float | None = None, **extra: Any) -> None:
    """Inclui ou substitui a página, mantendo uma entrada por arquivo."""
    arquivo = normalizar_nome(arquivo)
    resultado['imagens'] = [item for item in resultado['imagens']
                            if normalizar_nome(item['arquivo']) != arquivo]
    resultado['imagens'].append({'arquivo': arquivo, 'texto_transcrito': texto, 'custo': custo, **extra})


def importar_resultado_llm(origem: Path, destino: Path, combinacao: str = 'llm_original') -> Path:
    """Copia um JSON do LLM_Transcription.ipynb para a pasta de resultados integrada."""
    resultado = carregar(origem)
    resultado.setdefault('combinacao', combinacao)
    resultado.setdefault('entrada', 'original')
    salvar(resultado, destino / f'{combinacao}.json')
    return destino / f'{combinacao}.json'
