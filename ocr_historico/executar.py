"""Executa combinações e gera a tabela comparativa.

    python -m ocr_historico.executar rodar kraken_original kraken_binarizada --paginas 7 10 12
    python -m ocr_historico.executar rodar llm_corrige_kraken_binarizada --simular
    python -m ocr_historico.executar avaliar

Cada combinação grava resultados/<nome>.json após cada página e retoma de onde parou.
Combinações de LLM só gastam créditos com --permitir-gasto; --simular grava em
resultados/simulacao/ sem enviar requisições.
"""
import argparse
import logging
from pathlib import Path
from typing import Any

from .combinacoes import PADRAO_KRAKEN, Combinacao, interpretar
from .dados import RAIZ, Pagina, listar_paginas, numero_pagina
from .entradas import carregar_entrada
from .metricas import avaliar, salvar_csv
from .resultados import carregar, novo_resultado, registrar_pagina, salvar, textos_por_arquivo

logger = logging.getLogger(__name__)
PASTA_RESULTADOS = RAIZ / 'resultados'


def executar(combinacao: Combinacao, paginas: list[Pagina], pasta: Path = PASTA_RESULTADOS,
             motor_kraken: Any = None, motor_llm: Any = None, refazer: bool = False) -> Path:
    destino = pasta / f'{combinacao.nome}.json'
    motor = motor_kraken if combinacao.motor == 'kraken' else motor_llm
    if motor is None:
        raise ValueError(f'{combinacao.nome} precisa de um motor {combinacao.motor}.')
    if destino.exists() and not refazer:
        resultado = carregar(destino)
        if resultado.get('modelo') != motor.nome:
            raise ValueError(f'{destino} foi gerado com {resultado.get("modelo")}, não {motor.nome}. '
                             'Use outra pasta ou refazer=True.')
    else:
        detalhes = {'entrada': combinacao.entrada, 'escala': combinacao.escala}
        if combinacao.motor != 'kraken':
            detalhes['esforco_raciocinio'] = motor.esforco
        if combinacao.base:
            detalhes['base'] = combinacao.base
        resultado = novo_resultado(combinacao.nome, motor.nome, **detalhes)
    textos_base = {}
    if combinacao.base:
        # numa simulação (resultados/simulacao/) o texto do Kraken vem da pasta de cima
        arquivo_base = next((p / f'{combinacao.base}.json' for p in (pasta, pasta.parent)
                             if (p / f'{combinacao.base}.json').exists()), None)
        if arquivo_base is None:
            raise FileNotFoundError(f'Rode {combinacao.base} antes de {combinacao.nome}.')
        textos_base = textos_por_arquivo(carregar(arquivo_base))
    feitas = set(textos_por_arquivo(resultado))
    for pagina in paginas:
        if pagina.arquivo in feitas:
            logger.info('%s: página %d já transcrita', combinacao.nome, pagina.numero)
            continue
        logger.info('%s: página %d', combinacao.nome, pagina.numero)
        imagem = carregar_entrada(pagina, combinacao.entrada, combinacao.escala)
        if combinacao.motor == 'kraken':
            texto, custo = motor.transcrever(imagem), None
        elif combinacao.motor == 'llm':
            texto, custo = motor.transcrever(imagem)
        else:
            if pagina.arquivo not in textos_base:
                raise ValueError(f'{combinacao.base} não tem a página {pagina.numero}.')
            texto, custo = motor.corrigir(imagem, textos_base[pagina.arquivo])
        registrar_pagina(resultado, pagina.arquivo, texto, custo)
        resultado['imagens'].sort(key=lambda item: numero_pagina(item['arquivo']))
        salvar(resultado, destino)
    return destino


def avaliar_pasta(pasta: Path = PASTA_RESULTADOS, paginas: list[Pagina] | None = None,
                  nomes: list[str] | None = None) -> list[dict[str, Any]]:
    """Avalia todos os JSON da pasta nas mesmas páginas e grava metricas_paginas.csv e comparacao.csv."""
    paginas = paginas or listar_paginas()
    arquivos = sorted(pasta.glob('*.json')) if nomes is None else [pasta / f'{n}.json' for n in nomes]
    todas, resumo = [], []
    for arquivo in arquivos:
        linhas, corpus = avaliar(carregar(arquivo), paginas)
        todas.extend(linhas)
        resumo.append(corpus)
    resumo.sort(key=lambda linha: linha['wer_sem_pontuacao'])
    salvar_csv(todas, pasta / 'metricas_paginas.csv')
    salvar_csv(resumo, pasta / 'comparacao.csv')
    return resumo


def imprimir_tabela(resumo: list[dict[str, Any]]) -> None:
    print(f"{'combinação':36} {'WER estrito':>11} {'WER s/pont':>10} {'CER s/pont':>10} {'PER':>6} {'custo':>7}")
    for linha in resumo:
        print(f"{linha['combinacao']:36} {linha['wer_estrito']:11.2%} {linha['wer_sem_pontuacao']:10.2%} "
              f"{linha['cer_sem_pontuacao']:10.2%} {linha['per']:6.2%} {linha['custo'] or 0:7.3f}"
              + ('' if linha['status'] == 'ok' else f"  ({linha['status']})"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='comando', required=True)
    rodar = sub.add_parser('rodar', help='executa combinações')
    rodar.add_argument('combinacoes', nargs='*', default=list(PADRAO_KRAKEN))
    rodar.add_argument('--paginas', nargs='*', type=int)
    rodar.add_argument('--pasta', type=Path, default=PASTA_RESULTADOS)
    rodar.add_argument('--modelo-kraken', type=Path, help='padrão: baixa o PP-OCRv6 medium')
    rodar.add_argument('--dispositivo', default=None, help='cpu, cuda:0... (padrão: automático)')
    rodar.add_argument('--modelo-llm', default=None)
    rodar.add_argument('--permitir-gasto', action='store_true', help='autoriza requisições pagas ao OpenRouter')
    rodar.add_argument('--simular', action='store_true', help='monta as requisições de LLM sem enviar')
    rodar.add_argument('--refazer', action='store_true')
    aval = sub.add_parser('avaliar', help='gera comparacao.csv com todas as combinações da pasta')
    aval.add_argument('--paginas', nargs='*', type=int)
    aval.add_argument('--pasta', type=Path, default=PASTA_RESULTADOS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.getLogger('kraken').setLevel(logging.WARNING)
    paginas = listar_paginas(args.paginas or None)

    if args.comando == 'avaliar':
        imprimir_tabela(avaliar_pasta(args.pasta, paginas))
        return

    combinacoes = [interpretar(nome) for nome in args.combinacoes]
    motor_kraken = motor_llm = None
    if any(c.motor == 'kraken' for c in combinacoes):
        from .motor_kraken import MotorKraken, baixar_modelo

        dispositivo = args.dispositivo
        if dispositivo is None:
            import torch
            dispositivo = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        motor_kraken = MotorKraken(args.modelo_kraken or baixar_modelo('medium', RAIZ / 'models'), dispositivo)
    if any(c.gasta_dinheiro for c in combinacoes):
        from .motor_llm import MODELO_PADRAO, MotorLLM

        motor_llm = MotorLLM(args.modelo_llm or MODELO_PADRAO, permitir_gasto=args.permitir_gasto,
                             simular=args.simular)
    for combinacao in combinacoes:
        pasta = args.pasta / 'simulacao' if args.simular and combinacao.gasta_dinheiro else args.pasta
        print(executar(combinacao, paginas, pasta, motor_kraken, motor_llm, args.refazer))


if __name__ == '__main__':
    main()
