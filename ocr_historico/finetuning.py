"""Fine-tuning do Kraken (PP-OCRv6) na letra do escrivão deste livro.

1. `gerar_dataset`: segmenta e reconhece as páginas de treino com a mesma entrada usada
   na inferência, corta o gabarito em linhas e grava pares `.png` + `.gt.txt`.
2. `treinar`: chama `ketos train` a partir do modelo base e devolve `best_*.safetensors`.
3. Avaliação: o modelo treinado vira um `MotorKraken` comum; compare nas páginas reservadas,
   que nunca entram no treino.

    python -m ocr_historico.finetuning --dispositivo cuda:0

Precisa de GPU: em CPU o treino do medium estoura a memória ou leva horas por época.
O modelo aprende a grafia do gabarito: revise `ground_truth/` antes (ver comparison_review.md).
"""
import argparse
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .alinhamento import distancia, dividir_gabarito
from .dados import RAIZ, Pagina, listar_paginas
from .entradas import carregar_entrada

logger = logging.getLogger(__name__)
PASTA_FINETUNING = RAIZ / 'finetuning'


def dividir_paginas(paginas: list[Pagina], reservadas: int = 3, semente: int = 42) -> tuple[list[Pagina], list[Pagina]]:
    """Mesmo sorteio do test_kraken (split_pages): treino e páginas reservadas para avaliação."""
    from . import motor_kraken  # noqa: F401 - põe test_kraken/ no sys.path
    from kraken_pipeline.tuning import split_pages

    por_imagem = {p.imagem: p for p in paginas}
    treino, reserva = split_pages([p.imagem for p in paginas], reservadas, semente)
    return ([por_imagem[i] for i in sorted(treino, key=lambda i: por_imagem[i].numero)],
            [por_imagem[i] for i in sorted(reserva, key=lambda i: por_imagem[i].numero)])


def gerar_dataset(motor: Any, paginas: list[Pagina], pasta: Path, entrada: str = 'original',
                  escala: float = 2.0, distancia_maxima: float = 0.5) -> dict[str, Any]:
    """Grava `p<página>_<linha>.png` + `.gt.txt` e `manifesto.json` (linhas usadas e descartadas).

    Linhas cujo OCR difere demais do trecho do gabarito (rubricas, números de página,
    erros de segmentação) são descartadas e listadas no manifesto para conferência.
    """
    from kraken.lib.segmentation import extract_polygons

    pasta.mkdir(parents=True, exist_ok=True)
    manifesto: dict[str, Any] = {'modelo_base': motor.nome, 'entrada': entrada, 'escala': escala,
                                 'distancia_maxima': distancia_maxima, 'paginas': [p.numero for p in paginas],
                                 'usadas': [], 'descartadas': []}
    for pagina in paginas:
        imagem = carregar_entrada(pagina, entrada, escala)
        segmentacao = motor.engine.segment(imagem)
        linhas_ocr = [linha.prediction for linha in motor.engine.recognize(imagem, segmentacao)]
        linhas_gabarito = dividir_gabarito(linhas_ocr, pagina.texto_gabarito())
        recortes = extract_polygons(imagem, segmentacao)
        for indice, ((recorte, _), ocr, texto) in enumerate(zip(recortes, linhas_ocr, linhas_gabarito)):
            item = {'pagina': pagina.numero, 'linha': indice, 'ocr': ocr, 'gabarito': texto}
            d = distancia(ocr.casefold(), texto.casefold())
            if not texto or d > distancia_maxima:
                manifesto['descartadas'].append(item)
                continue
            nome = pasta / f'p{pagina.numero:02d}_{indice:03d}'
            recorte.save(nome.with_suffix('.png'))
            nome.with_suffix('.gt.txt').write_text(texto, encoding='utf-8')
            manifesto['usadas'].append({**item, 'arquivo': nome.with_suffix('.png').name, 'distancia': round(d, 3)})
        logger.info('Página %d: %d linhas usadas até agora', pagina.numero, len(manifesto['usadas']))
    (pasta / 'manifesto.json').write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifesto


def dividir_linhas(pasta: Path, fracao_validacao: float = 0.1, semente: int = 42) -> tuple[Path, Path]:
    """Listas `treino.txt` e `validacao.txt` para o ketos (validação só para escolher a época)."""
    linhas = sorted(pasta.glob('*.png'))
    if len(linhas) < 10:
        raise ValueError(f'Poucas linhas em {pasta} ({len(linhas)}). Rode gerar_dataset primeiro.')
    random.Random(semente).shuffle(linhas)
    corte = max(1, round(len(linhas) * fracao_validacao))
    arquivos = {'validacao.txt': linhas[:corte], 'treino.txt': linhas[corte:]}
    for nome, lista in arquivos.items():
        (pasta / nome).write_text(''.join(f'{p.resolve()}\n' for p in sorted(lista)), encoding='utf-8')
    return pasta / 'treino.txt', pasta / 'validacao.txt'


def largura_necessaria(pasta: Path, altura: int = 96, margem: int = 32, multiplo: int = 64) -> int:
    """Menor --max-width que comporta todas as linhas.

    O ppocrv6 redimensiona cada linha para `altura`, soma a margem lateral do ketos (--pad 16
    de cada lado) e completa todas até --max-width: a memória da GPU cresce com esse valor.
    """
    from PIL import Image

    maior = 0.0
    for linha in pasta.glob('*.png'):
        with Image.open(linha) as imagem:
            maior = max(maior, imagem.width * altura / imagem.height)
    if not maior:
        raise ValueError(f'Nenhuma linha .png em {pasta}.')
    return -(-int(maior + margem) // multiplo) * multiplo


def comando_treino(modelo_base: Path, treino: Path, validacao: Path, saida: Path, dispositivo: str,
                   largura_maxima: int, epocas: int = 30, lote: int = 4, acumular: int = 4,
                   taxa: float = 1e-4, aumentar: bool = True, semente: int = 42) -> list[str]:
    # lote 4 × acumulação 4 = lote efetivo 16 com ~1/4 da memória (lote 16 estourou uma T4 de 15 GB)
    # ppocrv6 usa schedule cosine, que exige número fixo de épocas (-q fixed)
    return [sys.executable, '-c', 'from kraken.ketos import cli; cli()',
            '-d', dispositivo, '-s', str(semente), '--workers', '2', '--threads', str(os.cpu_count() or 1),
            'train', '-f', 'path', '-i', str(modelo_base), '--resize', 'union',
            '-t', str(treino), '-e', str(validacao), '-o', str(saida),
            '-q', 'fixed', '-N', str(epocas), '--cos-max', str(epocas),
            '-B', str(lote), '--accumulate-grad-batches', str(acumular), '-r', str(taxa),
            '--max-width', str(largura_maxima), '--augment' if aumentar else '--no-augment']


def melhor_modelo(saida: Path) -> Path:
    """O ketos converte o melhor checkpoint (maior acurácia de validação) em best_<score>.safetensors."""
    modelos = sorted(saida.glob('best_*.safetensors'), key=lambda p: float(p.stem.split('_', 1)[1]))
    if not modelos:
        raise FileNotFoundError(f'Nenhum best_*.safetensors em {saida}. Veja o log do treino.')
    return modelos[-1]


def treinar(modelo_base: Path, pasta_linhas: Path, saida: Path, dispositivo: str, **opcoes: Any) -> Path:
    """Treina numa subpasta nova de `saida` (uma por execução) e devolve o melhor modelo dela."""
    treino, validacao = dividir_linhas(pasta_linhas)
    opcoes.setdefault('largura_maxima', largura_necessaria(pasta_linhas))
    # pasta nova por execução: checkpoints de um treino anterior não podem virar o "melhor modelo"
    saida = saida / datetime.now().strftime('%Y%m%d-%H%M%S')
    saida.mkdir(parents=True)
    comando = comando_treino(modelo_base, treino, validacao, saida, dispositivo, **opcoes)
    logger.info('Treinando: %s', ' '.join(comando))
    ambiente = {**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
    # repassa a saída do ketos linha a linha: no Colab, a saída de um subprocesso não aparece sozinha
    with subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                          env=ambiente) as processo, (saida / 'treino.log').open('w', encoding='utf-8') as log:
        for linha in processo.stdout:
            print(linha, end='', flush=True)
            log.write(linha)
    if processo.returncode:
        raise RuntimeError(f'ketos train terminou com código {processo.returncode}; veja {saida / "treino.log"}')
    return melhor_modelo(saida)


def liberar_gpu() -> None:
    """Devolve a memória de modelos já descartados (sem referência) antes do treino."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dispositivo', default='cuda:0')
    parser.add_argument('--modelo-base', type=Path, help='padrão: baixa o PP-OCRv6 medium')
    parser.add_argument('--pasta', type=Path, default=PASTA_FINETUNING)
    parser.add_argument('--entrada', default='original')
    parser.add_argument('--escala', type=float, default=2.0)
    parser.add_argument('--epocas', type=int, default=30)
    parser.add_argument('--lote', type=int, default=4)
    parser.add_argument('--acumular', type=int, default=4, help='lote efetivo = lote × acumular')
    parser.add_argument('--taxa', type=float, default=1e-4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.getLogger('kraken').setLevel(logging.WARNING)

    from .executar import avaliar_pasta, executar, imprimir_tabela, PASTA_RESULTADOS
    from .combinacoes import interpretar
    from .motor_kraken import MotorKraken, baixar_modelo

    modelo_base = args.modelo_base or baixar_modelo('medium', RAIZ / 'models')
    treino, reserva = dividir_paginas(listar_paginas())
    logger.info('Treino: %s | reservadas: %s', [p.numero for p in treino], [p.numero for p in reserva])
    gerar_dataset(MotorKraken(modelo_base, args.dispositivo), treino, args.pasta / 'linhas', args.entrada, args.escala)
    liberar_gpu()  # o treino roda em outro processo e precisa da GPU inteira
    modelo = treinar(modelo_base, args.pasta / 'linhas', args.pasta / 'modelo', args.dispositivo,
                     epocas=args.epocas, lote=args.lote, acumular=args.acumular, taxa=args.taxa)
    logger.info('Modelo treinado: %s', modelo)
    base = MotorKraken(modelo_base, args.dispositivo)

    nome = f'kraken_{args.entrada}' + ('_2x' if args.escala == 2 else '')
    combinacao = interpretar(nome)
    executar(combinacao, reserva, args.pasta / 'antes', motor_kraken=base)
    executar(combinacao, reserva, args.pasta / 'depois', motor_kraken=MotorKraken(modelo, args.dispositivo))
    for rotulo in ('antes', 'depois'):
        print(f'\n{rotulo.upper()} do fine-tuning (páginas reservadas {[p.numero for p in reserva]}):')
        imprimir_tabela(avaliar_pasta(args.pasta / rotulo, reserva))


if __name__ == '__main__':
    main()
