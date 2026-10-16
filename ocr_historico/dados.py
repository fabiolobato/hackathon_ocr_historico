"""Páginas do livro: imagem original, binarização do time e gabarito, ligados pelo número."""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_IMAGENS = RAIZ / 'imagens'
PASTA_BINARIZADAS = RAIZ / 'imagens_preprocessadas'
PASTA_GABARITOS = RAIZ / 'ground_truth'

_NUMERO = re.compile(r'^(?:binaria_)?(\d+)\.')


def normalizar_nome(nome: str) -> str:
    # Os nomes têm "n°"; zip, Drive e macOS podem trocar a forma Unicode.
    return unicodedata.normalize('NFC', Path(nome).name)


def numero_pagina(nome: str) -> int:
    encontrado = _NUMERO.match(normalizar_nome(nome))
    if not encontrado:
        raise ValueError(f'Nome sem número de página: {nome}')
    return int(encontrado[1])


@dataclass(frozen=True)
class Pagina:
    numero: int
    imagem: Path
    binarizada: Path | None
    gabarito: Path | None

    @property
    def arquivo(self) -> str:
        return normalizar_nome(self.imagem.name)

    def texto_gabarito(self) -> str:
        if self.gabarito is None:
            raise FileNotFoundError(f'Página {self.numero} sem gabarito em ground_truth/')
        return self.gabarito.read_text(encoding='utf-8')


def _por_numero(pasta: Path, sufixo: str) -> dict[int, Path]:
    arquivos = {}
    for caminho in pasta.glob(f'*{sufixo}') if pasta.is_dir() else ():
        if _NUMERO.match(normalizar_nome(caminho.name)):
            arquivos[numero_pagina(caminho.name)] = caminho
    return arquivos


def listar_paginas(numeros: list[int] | None = None, imagens: Path = PASTA_IMAGENS,
                   binarizadas: Path = PASTA_BINARIZADAS,
                   gabaritos: Path = PASTA_GABARITOS) -> list[Pagina]:
    originais = _por_numero(imagens, '.jpg')
    if not originais:
        raise FileNotFoundError(f'Nenhuma imagem .jpg em {imagens}')
    binarias = _por_numero(binarizadas, '.jpg')
    textos = _por_numero(gabaritos, '.txt')
    escolhidos = sorted(originais) if numeros is None else numeros
    faltando = [n for n in escolhidos if n not in originais]
    if faltando:
        raise ValueError(f'Páginas inexistentes: {faltando}. Disponíveis: {sorted(originais)}')
    return [Pagina(n, originais[n], binarias.get(n), textos.get(n)) for n in escolhidos]
