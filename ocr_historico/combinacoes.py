"""Combinações entrada × motor. O nome de cada combinação é também o nome do JSON em resultados/.

- `kraken_<entrada>[_2x]`: imagem de entrada → Kraken (segmentação neural + PP-OCRv6).
- `kraken_ft_<entrada>[_2x]`: idem, com o modelo fine-tunado na letra do livro (finetuning.py).
- `llm_<entrada>[_2x]`: imagem de entrada → LLM.
- `llm_corrige_<combinação Kraken>`: imagem original + texto do Kraken → LLM corrige.
  Lê o JSON da combinação Kraken já executada; não roda o Kraken de novo.
"""
from dataclasses import dataclass

from .entradas import ENTRADAS

PREFIXO_CORRECAO = 'llm_corrige_'


@dataclass(frozen=True)
class Combinacao:
    nome: str
    motor: str  # 'kraken', 'llm' ou 'correcao'
    entrada: str
    escala: float = 1.0
    base: str | None = None  # combinação Kraken corrigida pelo LLM
    finetunado: bool = False  # Kraken com o modelo do fine-tuning em vez do modelo base

    @property
    def gasta_dinheiro(self) -> bool:
        return self.motor in ('llm', 'correcao')


def interpretar(nome: str) -> Combinacao:
    if nome.startswith(PREFIXO_CORRECAO):
        base = interpretar(nome[len(PREFIXO_CORRECAO):])
        if base.motor != 'kraken':
            raise ValueError(f'{nome}: a correção precisa de uma combinação kraken_*')
        return Combinacao(nome, 'correcao', 'original', base=base.nome)
    motor, _, resto = nome.partition('_')
    finetunado = motor == 'kraken' and resto.startswith('ft_')
    if finetunado:
        resto = resto[3:]
    escala = 1.0
    if resto.endswith('_2x'):
        resto, escala = resto[:-3], 2.0
    if motor not in ('kraken', 'llm') or resto not in ENTRADAS:
        raise ValueError(f'Combinação inválida: {nome}. Exemplos: kraken_binarizada, '
                         f'llm_original, kraken_ft_original_2x, llm_corrige_kraken_ft_original_2x.')
    return Combinacao(nome, motor, resto, escala, finetunado=finetunado)


PADRAO_KRAKEN = ('kraken_original', 'kraken_nlbin', 'kraken_binarizada',
                 'kraken_original_2x', 'kraken_binarizada_2x')
PADRAO_FINETUNADO = ('kraken_ft_original_2x',)
PADRAO_LLM = ('llm_original', 'llm_binarizada', 'llm_corrige_kraken_ft_original_2x')
# sem modelo fine-tunado disponível, a correção usa a melhor combinação do modelo base
CORRECAO_SEM_FINETUNING = 'llm_corrige_kraken_original_2x'
