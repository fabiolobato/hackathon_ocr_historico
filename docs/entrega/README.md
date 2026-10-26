# OCR do Livro de Registro de Leis nº 1 (1892–1894)

Esta pasta tem tudo o que é preciso para rodar o melhor pipeline de OCR encontrado e apresentar os resultados nas 12 páginas transcritas (páginas 5 a 16):

- código
- configurações
- imagens e GT
- pesos dos modelos e léxico
- caderno de apresentação

O processo completo (experimentos, decisões e números) está em [`PROCESSO.md`](PROCESSO.md).

## Resultado

WER normalizado (minúsculas, sem pontuação; quanto menor, melhor):

| configuração                                                    | páginas de ajuste (9) | páginas reservadas (7, 10, 12) |
| --------------------------------------------------------------- | --------------------: | -----------------------------: |
| imagem crua, sem tratamento                                     |                 63,15 |                              – |
| **v4**: busca gulosa, modelo base (`configs/best_v4.json`)      |             **43,17** |                      **43,94** |
| **v3**: reconhecedor ajustado ft_v1 (`configs/best_v3_ft.json`) |       (treinou nelas) |                      **29,61** |

A meta era WER ≤ 50%. A **v4** é o melhor resultado sem treinar pesos e vale para as 12 páginas. A **v3** é o melhor resultado de todos, mas o ft_v1 foi treinado com as 9 páginas de ajuste, então o número honesto dela é o das reservadas.

## Conteúdo

```
apresentacao.ipynb    caderno: cada etapa sobre a original, imagens ↔ texto, WER contra o GT, página por página
PROCESSO.md           documentação completa do processo
setup_pipeline.py     instala as dependências e confere o hash de modelos e léxico
requirements.txt      dependências fixadas (Python 3.12)
ocr_pipeline/         código do pipeline (só o necessário para rodar, sem o código de busca)
configs/              best_v4.json, best_v3_ft.json (finais); best_v1.json, best_v2.json (históricas)
imagens/              12 páginas (JPEG)
ground_truth/         transcrição de referência de cada página (.txt)
models/               PP-OCRv6 medium, ft_v1 (ajustado) e léxico pt-BR 50k
relatorios/           relatórios HTML já gerados: a busca gulosa e a comparação nas reservadas
```

## Como rodar

### Google Colab (gratuito)

1. Copie esta pasta inteira para o Google Drive, em `Meu Drive/entrega_ocr`. Se usar outro nome ou lugar, ajuste o caminho na primeira célula do caderno.
2. Abra `apresentacao.ipynb` no Colab.
3. A GPU é opcional (Ambiente de execução → Alterar o tipo de ambiente de execução → T4). Em CPU também funciona, só demora mais.
4. Execute tudo (Ambiente de execução → Executar tudo). A primeira célula monta o Drive, entra na pasta e roda `setup_pipeline.py`. Esse script instala o Kraken e as bibliotecas de texto sem reinstalar o torch do Colab, e confere o hash dos pesos.

A primeira execução leva de 20 a 60 s por página em CPU. Os resultados intermediários ficam em `cache/`, e as execuções seguintes são imediatas.

### Local (macOS/Linux, Python 3.12)

```bash
cd entrega_ocr
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt jupyter
.venv/bin/python setup_pipeline.py --skip-install   # só confere os hashes dos arquivos
.venv/bin/jupyter notebook apresentacao.ipynb
```

### Linha de comando, sem caderno

```bash
# todas as páginas com a v4; grava texto, métricas e um relatório HTML (imagens ↔ texto)
python -m ocr_pipeline.run --config configs/best_v4.json --pages all --references ground_truth --viz --name v4_todas

# v3 (ft_v1) só nas reservadas, onde a avaliação é honesta
python -m ocr_pipeline.run --config configs/best_v3_ft.json --pages heldout --references ground_truth --viz --name v3_reservadas

# relatório visual comparando configurações
python -m ocr_pipeline.visualize --config configs/best_v4.json --config configs/best_v3_ft.json \
    --pages heldout --references ground_truth --out outputs/comparacao.html
```

`--pages` aceita `all`, `tuning` (as 9 de ajuste), `heldout` (7, 10 e 12) ou uma lista (`5,8,13`). Saída em `outputs/runs/<nome>/`: `result.json` (configuração e métricas por página e do corpus), `page_NN.txt` e, com `--viz`, `report.html`. Use `--device cuda:0` se houver GPU.

## O que o caderno mostra

- O que foi testado, o que ficou e quanto cada escolha vale. São tabelas fixas: o caderno **não** refaz a busca.
- As configurações finais, com cada valor.
- Para cada página e cada configuração:
  - cada operação de imagem aplicada sobre a original, em sequência;
  - a segmentação de linhas desenhada sobre a imagem;
  - o WER depois de cada etapa de texto;
  - o texto final ao lado do GT, com as palavras erradas marcadas;
  - linha a linha, o recorte da imagem ao lado do texto lido.
- No fim, um resumo das 12 páginas. Os textos finais são gravados em `resultados/textos/`.

## Observações

- **Pesos do ft_v1** (`models/finetuned/ft_v1.safetensors`, SHA-256 `986c7670…fea72f`): não são públicos e vêm só nesta pasta. Sem eles, o caderno mostra apenas a v4.
- **Modelo base e léxico:** PP-OCRv6 medium (Zenodo 10.5281/zenodo.21788410) e FrequencyWords pt_BR 50k. Se faltarem, `setup_pipeline.py` baixa os dois.
- **Léxico do domínio:** é montado com o GT das páginas de ajuste e **nunca** inclui a página avaliada. Por isso o WER da v4 vale para todas as páginas.
- **Fixos:** semente 42 e divisão ajuste/reservadas.
- **Fora desta pasta:** o código de busca e de ajuste fino (`ocr_pipeline/tuning/`) e a reprodução dos experimentos ficam no repositório (ver `PROCESSO.md`, seção Reprodução).
