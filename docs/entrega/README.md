# OCR do Livro de Registro de Leis nº 1 (1892–1894)

Esta pasta tem tudo o que é preciso para rodar o pipeline de OCR final e apresentar os resultados nas 12 páginas transcritas (páginas 5 a 16):

- código
- configuração
- imagens e GT
- modelo e léxico
- caderno de apresentação

O processo completo (experimentos, decisões e números) está em [`PROCESSO.md`](PROCESSO.md).

## Resultado

WER normalizado (minúsculas, sem pontuação; quanto menor, melhor):

| configuração                                              | ajuste (9) | **reservadas (7, 10, 12)** | todas (12) |
| --------------------------------------------------------- | ---------: | -------------------------: | ---------: |
| modelo como publicado, imagem crua, sem pós-processamento |      63,15 |                      64,57 |      63,52 |
| **final** (`configs/final.json`)                          |      46,95 |                  **48,66** |      47,40 |

A meta era WER ≤ 50%.

**Sem vazamento de dados:**

- O reconhecedor (PP-OCRv6 medium) é usado como foi publicado.
- O léxico é genérico (pt-BR), e nenhuma transcrição deste livro entra no pipeline.
- Só parâmetros (limiares, operações de imagem, pesos) foram escolhidos, e só nas 9 páginas de ajuste.
- As páginas 7, 10 e 12 não participaram de nenhuma escolha e dão a estimativa honesta.

## Conteúdo

```
apresentacao.ipynb    caderno: cada etapa sobre a original, imagens ↔ texto, WER contra o GT, página por página
PROCESSO.md           documentação completa do processo
setup_pipeline.py     instala as dependências e confere o hash do modelo e do léxico
requirements.txt      dependências fixadas (Python 3.12)
ocr_pipeline/         código do pipeline
configs/final.json    configuração final
imagens/              12 páginas (JPEG)
ground_truth/         transcrição de referência de cada página (.txt), usada só para medir o erro
models/               PP-OCRv6 medium e léxico pt-BR 50k
relatorios/           relatório HTML da busca de parâmetros nas páginas de ajuste
```

## Como rodar

### Google Colab (gratuito)

1. Copie esta pasta inteira para o Google Drive, em `Meu Drive/entrega_ocr`. Se usar outro nome ou lugar, ajuste o caminho na primeira célula do caderno.
2. No Drive, clique com o botão direito em `apresentacao.ipynb` → Abrir com → Google Colaboratory.
3. A GPU é opcional (Ambiente de execução → Alterar o tipo de ambiente de execução → T4). Em CPU também funciona, só demora mais.
4. Execute tudo (Ambiente de execução → Executar tudo). A primeira célula monta o Drive, entra na pasta e roda `setup_pipeline.py`. Esse script instala o Kraken e as bibliotecas de texto sem reinstalar o torch do Colab, e confere o hash do modelo e do léxico.

O caderno já vem com todas as saídas, então dá para apresentar sem executar. Executar só é preciso para regenerar os resultados. A primeira execução leva de 20 a 60 s por página em CPU, e os intermediários ficam em `cache/`.

### Local (macOS/Linux, Python 3.12)

```bash
cd entrega_ocr
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt jupyter
.venv/bin/python setup_pipeline.py --skip-install   # só confere os hashes
.venv/bin/jupyter notebook apresentacao.ipynb
```

### Linha de comando, sem caderno

```bash
# todas as páginas; grava texto, métricas e um relatório HTML (imagens ↔ texto)
python -m ocr_pipeline.run --config configs/final.json --pages all --references ground_truth --viz --name final_todas

# só as reservadas (a estimativa honesta)
python -m ocr_pipeline.run --config configs/final.json --pages heldout --references ground_truth --name final_reservadas
```

`--pages` aceita `all`, `tuning` (as 9 de ajuste), `heldout` (7, 10 e 12) ou uma lista (`5,8,13`). Saída em `outputs/runs/<nome>/`: `result.json` (configuração e métricas por página e do corpus), `page_NN.txt` e, com `--viz`, `report.html`. Use `--device cuda:0` se houver GPU.

## O que o caderno mostra

- O que foi testado, o que ficou e quanto cada escolha vale. São tabelas fixas: o caderno **não** refaz a busca.
- A configuração final, com cada valor.
- Para cada página:
  - cada operação de imagem aplicada sobre a original, em sequência;
  - a segmentação de linhas desenhada sobre a imagem;
  - o WER depois de cada etapa de texto;
  - o texto final ao lado do GT, com as palavras erradas marcadas;
  - linha a linha, o recorte da imagem ao lado do texto lido.
- No fim, um resumo das 12 páginas. Os textos finais são gravados em `resultados/textos/`.

## Observações

- **Modelo:** PP-OCRv6 medium (Zenodo 10.5281/zenodo.21788410, MD5 `e3411a45…`).
- **Léxico:** FrequencyWords pt_BR 50k (SHA-256 `a61d6f2e…`).
- Se o modelo ou o léxico faltarem, `setup_pipeline.py` baixa os dois.
- Semente 42 e divisão ajuste/reservadas fixas.
