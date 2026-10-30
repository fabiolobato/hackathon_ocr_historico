# OCR do Livro de Registro de Leis nº 1 (1892–1894)

Esta pasta tem tudo o que é preciso para rodar o pipeline de OCR final e apresentar os resultados nas 12 páginas transcritas (páginas 5 a 16):

- código;
- configuração final;
- imagens e GT;
- dois cadernos: apresentação e imagens novas;
- reconhecedor, dicionário e modelo de contexto.

`setup_pipeline.py` instala as dependências e confere se os arquivos de modelo estão presentes.

O processo completo, com experimentos, decisões e números, está em [`PROCESSO.md`](PROCESSO.md).

## Resultado

WER normalizado (minúsculas, sem pontuação; quanto menor, melhor):

| configuração | ajuste (9) | **reservadas (7, 10, 12)** | todas (12) |
|---|---:|---:|---:|
| modelo como publicado, imagem crua, sem pós-processamento | 63,15 | 64,57 | 63,52 |
| primeira entrega (v1) | 46,95 | 48,66 | 47,40 |
| mesmo pipeline final com o modelo publicado (medium) | 29,63 | 28,50 | 29,33 |
| **final** (`configs/final.json`) | 24,77 | **22,05** | 24,05 |

- CER normalizado nas reservadas: 21,71 → **8,16**.
- A meta era WER ≤ 50%.

## Sem vazamento de dados

- **Nenhuma das 12 páginas entrou no treino de pesos.**
  - O reconhecedor final (st_v3) é o PP-OCRv6 medium ajustado com as **outras 91 páginas do mesmo livro**.
  - Os rótulos dessas páginas foram gerados pelo próprio pipeline, sem nenhuma transcrição humana.
- **Dicionário e contexto** vêm de textos externos da época:
  - livros do Projeto Gutenberg de autores nascidos entre 1790 e 1875;
  - leis de São Paulo de 1889 a 1910, sem os atos que citam São Carlos.

  Nenhuma transcrição deste livro entra neles.
- **Todos os parâmetros foram escolhidos só nas 9 páginas de ajuste**, inclusive o filtro de confiança do auto-treino.
- **As páginas 7, 10 e 12 não participaram de nenhuma escolha** e foram avaliadas uma única vez, no fim. O número delas é a estimativa honesta.

## Conteúdo

```
apresentacao.ipynb    caderno: cada etapa sobre a original, imagens ↔ texto, WER contra o GT, página por página
novas_imagens.ipynb   caderno: a configuração final num lote de imagens novas (.txt por imagem, WER se houver referência)
PROCESSO.md           documentação completa do processo
setup_pipeline.py     instala as dependências e confere os arquivos de modelo
requirements.txt      dependências fixadas (Python 3.12)
ocr_pipeline/         código do pipeline
configs/final.json    configuração final
imagens/              12 páginas (JPEG)
ground_truth/         transcrição de referência de cada página (.txt), usada só para medir o erro
models/               reconhecedor st_v3, dicionário de época e modelo de contexto
relatorios/           relatório HTML da busca final nas páginas de ajuste
resultados/textos/    texto final de cada página
```

## Como rodar

### Google Colab (gratuito)

Há duas formas de abrir o caderno:

- **Pelo GitHub** (mais simples): abra o caderno no Colab pelo link do README principal do repositório. Com `ORIGEM = 'github'`, a primeira célula clona o repositório e entra nesta pasta.
- **Pelo Drive:** copie esta pasta para `Meu Drive/entrega_ocr` (ajuste o caminho na primeira célula se usar outro), abra `apresentacao.ipynb` com o Google Colaboratory e use `ORIGEM = 'drive'`.

Depois, nos dois casos:

1. A GPU é opcional (Ambiente de execução → Alterar o tipo de ambiente de execução → T4). Em CPU também funciona, só demora mais.
2. Execute tudo (Ambiente de execução → Executar tudo). A primeira célula entra na pasta e roda `setup_pipeline.py`. Esse script instala o Kraken e as bibliotecas de texto sem reinstalar o torch do Colab, e confere o reconhecedor, o dicionário e o modelo de contexto.

O caderno já vem com todas as saídas, então dá para apresentar sem executar. Executar só é preciso para regenerar os resultados. A primeira execução leva de 20 a 60 s por página em CPU, e os intermediários ficam em `cache/`.

### Local (macOS/Linux, Python 3.12)

```bash
cd entrega_ocr
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt jupyter
.venv/bin/jupyter notebook apresentacao.ipynb   # ou novas_imagens.ipynb
```

### Linha de comando, sem caderno

```bash
# todas as páginas; grava texto, métricas e um relatório HTML (imagens ↔ texto)
python -m ocr_pipeline.run --config configs/final.json --pages all --references ground_truth --viz --name final_todas

# só as reservadas (a estimativa honesta)
python -m ocr_pipeline.run --config configs/final.json --pages heldout --references ground_truth --name final_reservadas
```

`--pages` aceita `all`, `tuning` (as 9 de ajuste), `heldout` (7, 10 e 12) ou uma lista (`5,8,13`). Saída em `outputs/runs/<nome>/`: `result.json` (configuração e métricas por página e do corpus), `page_NN.txt` e, com `--viz`, `report.html`. Use `--device cuda:0` se houver GPU.

### Imagens novas

`novas_imagens.ipynb` roda a mesma configuração, sem ajustar nada, num lote de imagens novas. Ele grava um `.txt` por imagem e mostra o processamento completo das primeiras páginas. Se houver transcrições com o mesmo nome da imagem, calcula também WER e CER.

Pela linha de comando:

```bash
python -m ocr_pipeline.batch --config configs/final.json --images <pasta> [--references <pasta>] --out saida
```

Para fotos inteiras do acervo da FPMSC, use `RECORTAR = True`: aplica o recorte fixo que tira a faixa da marca d'água.

## O que o caderno de apresentação mostra

- O que foi testado, o que ficou e quanto cada escolha vale. São tabelas fixas: o caderno **não** refaz a busca nem o treino.
- A configuração final, com cada valor.
- Para cada página:
  - cada operação de imagem aplicada sobre a original, em sequência;
  - a segmentação de linhas desenhada sobre a imagem;
  - o WER depois de cada etapa de texto;
  - o texto final ao lado do GT, com as palavras erradas marcadas;
  - o texto final previsto, sem marcação, ao lado do GT, para ler a saída como ela é;
  - linha a linha, o recorte da imagem ao lado do texto lido.
- No fim, um resumo das 12 páginas. Os textos finais são gravados em `resultados/textos/`.

## Observações

- **Reconhecedor:** `models/finetuned/st_v3.safetensors` (64 MB, SHA-256 `d92f40f5…fb08f501`). Foi treinado a partir do PP-OCRv6 medium (Zenodo 10.5281/zenodo.21788410).
- **Dicionário:** `models/lexicon/pt_hist_legal_w1.txt`. **Modelo de contexto:** `models/lexicon/pt_hist_legal_w4.ngrams.npz`. Os dois foram gerados pelas ferramentas do repositório (`tools/build_period_lexicon.py`, `tools/fetch_alesp_acts.py`).
- Semente 42 e divisão ajuste/reservadas fixas.
- Em CPU, os números das páginas de ajuste diferem até ~0,2 ponto dos obtidos em GPU (24,77 contra 24,60), por arredondamento numérico.
