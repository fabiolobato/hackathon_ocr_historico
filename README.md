# OCR histórico: Livro de Registro de Leis nº 1 (1892–1894)

Pipeline de OCR para manuscritos do século XIX (Kraken + PP-OCRv6), avaliado em 12 páginas transcritas do _Livro de Registro de Leis nº 1_.

## Resumo visual: comece por aqui

**[`docs/entrega/apresentacao.ipynb`](docs/entrega/apresentacao.ipynb)**: o caderno de apresentação, já executado, com todas as saídas. É a forma mais fácil de ver o pipeline funcionando.

- [Ver no nbviewer](https://nbviewer.org/github/fabiolobato/hackathon_ocr_historico/blob/feat/visual_report_greedy_search/docs/entrega/apresentacao.ipynb): mostra as imagens e tabelas completas.
- [Abrir no Colab](https://colab.research.google.com/github/fabiolobato/hackathon_ocr_historico/blob/feat/visual_report_greedy_search/docs/entrega/apresentacao.ipynb): mostra as saídas salvas; "Executar tudo" clona o repositório e roda de novo.

Para cada uma das 12 páginas, o caderno mostra:

- cada operação de imagem aplicada sobre a original e a segmentação de linhas desenhada na imagem;
- o WER depois de cada etapa de texto;
- o texto final ao lado do GT, com as palavras erradas marcadas;
- linha a linha, o recorte da imagem ao lado do texto lido.

Ele também resume o que foi testado, quanto cada escolha vale e o que cada etapa faz. A documentação completa dos experimentos está em [`docs/entrega/PROCESSO.md`](docs/entrega/PROCESSO.md).

## Resultado

WER normalizado (minúsculas, sem pontuação; quanto menor, melhor):

| configuração                                           | ajuste (9) | **reservadas (7, 10, 12)** | todas (12) |
| ------------------------------------------------------ | ---------: | -------------------------: | ---------: |
| modelo como publicado, imagem crua                     |      63,15 |                      64,57 |      63,52 |
| **final** ([`configs/final.json`](configs/final.json)) |      46,95 |                  **48,66** |      47,40 |

As páginas foram divididas como treino e teste, com semente 42:

- **ajuste** (5, 6, 8, 9, 11, 13, 14, 15, 16): usadas para escolher os parâmetros;
- **reservadas** (7, 10, 12): não participam de nenhuma escolha e dão a estimativa honesta.

Uma única configuração é aplicada igualmente a todas as páginas.

**Sem vazamento de dados:**

- o reconhecedor é usado como foi publicado, sem treino com estas páginas;
- o léxico é genérico (pt-BR);
- nenhuma transcrição deste livro entra no pipeline. O GT só é usado para medir o erro.

## O pipeline, etapa por etapa

```
original ─► geometria ─► base
             base ─► imagem de segmentação ─► segmentador blla ─► linhas (polígonos)
             base ─► imagem de reconhecimento ─► PP-OCRv6, linha a linha ─► texto
texto ─► reescore CTC ─► convenções ─► hifenização ─► correção léxica ─► texto final
```

A segmentação e o reconhecimento usam **imagens diferentes** da mesma base. As operações de cor e contraste não movem pixels, então as linhas achadas numa imagem valem na outra. Cada leitor recebe a imagem que funciona melhor para ele.

### 1. Imagem

| etapa                      | o que faz                                                                        | valor final               |
| -------------------------- | -------------------------------------------------------------------------------- | ------------------------- |
| ampliação                  | aumenta a página; o traço fino ganha pixels                                      | ×1,5                      |
| correção de inclinação     | gira a página até as linhas ficarem horizontais                                  | ±5°, passo 0,25°          |
| normalização de iluminação | divide a imagem pelo próprio fundo desfocado, tirando manchas e sombras do papel | σ = 25 (nas duas imagens) |
| CLAHE                      | realça o contraste local, só na imagem de segmentação                            | clip 2,0, grade 8         |
| gama                       | clareia os tons médios, só na imagem de reconhecimento                           | 0,7                       |

### 2. Segmentação de linhas

O segmentador neural **blla** (padrão do Kraken) encontra cada linha de texto como um polígono com uma linha de base, em ordem de leitura.

### 3. Reconhecimento

O **PP-OCRv6 medium** ([Zenodo 10.5281/zenodo.21788410](https://doi.org/10.5281/zenodo.21788410)) lê cada linha e dá, para cada fatia da imagem, a probabilidade de cada caractere. A leitura **gulosa** pega sempre o caractere mais provável.

### 4. Texto

| etapa           | o que faz                                                                                                                                      | valor final           |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| reescore CTC    | para cada palavra fora do dicionário pt-BR, busca palavras parecidas no dicionário e pergunta ao próprio modelo qual combina mais com a imagem | α = 2, 10 candidatos  |
| convenções      | regras fixas de notação: `n^o → nº`, `⁋ → §`                                                                                                   | –                     |
| hifenização     | se a linha termina em `-`, junta com a primeira palavra da seguinte                                                                            | –                     |
| correção léxica | troca palavras fora do dicionário pela mais próxima, se a diferença for pequena                                                                | razão 0,5, ≥ 3 letras |

O dicionário é o FrequencyWords pt_BR 50k (OpenSubtitles 2018).

### Como os parâmetros foram escolhidos

Uma **busca gulosa condicional** ([`ocr_pipeline/tuning/search.py`](ocr_pipeline/tuning/search.py), só nas páginas de ajuste) decide as etapas em ordem, sempre sobre o melhor conjunto atual:

1. Cada candidato da etapa é testado sozinho. O melhor só entra se superar a referência.
2. Todos os outros candidatos são testados por cima do vencedor, inclusive os que pioraram sozinhos. Cada um fica só se melhorar.
3. Um passe para trás remove qualquer escolha que não ajude.
4. Os passes se repetem até nada mudar.

Foram 3 passes e 330 configurações. O caderno mostra o caminho da busca e quanto cada escolha vale.

## Como rodar

### Só ver os resultados

Abra [`docs/entrega/apresentacao.ipynb`](docs/entrega/apresentacao.ipynb) por um dos links acima.

### Google Colab (gratuito)

Abra o caderno pelo link do Colab e use "Executar tudo". A primeira célula (com `ORIGEM = 'github'`):

1. clona este repositório;
2. instala o Kraken sem reinstalar o torch do Colab;
3. baixa e confere o modelo e o léxico.

A GPU é opcional.

### Pasta de entrega `entrega_ocr/` (local ou Drive)

A pasta `entrega_ocr/` **não fica no Git**. Ela é gerada automaticamente a partir deste repositório e reúne só o necessário para rodar e apresentar:

- caderno executado;
- código do pipeline e `configs/final.json`;
- imagens e GT;
- modelo e léxico;
- README e PROCESSO.

O caderno gerado é o mesmo de [`docs/entrega/apresentacao.ipynb`](docs/entrega/apresentacao.ipynb).

```bash
git clone -b feat/visual_report_greedy_search https://github.com/fabiolobato/hackathon_ocr_historico.git
cd hackathon_ocr_historico
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python docs/entrega/setup_pipeline.py --skip-install       # baixa e confere o modelo e o léxico em models/
.venv/bin/python tools/build_entrega.py --out entrega_ocr --execute  # gera a pasta e executa o caderno (~5–10 min em CPU)
```

Depois, há dois caminhos:

- **Local:** `cd entrega_ocr && ../.venv/bin/jupyter notebook apresentacao.ipynb`.
- **Drive:** copie `entrega_ocr/` para `Meu Drive/entrega_ocr`, abra `apresentacao.ipynb` com o Google Colaboratory, mude a primeira célula para `ORIGEM = 'drive'` e execute tudo.

`--execute` também grava o caderno executado em `docs/entrega/apresentacao.ipynb`. Sem `--execute`, a pasta é só montada. Detalhes da pasta em [`docs/entrega/README.md`](docs/entrega/README.md), que vira o `README.md` dela.

### Linha de comando

```bash
python -m ocr_pipeline.run --config configs/final.json --pages all --viz --name final_todas
```

`--pages` aceita `all`, `tuning`, `heldout` ou uma lista (`5,8,13`). Saída em `outputs/runs/<nome>/`: métricas, texto por página e, com `--viz`, um relatório HTML imagem ↔ texto.

## Estrutura do repositório

```
docs/entrega/           caderno de apresentação (executado), PROCESSO.md, README e setup da pasta de entrega
ocr_pipeline/           código do pipeline: imagem, segmentação, reconhecimento, pós-processamento, métricas, relatórios
ocr_pipeline/tuning/    busca de parâmetros e utilitários de experimentos
configs/                configurações (final.json é a final)
sweeps/                 especificações das buscas
tools/build_entrega.py  gera entrega_ocr/ e executa o caderno
imagens/                as 12 páginas
test_kraken/            experimentos anteriores e transcrições de referência (ground_truth/)
```

Os pesos e o léxico ficam fora do Git, em `models/`. `docs/entrega/setup_pipeline.py` baixa os dois e confere o hash.
