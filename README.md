# OCR histórico: Livro de Registro de Leis nº 1 (1892–1894)

Pipeline de OCR para manuscritos do século XIX (Kraken + PP-OCRv6), avaliado em 12 páginas transcritas do _Livro de Registro de Leis nº 1_.

## Resumo visual: comece por aqui

**[`entrega_ocr/apresentacao.ipynb`](entrega_ocr/apresentacao.ipynb)**: o caderno de apresentação, já executado, com todas as saídas. É a forma mais fácil de ver o pipeline funcionando.

- [Ver no nbviewer](https://nbviewer.org/github/fabiolobato/hackathon_ocr_historico/blob/feat/random_search_period_lexicon/entrega_ocr/apresentacao.ipynb): mostra as imagens e tabelas completas.
- [Abrir no Colab](https://colab.research.google.com/github/fabiolobato/hackathon_ocr_historico/blob/feat/random_search_period_lexicon/entrega_ocr/apresentacao.ipynb): mostra as saídas salvas; "Executar tudo" roda de novo, sem instalar nada no seu computador.

Para cada uma das 12 páginas, o caderno mostra:

- cada operação de imagem aplicada sobre a original e a segmentação de linhas desenhada na imagem;
- o WER depois de cada etapa de texto;
- o texto final ao lado do GT, com as palavras erradas marcadas;
- o texto final sem marcação, como sai do pipeline;
- linha a linha, o recorte da imagem ao lado do texto lido.

Ele também resume o que foi testado, quanto cada escolha vale e o que cada etapa faz.

- **Processo completo:** [`entrega_ocr/PROCESSO.md`](entrega_ocr/PROCESSO.md), com experimentos, números e o que não funcionou.
- **Imagens novas:** [`entrega_ocr/novas_imagens.ipynb`](entrega_ocr/novas_imagens.ipynb) roda a configuração final num lote de imagens novas.

## Resultado

WER normalizado (minúsculas, sem pontuação; quanto menor, melhor):

| configuração                                                 | ajuste (9) | **reservadas (7, 10, 12)** | todas (12) |
| ------------------------------------------------------------ | ---------: | -------------------------: | ---------: |
| modelo como publicado, imagem crua                           |      63,15 |                      64,57 |      63,52 |
| primeira entrega (v1)                                        |      46,95 |                      48,66 |      47,40 |
| pipeline completo com o modelo publicado                     |      29,63 |                      28,50 |      29,33 |
| **final** ([`configs/final_v4.json`](configs/final_v4.json)) |      24,77 |                  **22,05** |      24,05 |

CER normalizado nas reservadas: 21,71 → **8,16**.

As páginas foram divididas como treino e teste, com semente 42:

- **ajuste** (5, 6, 8, 9, 11, 13, 14, 15, 16): usadas para escolher os parâmetros;
- **reservadas** (7, 10, 12): não participam de nenhuma escolha. Foram avaliadas uma única vez, no fim, e dão a estimativa honesta.

Uma única configuração é aplicada igualmente a todas as páginas.

**Sem vazamento de dados:**

- Nenhuma das 12 páginas entrou no treino de pesos. O reconhecedor final é o PP-OCRv6 medium **auto-treinado com as outras 91 páginas do mesmo livro**, com rótulos gerados pelo próprio pipeline, sem nenhuma transcrição humana.
- O dicionário e o modelo de contexto vêm de textos externos da época: Projeto Gutenberg e leis de São Paulo de 1889 a 1910, sem os atos que citam São Carlos.
- O GT só é usado para medir o erro e, nas páginas de ajuste, para escolher parâmetros.

## O pipeline, etapa por etapa

```
original ─► ampliação ─► base
             base ─► contraste + remoção da pauta ─► segmentador blla ─► linhas (polígonos)
             base ─► PP-OCRv6 auto-treinado, linha a linha ─► probabilidades por caractere
probabilidades ─► reescore com dicionário de época + contexto ─► convenções, ordinais, hifenização ─► texto final
```

A segmentação e o reconhecimento usam **imagens diferentes** da mesma base. As operações de cor e contraste não movem pixels, então as linhas achadas numa imagem valem na outra. Cada leitor recebe a imagem que funciona melhor para ele.

### 1. Imagem

| etapa            | o que faz                                                         | valor final |
| ---------------- | ----------------------------------------------------------------- | ----------- |
| ampliação        | aumenta a página; o traço fino ganha pixels                       | ×2          |
| contraste        | realça a tinta, só na imagem de segmentação                       | ×1,8        |
| remoção da pauta | apaga as linhas horizontais do papel, só na imagem de segmentação | —           |

O reconhecedor lê a base sem tratamento.

### 2. Segmentação de linhas

O segmentador neural **blla** (padrão do Kraken) encontra cada linha de texto como um polígono com uma linha de base, em ordem de leitura.

### 3. Reconhecimento

O ponto de partida é o **PP-OCRv6 medium** ([Zenodo 10.5281/zenodo.21788410](https://doi.org/10.5281/zenodo.21788410)). Ele lê cada linha e dá, para cada fatia da imagem, a probabilidade de cada caractere.

O modelo final, **st_v3**, é esse medium ajustado em 3 rodadas de **auto-treino** com as 91 páginas do livro que não foram transcritas:

1. o pipeline lê as páginas;
2. as linhas lidas com confiança viram rótulos;
3. o modelo é ajustado com elas;
4. o modelo novo gera rótulos melhores para a rodada seguinte.

O auto-treino reduziu o WER das reservadas de 28,50 para 22,05. A 4ª rodada já não melhorou e ficou de fora.

### 4. Texto

| etapa                 | o que faz                                                                                                                                                                      | valor final           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------- |
| reescore CTC          | para cada palavra, busca palavras parecidas no dicionário de época (também a partir das 5 leituras mais prováveis) e pergunta ao próprio modelo qual combina mais com a imagem | α = 16, 30 candidatas |
| contexto              | escolhe a sequência de candidatas mais provável na linha, com trigramas de palavras do corpus de época                                                                         | β = 1, feixe de 64    |
| convenções e ordinais | regras fixas de notação: `n^o → nº`, `⁋ → §`, `2o → 2º`                                                                                                                        | –                     |
| hifenização           | se a linha termina em `-`, junta com a primeira palavra da seguinte                                                                                                            | –                     |
| correção léxica       | troca palavras fora do dicionário pela mais próxima, se a diferença for pequena                                                                                                | razão 0,5, ≥ 4 letras |

O dicionário de época junta livros em português do Projeto Gutenberg (autores nascidos entre 1790 e 1875) e 3.565 leis e decretos de São Paulo de 1889 a 1910. A grafia é anterior a 1911, como a do livro.

### Como os parâmetros foram escolhidos

Uma **busca gulosa condicional** ([`ocr_pipeline/tuning/search.py`](ocr_pipeline/tuning/search.py)) decide as etapas em ordem, só nas páginas de ajuste e sempre sobre o melhor conjunto atual:

1. Cada candidato da etapa é testado sozinho. O melhor só entra se superar a referência.
2. Todos os outros candidatos são testados por cima do vencedor, inclusive os que pioraram sozinhos. Cada um fica só se melhorar.
3. Um passe para trás remove qualquer escolha que não ajude.
4. Os passes se repetem até nada mudar.

Com o reconhecedor final, a busca convergiu no 3º passe. O caderno mostra quanto cada escolha vale.

## Como rodar

### Só ver os resultados

Abra [`entrega_ocr/apresentacao.ipynb`](entrega_ocr/apresentacao.ipynb) por um dos links acima.

### Google Colab (gratuito, um clique)

Abra o caderno pelo link do Colab e use "Executar tudo". A primeira célula (com `ORIGEM = 'github'`):

1. clona este repositório;
2. entra em `entrega_ocr/`;
3. instala o Kraken sem reinstalar o torch do Colab;
4. confere o reconhecedor, o dicionário e o modelo de contexto.

A GPU é opcional. A primeira execução leva de 20 a 60 s por página em CPU.

### A pasta `entrega_ocr/`

[`entrega_ocr/`](entrega_ocr/) reúne tudo o que é preciso para rodar e apresentar o pipeline:

- os dois cadernos (o de apresentação já executado);
- o código do pipeline e `configs/final.json`;
- imagens e GT;
- o reconhecedor st_v3, o dicionário de época e o modelo de contexto;
- README e PROCESSO.

Ela é gerada automaticamente a partir deste repositório:

- o código vem de `ocr_pipeline/`;
- a configuração vem de `configs/final_v4.json`;
- os arquivos de modelo são os que a configuração cita;
- os cadernos, o README e o PROCESSO vêm de `docs/entrega/`.

Para rodar localmente (Python 3.12):

```bash
git clone -b feat/random_search_period_lexicon https://github.com/fabiolobato/hackathon_ocr_historico.git
cd hackathon_ocr_historico/entrega_ocr
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt jupyter
.venv/bin/jupyter notebook apresentacao.ipynb
```

Também dá para copiar a pasta para `Meu Drive/entrega_ocr` e abrir no Colab com `ORIGEM = 'drive'` na primeira célula.

Para regenerar a pasta depois de mudar o código-fonte, é preciso ter em `models/` os arquivos que a configuração final cita. Copie-os de `entrega_ocr/models/`.

```bash
pip install -r requirements-dev.txt
python tools/build_entrega.py --out entrega_ocr --execute   # monta a pasta e executa o caderno
```

### Linha de comando

Dentro de `entrega_ocr/`:

```bash
python -m ocr_pipeline.run --config configs/final.json --pages all --references ground_truth --viz --name final_todas
python -m ocr_pipeline.batch --config configs/final.json --images <pasta> --out saida   # imagens novas
```

`--pages` aceita `all`, `tuning`, `heldout` ou uma lista (`5,8,13`). Saída em `outputs/runs/<nome>/`: métricas, texto por página e, com `--viz`, um relatório HTML imagem ↔ texto.

## Estrutura do repositório

```
entrega_ocr/            pasta de apresentação: cadernos, código, configuração, dados, modelos, docs
docs/entrega/           fontes dos cadernos, README, PROCESSO e setup da pasta de entrega
ocr_pipeline/           código do pipeline: imagem, segmentação, reconhecimento, pós-processamento, métricas, relatórios
ocr_pipeline/tuning/    busca de parâmetros, pseudo-rótulos para o auto-treino e utilitários de experimentos
colab/                  jobs do Colab: auto-treino (self_train.py) e busca (search_job.py)
configs/                configurações (final_v4.json é a final; greedy_v4_medium.json, a mesma busca com o modelo publicado)
sweeps/                 especificações das buscas
tools/                  build_entrega.py (gera entrega_ocr/), score_final.py (avaliação final), dicionário de época
imagens/                as 12 páginas
test_kraken/            experimentos anteriores e transcrições de referência (ground_truth/)
```

Na raiz, os pesos, o dicionário e o modelo de contexto ficam em `models/`, fora do Git. As cópias em `entrega_ocr/models/` estão no Git. O reconhecedor st_v3 tem SHA-256 `d92f40f5…fb08f501`.
