# OCR histórico: Livro de Registro de Leis nº 1 (1892–1894)

Transcrição automática de 12 páginas manuscritas da Câmara de São Carlos do Pinhal. O repositório junta o trabalho de cada frente do time num pipeline só: qualquer imagem de entrada pode ir para qualquer motor, e tudo é medido pelas mesmas métricas.

## Partes do time e onde cada uma entra

Todas as branches (`chore/add_dataset_files`, `feat/add_imagens_preprocessadas`, `feat/OCR`, `llm_implementation`) já estavam integradas na `main`. Não havia commit pendente em nenhuma delas. O que faltava era fazê-las trabalharem juntas.

| Parte | Pasta | Papel no pipeline integrado |
|---|---|---|
| Dataset | `imagens/` | imagens originais (entrada `original`) |
| Pré-processamento | `imagens_preprocessadas/` | binarização do time (entrada `binarizada`) |
| OCR Kraken | `test_kraken/` | motor Kraken: segmentação neural + PP-OCRv6 medium; também fornece a entrada `nlbin` |
| LLM | `LLM_Transcription.ipynb` | motor LLM (OpenRouter, Gemini 3.1 Pro) e as métricas WER/CER/PER |
| Gabarito | `ground_truth/` | um `.txt` por página, extraído do DOCX; revisão em `comparison_review.md` |
| **Integração** | `ocr_historico/`, `pipeline_integrado.ipynb` | liga as partes acima |
| **Fine-tuning** | `ocr_historico/finetuning.py`, `finetuning_kraken.ipynb` | ajusta o Kraken à letra do livro |

## Como rodar

**Colab:** abra `pipeline_integrado.ipynb` (**Arquivo → Abrir notebook → GitHub**), escolha um ambiente com GPU e execute em ordem. O notebook clona o repositório, instala as dependências, roda o Kraken e avalia. As células de LLM ficam em modo simulação até alguém mudar `PERMITIR_GASTO = True`.

**Linha de comando** (com `pip install -r requirements.txt`):

```bash
python -m ocr_historico.executar rodar kraken_original kraken_binarizada --paginas 7 10 12
```

```bash
python -m ocr_historico.executar rodar llm_corrige_kraken_original_2x --simular
```

```bash
python -m ocr_historico.executar avaliar
```

## Combinações

O nome da combinação diz o que entra e em qual motor. O nome é também o do arquivo em `resultados/`.

| Nome | O que faz | Custo |
|---|---|---|
| `kraken_original`, `kraken_nlbin`, `kraken_binarizada` | imagem original / nlbin do Kraken / binarização do time → Kraken | grátis |
| `kraken_original_2x`, `kraken_binarizada_2x` | idem, ampliada 2× antes | grátis |
| `llm_original`, `llm_binarizada` | imagem → LLM (o primeiro é o que o `LLM_Transcription.ipynb` já fazia) | ~US$ 0,06/página |
| `llm_corrige_<combinação kraken>` | imagem original **+ texto do Kraken** → LLM corrige (padrão: `llm_corrige_kraken_original_2x`) | ~US$ 0,06/página |

As entradas aceitam qualquer combinação entre `original`, `binarizada` e `nlbin`, com ou sem `_2x`.

## Formato comum de resultados

Todos os motores gravam o mesmo JSON do `LLM_Transcription.ipynb`, com campos a mais:

```json
{"combinacao": "kraken_binarizada", "modelo": "kraken/medium", "entrada": "binarizada", "escala": 1.0,
 "imagens": [{"arquivo": "5.BR FPMSC ....jpg", "texto_transcrito": "...", "custo": null}]}
```

Por isso um JSON já gerado no notebook do LLM pode ser importado sem conversão (`importar_resultado_llm`). A execução grava após cada página e retoma de onde parou. Resultados de outro modelo na mesma pasta não são misturados.

## Métricas

`ocr_historico/metricas.py` reúne as métricas das duas frentes:

- `wer_sem_pontuacao`, `cer_sem_pontuacao`, `wer_com_pontuacao`, `cer_com_pontuacao`, `per`: do notebook do LLM (minúsculas; PER segundo arXiv:2310.02943). A única mudança é a normalização Unicode NFC antes de comparar.
- `wer_estrito`: a métrica do `test_kraken` (caixa, acentos e pontuação preservados).

Os valores de corpus somam os erros de todas as páginas; não são médias de porcentagens. `resultados/comparacao.csv` tem uma linha por combinação, ordenada por `wer_sem_pontuacao`, e `resultados/metricas_paginas.csv` tem uma linha por página.

O gabarito ainda é provisório. Ele moderniza grafias, expande abreviações e tem erros de leitura listados em `ground_truth/comparison_review.md`. Uma transcrição fiel ao manuscrito perde pontos por isso.

## Resultados atuais (Kraken, 12 páginas)

Kraken medium em CPU, segmentação neural, mesmas 12 páginas e mesmo gabarito (`resultados/comparacao.csv`):

| Combinação | WER estrito | WER s/ pont. | CER s/ pont. | PER |
|---|---|---|---|---|
| `kraken_original_2x` | 64,2% | 62,9% | 22,2% | 35,4% |
| `kraken_original` | 64,7% | 63,3% | 22,8% | 34,7% |
| `kraken_nlbin` | 73,1% | 71,9% | 33,8% | 59,1% |
| `kraken_binarizada_2x` | 73,5% | 72,1% | 32,8% | 60,1% |
| `kraken_binarizada` | 74,9% | 73,7% | 35,0% | 59,2% |

Com a segmentação neural, **binarizar piora o Kraken em ~9 pontos de WER e ~11 de CER**, tanto com a binarização do time quanto com o nlbin do Kraken, que é o `kraken_neural` do `test_kraken`. A ampliação 2× ajuda pouco. Por isso a correção pelo LLM usa por padrão `kraken_original_2x` como base. As combinações de LLM ainda não foram executadas (ver custos abaixo). O código que gerou `imagens_preprocessadas/` não está no repositório; com ele, dá para testar outros parâmetros de binarização.

## Fine-tuning do Kraken

`finetuning_kraken.ipynb` (Colab com GPU) ou `python -m ocr_historico.finetuning --dispositivo cuda:0`:

1. **Dataset:** as 9 páginas de treino passam pela mesma entrada da melhor combinação (original 2×), são segmentadas e reconhecidas pelo modelo base. O gabarito de cada página é cortado nas quebras de linha do OCR (`ocr_historico/alinhamento.py`). Localmente, isso deu 259 linhas usadas e 52 descartadas (rubricas, números de página, linhas mal alinhadas), todas listadas em `finetuning/linhas/manifesto.json`.
2. **Treino:** `ketos train` a partir do PP-OCRv6 medium (`--resize union`, augmentation, 30 épocas, lote 16). O melhor checkpoint vira `finetuning/modelo/best_<acurácia>.safetensors`.
3. **Avaliação:** antes × depois nas páginas reservadas 7, 10 e 12, que nunca entram no treino (mesmo sorteio do `test_kraken`).

O caminho completo foi testado em CPU com um treino mínimo (3 linhas, 1 época). O treino de verdade precisa de GPU: em CPU, o lote padrão do ppocrv6 estoura a memória e cada época leva horas. O modelo aprende a grafia do gabarito, então revise `ground_truth/` antes de um treino definitivo.

## Custos e chave do OpenRouter

Nada é cobrado sem autorização explícita: `MotorLLM` recusa enviar sem `permitir_gasto=True` (`--permitir-gasto` na linha de comando). Com `simular=True` (`--simular`), as requisições são montadas e salvas em `resultados/simulacao/`, que fica fora do git, sem sair da máquina. A chave é lida de `OPENROUTER_API_KEY`, do segredo `API_KEY` do Colab ou de um arquivo `.env` na raiz, que também fica fora do git.

## Testes

```bash
python -m pytest
```

```bash
cd test_kraken && python -m pytest
```

Os testes de integração usam motores falsos e não fazem requisições.
