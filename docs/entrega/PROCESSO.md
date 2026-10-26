# Processo: OCR do Livro de Registro de Leis nº 1 (1892–1894)

Documentação de tudo o que foi feito até o pipeline final: dados, protocolo, cada experimento com seus números, o que ficou, o que não funcionou e os limites.

## 1. Dados e protocolo

- **Páginas:** 12 páginas manuscritas (5 a 16) do _Livro de Registro de Leis nº 1_, todas com transcrição de referência (GT). O GT não foi verificado e tem erros prováveis, por exemplo `com causa` onde o manuscrito parece dizer `sem causa`.
- **Divisão fixa** (semente 42):
  - **ajuste** = páginas 5, 6, 8, 9, 11, 13, 14, 15, 16 (1.772 palavras normalizadas);
  - **reservadas** = 7, 10, 12 (~640 palavras).
  - Todas as escolhas foram feitas só nas páginas de ajuste. As reservadas servem de validação.
- **Métricas:** WER e CER do corpus, somando as edições de todas as páginas e dividindo pelo tamanho total da referência (nunca média de páginas). São duas políticas:
  - **normalizada:** NFC, minúsculas, pontuação vira espaço. É a métrica da meta.
  - **estrita:** caixa e pontuação preservadas.
- **Meta:** WER normalizado ≤ 50%.
- **Ruído:** com 1.772 palavras, 1 ponto de WER ≈ 18 palavras. Nas reservadas, 1 ponto ≈ 6 palavras.

## 2. Arquitetura

```
original → geometria (ampliação, correção de inclinação)       = base (define as coordenadas)
base → operações da imagem de segmentação  → segmentador blla (Kraken) → filtro de linhas → polígonos
base → operações da imagem de reconhecimento → PP-OCRv6 (Kraken), linha a linha → texto
texto → reescore CTC com léxico → convenções → hifenização → correção por distância de edição → texto final
```

Operações fotométricas (iluminação, CLAHE, gama, binarização…) não movem pixels. Por isso os polígonos encontrados na imagem de segmentação valem na imagem de reconhecimento, e as duas podem ser diferentes. Essa separação deu o primeiro ganho.

Módulos (`ocr_pipeline/`):

| módulo                            | função                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------- |
| `config.py`                       | configuração tipada (dataclasses congeladas)                                      |
| `preprocessing.py`                | operações de imagem                                                               |
| `engine.py`                       | Kraken: segmentação e reconhecimento, com cache em disco por hash da configuração |
| `pipeline.py`                     | executa uma página                                                                |
| `ctc_rescore.py`                  | reescore com as probabilidades do reconhecedor                                    |
| `postprocess.py`                  | convenções, hifenização e léxico                                                  |
| `metrics.py`                      | WER e CER                                                                         |
| `run.py`                          | linha de comando                                                                  |
| `visualize.py`, `presentation.py` | relatórios imagem ↔ texto                                                         |

## 3. Experimentos

Todos os números são WER normalizado (%) nas 9 páginas de ajuste, salvo indicação.

### E01: separar a imagem da segmentação da do reconhecimento

| variante                                                       |       WER |
| -------------------------------------------------------------- | --------: |
| nlbin na segmentação **e** no reconhecimento (baseline antiga) |     70,54 |
| imagem crua, modelo small                                      |     67,10 |
| imagem crua, modelo medium                                     |     63,15 |
| **segmenta no nlbin, reconhece na original**                   | **60,95** |

O PP-OCRv6 foi treinado com imagens reais e perde ~9 pontos lendo pixels binarizados. A binarização, porém, ajuda a segmentação. O medium supera o small em todas as páginas.

### E02: convenções de notação e hifenização

O modelo escreve `n^o`, `1^o`, `⁋`, enquanto o GT usa `nº`, `1º`, `§`. Palavras partidas no fim da linha viram dois tokens. São duas regras fixas, sem consulta ao GT:

- **convenções:** `^o→º`, `^a→ª`, `⁋→§`;
- **hifenização:** se a linha termina em `-`, `=` ou `¬`, junta com a primeira palavra da linha seguinte.

Resultado: 60,95 → **59,20**.

### E03: tratamento da imagem de reconhecimento, isolado

| operação         |  Δ WER |
| ---------------- | -----: |
| contraste        |  −1,02 |
| nitidez          |  −0,68 |
| CLAHE            |  −0,23 |
| iluminação       |   0,00 |
| gama 0,7         |  +3,38 |
| tons de cinza    |  +4,40 |
| remoção de ruído | +13,66 |
| remoção de pauta | +16,42 |

Isoladamente, o reconhecedor prefere a imagem como ela é. A remoção de ruído e a de pauta apagam traços finos. Nada entrou nesta fase. Na busca do E08, decidida em conjunto, a iluminação passou a valer 5 pontos.

### E04: correção léxica

Cada palavra fora do léxico, com tamanho mínimo, é trocada pela palavra do léxico de menor distância de edição, se a distância for ≤ tamanho × razão. São dois léxicos:

- **domínio:** palavras do GT das páginas de ajuste, sempre **sem a página avaliada** (_leave-one-out_). Nas reservadas usa-se o GT das 9 de ajuste. Sem o _leave-one-out_ o léxico conteria a resposta.
- **genérico:** FrequencyWords pt_BR 50k (OpenSubtitles 2018). A grafia é moderna: não tem `effectivamente`, `contracto` etc.

| variante                                 |       WER |
| ---------------------------------------- | --------: |
| sem léxico                               |     59,20 |
| só genérico                              |     56,77 |
| só domínio                               |     53,44 |
| **os dois, razão 0,4, 50k inteiro (v1)** | **49,04** |

Foi o maior ganho isolado do projeto, de −10 pontos. Nas reservadas a v1 deu 49,76, sem sinal de sobreajuste.

### E05: geometria, segmentação e padding com o léxico ativo

Nenhuma das variantes testadas saiu de ±0,5 ponto da referência (ampliação, limiar do nlbin, padding, filtro de linhas, contraste). Decomposição do erro restante:

- substituições: 42,6%;
- deleções: 3,7%;
- inserções: 1,4%.

O gargalo é a leitura dos caracteres, não a segmentação. Isso motivou o ajuste fino (E07).

### E06: reescore CTC

O reconhecimento é refeito guardando as probabilidades por quadro (as 20 classes mais prováveis). Para cada palavra fora do léxico, os candidatos do léxico dentro do raio de edição recebem `log P_ctc(candidato | quadros) + α`. A palavra lida pelo modelo concorre com α = 0. No modo `ctc+edit`, a correção por edição roda depois.

Resultado: 49,04 → **47,74** (v2). Nas reservadas a v2 deu **47,87**, e a meta foi atingida sem treinar pesos. Controle de sanidade: com α = −1000 o resultado reproduz exatamente o "sem léxico" (59,20).

### E07: ajuste fino do reconhecedor (ft_v1)

- **Dados.** O GT é por página, não por linha. As palavras do GT foram alinhadas às linhas do OCR por programação dinâmica, e linhas com distância > 0,6 foram descartadas. Ficaram 256 de 333 linhas, das 9 páginas de ajuste, em PageXML.
- **Treino.** Colab L4:

  ```
  ketos train -f page -i medium.safetensors --resize union -u NFD -B 4 \
      --accumulate-grad-batches 4 -r 1e-4 --augment -q fixed -N 50 -p 0.9
  ```

  Semente 42.

- **Avaliação.** Só nas reservadas, porque o modelo viu as páginas de ajuste.

| reconhecedor                          | WER estrito | WER norm. | CER norm. |   p7 |  p10 |  p12 |
| ------------------------------------- | ----------: | --------: | --------: | ---: | ---: | ---: |
| medium + pós-processamento v2         |       51,34 |     47,87 |     19,18 | 48,8 | 44,8 | 49,4 |
| **ft_v1 + pós-processamento v2 (v3)** |       34,60 | **29,61** |  **9,95** | 31,6 | 24,3 | 31,8 |
| ft_v2 (100 épocas)                    |       35,86 |     30,08 |     10,54 | 33,0 | 24,9 | 31,4 |

O CER cai pela metade. Com 100 épocas há sobreajuste: a validação interna sobe, mas o resultado nas reservadas piora. Para ir além são precisas mais páginas transcritas, não mais épocas.

### E08: busca gulosa condicional (v4)

A busca decide **todas** as etapas em conjunto, sempre sobre o melhor conjunto atual:

1. As etapas seguem esta ordem:
   - reconhecedor
   - imagem de segmentação e seus parâmetros
   - geometria
   - filtro de linhas
   - imagem de reconhecimento e parâmetros do reconhecedor
   - texto
   - léxico e seus parâmetros
   - decodificador e seus parâmetros
2. Em cada etapa, a referência é o conjunto atual sem as escolhas da própria etapa. Cada candidato é testado sozinho, e o melhor só entra se superar a referência.
3. Todos os demais candidatos da etapa são testados por cima do vencedor, **inclusive os que pioraram sozinhos**. Cada um fica só se melhorar.
4. Um passe para trás remove qualquer escolha cuja retirada não piore.
5. Os passes se repetem até o conjunto não mudar mais (aqui, 3 passes, 330 configurações).

Resultado: **63,15 → 43,17** no ajuste e **43,94** nas reservadas (v2: 47,87). A busca não sobreajustou.

| etapa                     | escolha final                                            |
| ------------------------- | -------------------------------------------------------- |
| geometria                 | ampliação ×1,5 + correção de inclinação                  |
| imagem de segmentação     | iluminação + CLAHE (o nlbin saiu)                        |
| imagem de reconhecimento  | iluminação + gama 0,7                                    |
| filtro de linhas, padding | nenhum, 16                                               |
| texto                     | convenções + hifenização                                 |
| léxico                    | domínio + pt-BR 50k inteiro, razão 0,5, tamanho mínimo 3 |
| decodificador             | ctc+edit, α = 2                                          |

Achados:

- **O contraste piorou sozinho (+0,73) e ajudou sobre o nlbin.** Somando só o que melhora isoladamente, essa combinação nunca apareceria.
- **No passe 2 a segmentação trocou nlbin + contraste por iluminação + CLAHE**, depois que o reconhecimento e o léxico estavam ativos.
- **O custo de retirar cada escolha** (passe para trás):
  - léxico pt-BR: +8,35
  - iluminação no reconhecimento: +5,08
  - léxico do domínio: +3,78
  - ampliação: +2,03
  - ctc+edit: +2,03
  - hifenização: +1,41
  - o resto: entre +0,06 e +1,19
- **Bordas da grade.** Testados um por vez: α 1 e 4, razão 0,45 e 0,6, gama 0,5 e 0,85, ampliação 1,25 e 1,75, e 20 candidatos CTC. Nada passou do ruído; o melhor, gama 0,5, ganhou −0,34 (~6 palavras).
- **A imagem de reconhecimento depende do reconhecedor.** Com o ft_v1, a configuração v4 dá 38,90 nas reservadas, porque o ft_v1 foi treinado com recortes da imagem crua. Com a imagem de reconhecimento crua volta a 29,76, empatando com a v3. Para o ft_v1 essas etapas não podem ser buscadas nas páginas de ajuste, que ele viu no treino. Por isso a v3 continua sendo a configuração dele.

### Outros reconhecedores (português histórico)

Não há modelo público para manuscrito brasileiro do século XIX. Foram testados:

| modelo                                                                              | resultado                                         |
| ----------------------------------------------------------------------------------- | ------------------------------------------------- |
| TraPrInq (Inquisição portuguesa, séc. XVI–XIX, Kraken; Zenodo 14543904), sem ajuste | CER 38,8 contra 22,0 do medium (páginas 6, 8, 13) |
| McCATMuS (multilíngue, séc. XVI–XXI; Zenodo 13788177), sem ajuste                   | CER 51,2                                          |
| TraPrInq ajustado com os mesmos dados do ft_v1 (40 épocas, parada antecipada)       | reservadas 30,39 (p7 27,4 · p10 29,3 · p12 33,9)  |

Partir de um modelo de português histórico não supera o PP-OCRv6 ajustado. Os erros dos dois, porém, caem em lugares diferentes.

Houve também um **conjunto por linha**: mesma segmentação, e em cada linha fica a saída cuja leitura gulosa tem mais palavras do léxico (no empate, o ft_v1). A regra foi fixada antes e rodada uma única vez: **28,50** nas reservadas, um ganho de ~7 palavras. Como é uma medição só e dobraria o tempo de reconhecimento, não entrou no pipeline.

## 4. Configurações finais

|                                             | v4 (`configs/best_v4.json`)                    | v3 (`configs/best_v3_ft.json`) |
| ------------------------------------------- | ---------------------------------------------- | ------------------------------ |
| geometria                                   | ampliação ×1,5 → inclinação (±5°, passo 0,25°) | –                              |
| imagem de segmentação                       | iluminação (σ=25) → CLAHE (2,0; 8)             | nlbin (limiar 0,5)             |
| imagem de reconhecimento                    | iluminação (σ=25) → gama 0,7                   | original                       |
| reconhecedor                                | PP-OCRv6 medium                                | ft_v1                          |
| reescore                                    | ctc+edit, α = 2                                | ctc+edit, α = 16               |
| léxico                                      | domínio LOO + pt-BR 50k; razão 0,5; mínimo 3   | igual                          |
| ajuste: WER norm.                           | 43,17                                          | – (treinou nelas)              |
| reservadas: WER estrito / norm. / CER norm. | 46,45 / 43,94 / 16,11                          | 34,60 / 29,61 / 9,95           |

## 5. O que não funcionou

- Reconhecer a imagem binarizada: +9,6.
- Remoção de ruído (+13,7) e de pauta (+16,4): apagam traços.
- Modelo small (≈ +4) e modelos históricos prontos (TraPrInq, McCATMuS).
- Treinar mais épocas (ft_v2): sobreajuste.
- Filtros de linhas curtas: removem também fins de parágrafo.
- Dar ao ft_v1 a imagem tratada da v4: +9,3.

## 6. Limites e próximos passos

- Só 3 páginas reservadas: o intervalo de confiança é largo.
- Todas as páginas são da mesma mão. Não há evidência de generalização para outros escribas.
- O GT não foi verificado. Parte do erro restante é da referência, e o ft_v1 aprendeu esses erros.
- A página 5 (521×543 px, outro recorte) tem segmentação ruidosa.

Próximos passos, em ordem de valor esperado:

1. Transcrever mais páginas: é o que limita o ajuste fino e a validação.
2. Revisar o GT.
3. Validação cruzada por página no ajuste fino, que permitiria buscar a configuração do ft_v1 e confirmar o conjunto por linha.
4. Segmentação dedicada para páginas fora do padrão.

## 7. Reprodução

- Versões fixadas em `requirements.txt` (Kraken no commit `bb487219`, torch 2.11.0, Python 3.12). Semente 42 em todo lugar.
- Hashes conferidos por `setup_pipeline.py`:
  - medium: MD5 `e3411a45…`
  - léxico: SHA-256 `a61d6f2e…`
  - ft_v1: SHA-256 `986c7670…`
- Resultados por configuração em `outputs/runs/<nome>/result.json`, com a configuração completa.

No repositório, não nesta pasta:

```bash
# busca gulosa (retoma do que já foi avaliado)
python -m ocr_pipeline.tuning.search sweeps/search_v1.json --pages tuning --probe 8
# dados do ajuste fino
python -m ocr_pipeline.tuning.finetune_data --config configs/best_v1.json --pages tuning --out outputs/finetune/v1
# esta pasta de entrega, com o caderno já executado
python tools/build_entrega.py --out entrega_ocr --execute
```
