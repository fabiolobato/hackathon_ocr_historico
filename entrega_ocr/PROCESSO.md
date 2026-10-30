# Processo: OCR do Livro de Registro de Leis nº 1 (1892–1894)

Documentação de tudo o que levou ao pipeline final: dados, protocolo, cada experimento com seus números, o que ficou, o que não funcionou e os limites.

**Resultado final** (`configs/final.json`, WER normalizado nas reservadas 7, 10, 12): **22,05%** (CER 8,16%). O modelo como publicado, sobre a imagem crua, dava 64,57%.

## 1. Dados e protocolo

- **Páginas:** 12 páginas manuscritas (5 a 16) do _Livro de Registro de Leis nº 1_, todas com transcrição de referência (GT). O GT não foi verificado e tem erros prováveis, por exemplo `com causa` onde o manuscrito parece dizer `sem causa`.
- **Divisão fixa** (semente 42):
  - **ajuste** = páginas 5, 6, 8, 9, 11, 13, 14, 15, 16 (1.772 palavras normalizadas);
  - **reservadas** = 7, 10, 12 (~640 palavras).
  - Todas as escolhas de parâmetros foram feitas só nas páginas de ajuste. As reservadas servem de validação.
- **Nenhuma das 12 páginas entra no treino de pesos.** O reconhecedor final foi adaptado só com as **outras 91 páginas do mesmo livro**, com rótulos gerados pelo próprio pipeline, sem nenhuma transcrição humana (seção E11).
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
texto → reescore CTC com léxico de época + contexto n-grama → convenções, ordinais, hifenização → correção por edição → texto final
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
| `context_lm.py`                   | modelo de n-gramas de palavras e escolha por linha (Viterbi)                      |
| `beam_decode.py`                  | busca em feixe CTC com léxico; aqui só sugere candidatas extras                   |
| `postprocess.py`                  | convenções, ordinais, hifenização e léxico                                        |
| `metrics.py`                      | WER e CER                                                                         |
| `run.py`, `batch.py`              | linha de comando: páginas do livro / lote de imagens novas                        |
| `page_crop.py`                    | recorte fixo das fotos inteiras do acervo (tira a faixa da marca d'água)          |
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

O PP-OCRv6 foi treinado pelos autores com imagens reais e perde ~9 pontos lendo pixels binarizados. A binarização, porém, ajuda a segmentação. O medium supera o small em todas as páginas.

### E02: convenções de notação e hifenização

O modelo escreve `n^o`, `1^o`, `⁋`, enquanto o GT usa `nº`, `1º`, `§`. Palavras partidas no fim da linha viram dois tokens. São duas regras fixas, iguais para todas as páginas:

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

Isoladamente, o reconhecedor prefere a imagem como ela é. A remoção de ruído e a de pauta apagam traços finos. Na busca do E07, decidida em conjunto, a iluminação passou a valer 5 pontos.

### E04: correção léxica

Cada palavra fora do léxico, com tamanho mínimo, é trocada pela palavra do léxico de menor distância de edição, se a distância for ≤ tamanho × razão.

- **Léxico genérico:** FrequencyWords pt_BR 50k (OpenSubtitles 2018). A grafia é moderna: não tem `effectivamente`, `contracto` etc. Sozinho, leva 59,20 → **56,77**.
- **Léxico do domínio:** palavras do GT das outras páginas de ajuste, sem a página avaliada. Também foi testado e somava cerca de 6 pontos. Por usar transcrições do conjunto avaliado, **ficou fora da configuração final** (ver E07).

### E05: geometria, segmentação e padding com o léxico ativo

Com cada variante testada isoladamente, nada saiu de ±0,5 ponto da referência (ampliação, limiar do nlbin, padding, filtro de linhas, contraste). Decomposição do erro restante:

- substituições: 42,6%;
- deleções: 3,7%;
- inserções: 1,4%.

O gargalo é a leitura dos caracteres, não a segmentação.

### E06: reescore CTC

O reconhecimento é refeito guardando as probabilidades por quadro (as 20 classes mais prováveis). Para cada palavra fora do léxico, os candidatos do léxico dentro do raio de edição recebem `log P_ctc(candidato | quadros) + α`. A palavra lida pelo modelo concorre com α = 0. No modo `ctc+edit`, a correção por edição roda depois.

O reescore escolhe melhor entre candidatos empatados em distância de edição: ganhou cerca de 1,3 ponto sobre a correção por edição pura. Controle de sanidade: com α = −1000 o resultado reproduz exatamente o "sem léxico".

### E07: busca gulosa condicional (v1)


Primeira configuração entregue (v1). A busca decide **todas** as etapas em conjunto, sempre sobre o melhor conjunto atual, só nas páginas de ajuste:

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

A mesma busca, com mais etapas e candidatos, decidiu as versões seguintes (E09, E10 e E12).

| etapa                     | escolha (v1)                                   |
| ------------------------- | ---------------------------------------------- |
| geometria                 | ampliação ×1,5 + correção de inclinação        |
| imagem de segmentação     | iluminação + CLAHE (o nlbin saiu)              |
| imagem de reconhecimento  | iluminação + gama 0,7                          |
| filtro de linhas, padding | nenhum, 16                                     |
| texto                     | convenções + hifenização                       |
| léxico                    | pt-BR 50k inteiro, razão 0,5, tamanho mínimo 3 |
| decodificador             | ctc+edit, α = 2                                |

Achados:

- **O contraste piorou sozinho (+0,73) e ajudou sobre o nlbin.** Somando só o que melhora isoladamente, essa combinação nunca apareceria.
- **No passe 2 a segmentação trocou nlbin + contraste por iluminação + CLAHE**, depois que o reconhecimento e o léxico estavam ativos.
- **O custo de retirar cada escolha** (passe para trás):
  - léxico pt-BR: +8,35
  - iluminação no reconhecimento: +5,08
  - ampliação: +2,03
  - ctc+edit: +2,03
  - hifenização: +1,41
  - o resto: entre +0,06 e +1,19
- **Bordas da grade.** Testados um por vez: α 1 e 4, razão 0,45 e 0,6, gama 0,5 e 0,85, ampliação 1,25 e 1,75, e 20 candidatos CTC. Nada passou do ruído.
- **Léxico do domínio retirado.** A busca usou também o léxico do domínio (com ele: 43,17 no ajuste, 43,94 nas reservadas). Por ser montado com transcrições do conjunto avaliado, ele saiu da configuração final, que sem ele dá **46,95 no ajuste e 48,66 nas reservadas**. A meta (≤ 50%) foi atingida aqui, sem treinar pesos.

### E08: ajuste fino com o GT (descartado pelo protocolo)

O medium foi ajustado com as linhas das 9 páginas de ajuste e avaliado só nas reservadas: 47,87 → **29,61** (CER 19,2 → 10,0).

Isso mostrou que adaptar o reconhecedor à mão do escrivão vale mais do que qualquer outro ajuste. Mas usa transcrições do conjunto avaliado, então **não entra na configuração final**. Ficou só como medida do quanto há para ganhar, e deu origem ao E11.

### E09: dicionário de época e modelo de contexto (v3)

O léxico pt-BR moderno "corrigia" _approvação_ para _aprovação_, o que conta como erro contra o GT, que segue a grafia anterior a 1911. Foram adicionados:

- **Dicionário de época** (`tools/build_period_lexicon.py`). Livros em português do Projeto Gutenberg de autores nascidos entre 1790 e 1875: 443 livros, ~16 milhões de palavras. A regra de seleção usa só o catálogo, nunca as páginas.
- **Modelo de contexto.** N-gramas de palavras do mesmo corpus. Entre as candidatas que o CTC acha parecidas, escolhe a sequência mais provável da linha inteira (_camara municipal_ contra _camara municipaes_).
- **Busca aleatória × gulosa.** Com ~1.000 configurações sorteadas no mesmo espaço, a aleatória chegou a 43,62. A gulosa chegou a **36,00 no ajuste e 36,85 nas reservadas**.

### E10: diagnóstico dos erros e corpus legislativo

Os erros restantes foram classificados um a um. Daí saíram as mudanças:

- **Ordinais.** O modelo lê `2o` e `3a`; o GT tem `2º` e `3ª`.
- **Palavras curtas.** 145 erros eram _a/o/e/de/da/do_. Essas palavras ganharam candidatas, e o contexto escolhe entre elas.
- **N-best.** As candidatas também são buscadas a partir da 2ª à 5ª leitura mais provável do CTC.
- **Corpus legislativo.** Leis e decretos de São Paulo de 1889 a 1910, da API da Assembleia Legislativa: 3.565 atos, ~2 milhões de palavras. É o mesmo gênero e a mesma época.
  - Os atos que citam São Carlos foram excluídos.
  - Só 11 dos 2.258 8-gramas do GT aparecem no corpus, e todos são fórmulas genéricas (_revogam-se as disposições em contrario_). É o vocabulário que um transcritor da época também teria.
  - Entra com peso 1 no dicionário e peso 4 no modelo de contexto.

O diagnóstico também mostrou onde está o limite. O pós-processamento já recuperava 23,6 pontos (reconhecedor sozinho 55,25 → 31,66). Das palavras ainda erradas, as lidas a 3 ou mais edições da certa somavam ~10,8 pontos, e nenhum método de léxico conserta essas leituras. **O teto está na leitura do reconhecedor.**

### E11: auto-treino do reconhecedor com as outras páginas do livro

O acervo da Fundação Pró-Memória de São Carlos tem o livro inteiro: 106 fotos. As 12 páginas avaliadas são as fotos 5 a 16. O uso foi autorizado pela parceria do projeto com o dono do material.

As 91 outras páginas, sem capas e sem as fotos 5 a 16, foram recortadas com uma caixa fixa (IoU 0,96–0,99 contra os recortes feitos à mão). Não há transcrição delas.

**Como funciona:**

1. O pipeline completo lê as 91 páginas.
2. As linhas lidas com confiança viram rótulos. O filtro é: ≥ 90% das palavras no dicionário de época e confiança média do CTC ≥ limiar.
3. O medium é ajustado com essas linhas (20 épocas, lr 1e-4, com aumento de dados, numa GPU L4).
4. O modelo novo volta ao pipeline e gera rótulos melhores para a rodada seguinte.

**Calibração do limiar.** Nas 9 páginas de ajuste, que têm GT, mede-se o erro dos rótulos para cada limiar. O escolhido mantém o CER dos rótulos em ~9%. É um parâmetro como os outros, escolhido só no ajuste. As reservadas não participam.

| reconhecedor | linhas de treino | só reconhecedor (WER / CER) | pipeline (WER / CER) |
|---|---:|---:|---:|
| medium, como publicado | — | 61,17 / 21,53 | 29,91 / 12,53 |
| st_v1 | 1.339 | 43,45 / 14,46 | 28,67 / 11,50 |
| st_v2 | 1.577 | 42,10 / 13,27 | 27,37 / 10,85 |
| **st_v3** | 1.840 | 40,18 / 12,98 | **26,07 / 10,32** |
| st_v4 | 1.931 | 39,05 / 12,95 | 26,69 / 10,62 |

Os valores são das páginas de ajuste, com a mesma configuração de pós-processamento.

**Critério de parada combinado antes:** parar quando a rodada ganhar menos de 0,5 ponto no pipeline.

- Até o st_v3, cada rodada ganhou ~1,3 ponto.
- O st_v4 ainda lê melhor sozinho, mas piora no pipeline (+0,62). Os rótulos dele já vêm moldados pelo léxico e pelo contexto, e o modelo passa a repetir as escolhas do pós-processamento em vez de trazer informação nova da imagem.
- **Ficou o st_v3.**

**Detalhe técnico que importou:** o recorte das linhas no treino (`ketos compile`) e na leitura usa o mesmo padding e o mesmo modo de polígono. Isso foi conferido no código do Kraken.

### E12: busca final com o reconhecedor novo

Um reconhecedor novo muda o que vale a pena corrigir, então as etapas que dependem da leitura foram redecididas com o st_v3:

- **Redecidas:** geometria, filtro de linhas, imagem de reconhecimento, padding, léxico, decodificador e contexto.
- **Mantidas:** segmentação, regras de texto e tipo de decodificador. Não mudaram em nenhum passe da busca anterior (gulosa v4 com o medium, que convergiu em 29,63).

| passe | WER (ajuste) | mudança |
|---|---:|---|
| início (escolhas da gulosa v4) | 26,07 | — |
| 1 | 24,77 | padding 32, feixe como candidatas extras; saem o léxico genérico e a comparação sem acento |
| 2 | **24,60** | padding 24; sai a distância mínima 2 do contexto |
| 3 | 24,60 | nada mudou: fim |

Com o reconhecedor adaptado, a linha precisa de mais margem: padding 24, contra 8 com o medium. E o léxico moderno deixa de ajudar.

**Quanto cada escolha vale** (WER no ajuste ao retirar só ela, medido no passo para trás do passe 2):

| retirar | custo |
|---|---:|
| dicionário de época | +15,58 |
| modelo de contexto | +6,49 |
| trocar também palavras que estão no dicionário (`open_known`) | +3,33 |
| ampliação ×2 | +2,43 |
| candidatas para palavras curtas | +1,52 |
| feixe como candidatas extras | +0,68 |
| padding 24 | +0,62 |
| trigrama (contra bigrama) | +0,51 |
| as demais | < 0,4 cada |

Os ~5 pontos do reconhecedor (medium 29,63 contra st_v3 24,77 no ajuste) se somam a esses.

### Outros reconhecedores, como publicados


Não há modelo público para manuscrito brasileiro do século XIX. CER normalizado nas páginas 6, 8 e 13:

| modelo                                                                  |      CER |
| ----------------------------------------------------------------------- | -------: |
| **PP-OCRv6 medium**                                                     | **22,0** |
| TraPrInq (Inquisição portuguesa, séc. XVI–XIX, Kraken; Zenodo 14543904) |     38,8 |
| McCATMuS (multilíngue, séc. XVI–XXI; Zenodo 13788177)                   |     51,2 |

Os modelos históricos portugueses disponíveis leem esta mão pior que o PP-OCRv6.

## 4. Configuração final (`configs/final.json`)

| etapa | valor |
|---|---|
| geometria | ampliação ×2 |
| imagem de segmentação | contraste ×1,8 → remoção de pauta |
| segmentador | blla (padrão do Kraken), sem filtro de linhas |
| imagem de reconhecimento | a própria base, sem tratamento |
| reconhecedor | **st_v3**: PP-OCRv6 medium auto-treinado nas 91 outras páginas (E11), padding 24 |
| reescore CTC | ctc+edit, α = 16, 30 candidatas por palavra, buscadas a partir das 5 melhores leituras |
| contexto | trigramas do corpus de época + legislativo (peso 4), β = 1; feixe de 64 como fonte extra de candidatas |
| texto | convenções, ordinais, hifenização |
| léxico | só o dicionário de época + legislativo (peso 1); razão 0,5; palavras com ≥ 4 letras; 60 palavras curtas |

| páginas | WER norm. | CER norm. |
|---|---:|---:|
| ajuste (9) | 24,77 | 9,78 |
| **reservadas (7, 10, 12)** | **22,05** | **8,16** |
| todas (12) | 24,05 | 9,35 |

Por página, nas reservadas: 7: 24,7 · 10: 19,3 · 12: 21,8.

Valores em CPU. No Colab, com GPU, o ajuste dá 24,60: as diferenças numéricas entre CPU e CUDA mudam alguns empates do reescore.

### Progressão nas reservadas

Cada linha foi avaliada uma única vez nas reservadas, depois de decidida no ajuste.

| versão | o que entrou | reservadas WER / CER |
|---|---|---:|
| modelo como publicado, imagem crua | — | 64,57 / 21,71 |
| v1 | imagens separadas, léxico pt-BR, reescore CTC, busca gulosa | 48,66 / 17,79 |
| v3 | dicionário de época, contexto n-grama | 36,85 / 14,00 |
| v4, medium | ordinais, curtas, n-best, corpus legislativo; todas as etapas redecididas | 28,50 / 10,64 |
| **final** | reconhecedor auto-treinado (st_v3) + busca final | **22,05 / 8,16** |

O ganho nas reservadas acompanha o do ajuste, e no fim é até maior. Não há sinal de sobreajuste dos parâmetros às páginas de ajuste.

## 5. O que não funcionou

- Reconhecer a imagem binarizada: +9,6.
- Remoção de ruído (+13,7) e de pauta (+16,4) na imagem de reconhecimento: apagam traços. A remoção de pauta só ajuda na imagem de segmentação.
- Modelo small (≈ +4) e modelos históricos prontos (TraPrInq, McCATMuS).
- Filtros de linhas curtas: removem também fins de parágrafo.
- **Busca aleatória** no lugar da gulosa: 43,62 contra 36,00, com mais do que o dobro de avaliações (1.073 contra 465).
- **Busca em feixe com léxico e n-gramas como decodificador principal:** ~8 pontos pior que o reescore. Ela troca palavras inteiras por outras mais frequentes onde a imagem é ambígua. Ficou só como fonte de candidatas extras.
- **Proteger palavras lidas com confiança:** o modelo erra com convicção. Com a proteção, o resultado ficou entre 31,60 e 32,28, contra 31,66 sem ela.
- **Novos tratamentos de imagem para o reconhecedor** (canal azul, engrossar ou afinar o traço, corrigir a inclinação das letras, média entre versões da imagem): no máximo 1 ponto sozinhos (canal azul), e nada na busca final. O gargalo era a letra, não a imagem.
- **4ª rodada de auto-treino:** ver E11.

## 6. Limites e próximos passos

- Só 3 páginas reservadas, ~640 palavras: 1 ponto de WER ≈ 6 palavras.
- As reservadas já tinham sido olhadas em campanhas exploratórias anteriores do projeto, antes deste pipeline. Nenhuma escolha deste pipeline foi feita com elas, mas a separação não é perfeitamente cega.
- O reconhecedor foi adaptado a este livro e a este escrivão. Em outro livro, o ponto de partida é o medium publicado (`configs/greedy_v4_medium.json`, 28,50 nas reservadas), ou uma nova rodada de auto-treino com as páginas daquele livro.
- Os rótulos do auto-treino vêm de um pipeline cujos parâmetros foram escolhidos nas páginas de ajuste. Por isso o número honesto é o das reservadas.
- O GT não foi verificado, e parte do erro restante é da referência.
- A página 5 (521×543 px, outro recorte) continua sendo a pior: 57,8% de WER.

Próximos passos:

1. Revisar o GT.
2. Transcrever algumas das 91 páginas extras: dão validação maior e permitiriam um ajuste fino supervisionado dentro do protocolo.
3. Segmentação dedicada para páginas fora do padrão.

## 7. Reprodução

- Versões fixadas em `requirements.txt` (Kraken no commit `bb487219`, torch 2.11.0, Python 3.12). Semente 42 em todo lugar.
- **Arquivos do modelo e do léxico que vêm nesta pasta** (`setup_pipeline.py` confere que estão presentes):

| arquivo | tamanho | SHA-256 |
|---|---:|---|
| `models/finetuned/st_v3.safetensors` | 64 MB | `d92f40f5…fb08f501` |
| `models/lexicon/pt_hist_legal_w1.txt` | 2,4 MB | `207246e6…dde3ef51` |
| `models/lexicon/pt_hist_legal_w4.ngrams.npz` | 15 MB | `e99eb83e…3eddbb` |

- O medium publicado (Zenodo 10.5281/zenodo.21788410, MD5 `e3411a45…`) e o léxico pt-BR 50k são baixados e conferidos por `setup_pipeline.py` quando uma configuração os usa.
- A busca, o auto-treino e os scripts do Colab (`ocr_pipeline/tuning/`, `colab/`, `tools/`) ficam no repositório do projeto, não nesta pasta. O relatório da busca final está em `relatorios/busca_gulosa.html`.
- A avaliação final está em `outputs/search/final_scores_v4.json` no repositório, gerada por `tools/score_final.py`.
