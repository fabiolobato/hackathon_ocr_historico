# Processo: OCR do Livro de Registro de Leis nº 1 (1892–1894)

Documentação de tudo o que levou ao pipeline final: dados, protocolo, cada experimento com seus números, o que ficou, o que não funcionou e os limites.

## 1. Dados e protocolo

- **Páginas:** 12 páginas manuscritas (5 a 16) do _Livro de Registro de Leis nº 1_, todas com transcrição de referência (GT). O GT não foi verificado e tem erros prováveis, por exemplo `com causa` onde o manuscrito parece dizer `sem causa`.
- **Divisão fixa** (semente 42):
  - **ajuste** = páginas 5, 6, 8, 9, 11, 13, 14, 15, 16 (1.772 palavras normalizadas);
  - **reservadas** = 7, 10, 12 (~640 palavras).
  - Todas as escolhas de parâmetros foram feitas só nas páginas de ajuste. As reservadas servem de validação.
- **Sem treino:** o reconhecedor é usado como foi publicado. Nenhum peso de modelo foi ajustado com estas páginas.
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

### E07: busca gulosa condicional (configuração final)

A busca decide **todas** as etapas em conjunto, sempre sobre o melhor conjunto atual, só nas páginas de ajuste:

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

| etapa                     | escolha final                                  |
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
- **Léxico do domínio retirado.** A busca usou também o léxico do domínio (com ele: 43,17 no ajuste, 43,94 nas reservadas). Por ser montado com transcrições do conjunto avaliado, ele saiu da configuração final, que sem ele dá **46,95 no ajuste e 48,66 nas reservadas**. O resto da configuração foi mantido como a busca decidiu.

### Outros reconhecedores, como publicados

Não há modelo público para manuscrito brasileiro do século XIX. CER normalizado nas páginas 6, 8 e 13:

| modelo                                                                  |      CER |
| ----------------------------------------------------------------------- | -------: |
| **PP-OCRv6 medium**                                                     | **22,0** |
| TraPrInq (Inquisição portuguesa, séc. XVI–XIX, Kraken; Zenodo 14543904) |     38,8 |
| McCATMuS (multilíngue, séc. XVI–XXI; Zenodo 13788177)                   |     51,2 |

Os modelos históricos portugueses disponíveis leem esta mão pior que o PP-OCRv6.

## 4. Configuração final (`configs/final.json`)

| etapa                    | valor                                          |
| ------------------------ | ---------------------------------------------- |
| geometria                | ampliação ×1,5 → inclinação (±5°, passo 0,25°) |
| imagem de segmentação    | iluminação (σ=25) → CLAHE (2,0; 8)             |
| imagem de reconhecimento | iluminação (σ=25) → gama 0,7                   |
| reconhecedor             | PP-OCRv6 medium, como publicado                |
| reescore                 | ctc+edit, α = 2                                |
| léxico                   | só pt-BR 50k; razão 0,5; mínimo 3              |

| páginas                    | WER estrito | WER norm. | CER norm. |
| -------------------------- | ----------: | --------: | --------: |
| ajuste (9)                 |       50,26 |     46,95 |     20,60 |
| **reservadas (7, 10, 12)** |       51,03 | **48,66** |     17,79 |
| todas (12)                 |       50,46 |     47,40 |     19,86 |

Referência, com o modelo como publicado e imagem crua: 64,57 nas reservadas e 63,52 em todas.

## 5. O que não funcionou

- Reconhecer a imagem binarizada: +9,6.
- Remoção de ruído (+13,7) e de pauta (+16,4): apagam traços.
- Modelo small (≈ +4) e modelos históricos prontos (TraPrInq, McCATMuS).
- Filtros de linhas curtas: removem também fins de parágrafo.

## 6. Limites e próximos passos

- Só 3 páginas reservadas: o intervalo de confiança é largo.
- As reservadas já tinham sido olhadas em campanhas exploratórias anteriores do projeto, antes deste pipeline. Nenhuma escolha deste pipeline foi feita com elas, mas a separação não é perfeitamente cega.
- Todas as páginas são da mesma mão. Não há evidência de generalização para outros escribas.
- O GT não foi verificado, e parte do erro restante é da referência.
- A página 5 (521×543 px, outro recorte) tem segmentação ruidosa.

Próximos passos:

1. Um léxico histórico externo, de grafia do século XIX, sem usar o GT deste livro.
2. Revisar o GT.
3. Mais páginas transcritas, para validar com mais confiança.
4. Segmentação dedicada para páginas fora do padrão.

## 7. Reprodução

- Versões fixadas em `requirements.txt` (Kraken no commit `bb487219`, torch 2.11.0, Python 3.12). Semente 42 em todo lugar.
- Hashes conferidos por `setup_pipeline.py`:
  - medium: MD5 `e3411a45…`
  - léxico: SHA-256 `a61d6f2e…`
- Resultados por configuração em `outputs/runs/<nome>/result.json`, com a configuração completa.
- A busca de parâmetros (`ocr_pipeline/tuning/`) fica no repositório do projeto, não nesta pasta. O relatório dela está em `relatorios/busca_gulosa.html`.
