# Registros do desenvolvimento

Tudo o que as buscas e o treino final produziram, para que quem continue o trabalho parta do que já foi medido
em vez de recomeçar. Os pesos do reconhecedor final não estão duplicados aqui: ficam em
`models/finetuned/st_c1m.safetensors` (o mesmo arquivo de `entrega_ocr/models/finetuned/`).

## Conteúdo

| pasta                                                  | o que é                                                                                  |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `buscas/v1_medium`                                     | busca gulosa condicional da v1 (PP-OCRv6 medium, CPU): 330 avaliações, 3 passes          |
| `buscas/v3_medium_3passes`, `buscas/v3_medium_4passes` | busca da v3: convergiu no passe 3 (465 avaliações); o passe 4 confirmou (518)            |
| `buscas/v4_medium`                                     | busca da v4 ainda com o medium: 767 avaliações                                           |
| `buscas/v4_st_v3`                                      | busca da v4 com o reconhecedor adaptado st_v3 (GPU L4): 306 avaliações                   |
| `buscas/g1_interrompida`                               | busca com o st_g1, interrompida (72 avaliações)                                          |
| `buscas/v5_st_g2`                                      | busca da v5 com o st_g2 (GPU L4): 313 avaliações                                         |
| `buscas/aleatoria`                                     | buscas aleatórias usadas na comparação com a gulosa (dois processos por versão)          |
| `buscas/v6_varredura_st_c1m`                           | varredura final com o reconhecedor da cadeia limpa: 20 mudanças isoladas, nenhuma aceita |
| `buscas/pontos_de_partida`                             | escolhas iniciais usadas para retomar buscas (`--init`)                                  |
| `buscas/logs`                                          | logs das buscas e das avaliações finais                                                  |
| `avaliacao_final`                                      | avaliação congelada de cada versão (v3 a v6) contra as duas referências                  |
| `treino/st_c1m`                                        | a run do reconhecedor final: log de treino, métricas, seleção das linhas e rótulos       |

Cada busca guarda:

- `evaluations.jsonl`: uma linha por configuração avaliada, com a configuração completa e o WER/CER por página;
- `decisions.json`: o caminho da busca (cada passe, cada etapa, o que entrou e o que saiu) e a escolha final;
- `best_config.json`: a melhor configuração encontrada;
- `run_config.json`: os argumentos e a especificação usados.

As especificações das buscas estão em `sweeps/`, e as configurações congeladas de cada versão, em `configs/`.
A configuração final é `configs/final_v6.json`. Todas as medições usam só as 9 páginas de ajuste; as páginas de
teste não entram em nenhum destes arquivos de busca.

## Partir do que já foi feito

Para rodar o pipeline final, use `configs/final_v6.json` (ou a pasta `entrega_ocr/`). Os modelos base do
Zenodo, quando necessários, são baixados por `setup_pipeline.py`.

Para continuar a busca, a ferramenta tem duas opções:

- `--init` retoma a partir das escolhas finais de uma busca anterior (um `decisions.json`);
- `--reuse` lê os `evaluations.jsonl` anteriores e não recalcula nenhuma configuração já avaliada nas mesmas páginas.

```bash
python -m ocr_pipeline.tuning.search sweeps/search_g2.json --pages tuning \
    --init registros/buscas/v5_st_g2/decisions.json \
    --reuse registros/buscas/*/evaluations.jsonl
```

O reaproveitamento vale para configurações idênticas, com o mesmo reconhecedor. As avaliações com o reconhecedor
final (st_c1m) são só as da varredura da v6 (`buscas/v6_varredura_st_c1m/sweep.json`, em outro formato), então uma
busca nova com ele recalcula as configurações, mas pode começar das escolhas da v5. O cache de segmentação e de
saídas do reconhecedor (`outputs/cache/`) não foi guardado; ele é refeito na primeira execução.

`imagens_extra/` traz as outras páginas do livro, os recortes usados no treino e as transcrições do Gemini
(rótulos de treino e segunda referência).
