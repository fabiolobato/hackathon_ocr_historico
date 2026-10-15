# Experimentos de OCR com Kraken

Os notebooks são orquestradores; pré-processamento, inferência, busca e avaliação ficam em `kraken_pipeline/*.py`. A meta é executar e inspecionar cada etapa, comparar tratamentos e registrar falhas. Não há meta de precisão mínima. A única métrica de qualidade é WER.

## Executar no Colab

1. Abra `kraken_experiments.ipynb` usando **Arquivo → Fazer upload de notebook**. Envie também o ZIP do projeto pelo painel de arquivos e extraia-o em `/content/`. A estrutura esperada é `/content/test_kraken/{kraken_pipeline,Imagens,ground_truth,configs}` com `requirements.txt` na raiz.
2. Antes da primeira célula do projeto, execute `%cd /content/test_kraken`. Se o ZIP estiver em `/content`, pode extraí-lo com `!unzip -q /content/test_kraken_colab_ready.zip -d /content`.
3. Execute as células em ordem. A primeira instala as dependências fixadas. Se indicar versões binárias antigas carregadas, use **Ambiente de execução → Reiniciar sessão** e execute novamente desde o início. Os arquivos permanecem no runtime.
4. O modelo é baixado e verificado dentro do Colab; não envie pesos, `.venv`, `outputs/` ou o link `vault/`. GPU é recomendada, mas há seleção automática de CPU quando CUDA não está disponível. CPU pode demorar bastante.
5. Por padrão, o principal usa todas as 12 páginas. Para conferir rapidamente a execução, altere `selected_images` para `[images[6]]` (imagem 11). Não confunda esse teste com uma campanha no conjunto todo.

O notebook `kraken_tuning.ipynb` é independente: abra-o em outra aba quando quiser executar a busca. Cada sessão precisa dos arquivos, instalação e inicialização; executar um notebook não inicializa o outro. Não é necessário manter os dois em execução simultânea. Para transferir o ajuste, baixe o `best.json`, envie-o para `configs/best_preprocessing.json` na sessão do principal e execute a última etapa. O notebook inicia com `BEST_RESULTS=None`. Use somente um perfil de nova busca neural; perfis legados antigos são rejeitados e não entram no ZIP atualizado.

## Etapas e controles

- `raw`: pixels originais → segmentação neural → reconhecimento.
- `kraken_neural` (baseline principal): original → nlbin nativo, limiar 0,5 → segmentação neural (`segment -bl`) → reconhecimento. A segmentação neural é nativa do Kraken. O pipeline não oferece execução legada. `KRAKEN_BASELINE` é somente um alias Python deste mesmo controle; não execute ambos como variantes separadas.
- Individuais: grayscale, contraste global, CLAHE, denoise, normalização de iluminação, deskew, dewarp, ampliação e Otsu. Cada tratamento recomeça na imagem original e termina em nlbin + segmentação neural.
- Combinações: operações aplicadas na ordem das tuplas em `config.py`, com o mesmo reconhecedor e segmentador dos tratamentos individuais.
- Perfil ajustado: parâmetros, operações e segmentador explicitamente exportados pela busca. Não altera silenciosamente o controle fixo.

Os controles são processados uma vez. `compare_runs()` reúne etapas verificando hashes de dados, pesos e código, política de pontuação e semente. `show_stages()` mostra original, operações intermediárias e overlay da segmentação em cada página selecionada, para cada variante, pelos laços do notebook. O texto predito aparece aberto logo abaixo dessas imagens; textos vazios e arquivos ausentes ficam identificados. `show_transcriptions()` mostra GT e OCR de todas as páginas. `export_report()` salva `report.html` com textos e imagens; preserve a pasta completa para abrir suas imagens relativas. Todas as páginas selecionadas mantêm seus artefatos; exibir todas as etapas pode gerar um notebook grande.

Os métodos têm limites: ampliação não recupera informação perdida; contraste pode apagar tinta; deskew pode seguir pautas; dewarp estima curvatura compartilhada a partir de baselines e não corrige toda perspectiva ou toda linha individual. Esses limites são parte do experimento.

## Modelo e avaliação

Padrão: **PP-OCRv6 medium**, aproximadamente 64 MB, [Zenodo 21788410](https://zenodo.org/records/21788410). Inclui português no treinamento; não é apresentado como especializado em pt-BR. O helper `download_multilingual_model(variant="medium")` verifica MD5 e registra DOI, URL e SHA-256. `variant="small"` usa o [modelo small](https://zenodo.org/records/21788405), aproximadamente 13 MB. TraPrInq V1 (aproximadamente 160 MB) continua disponível por `download_portuguese_model()`, com treinamento em documentos da Inquisição portuguesa. Os pesos não são treinados durante a busca.

Nas mesmas três páginas de ajuste, raw + neural teve WER de 80,82% (TraPrInq), 68,65% (small) e 65,45% (medium). Medium foi escolhido nesse recorte; isso não estabelece superioridade geral. **Resultado histórico, não atribuível ao fluxo atual:** na validação das 12 páginas, medium + ampliação 2× + nlbin 0,5 + legacy produziu texto em 12/12 e WER de **59,73%**. A baseline legada sem ampliação ficou vazia nas 12 páginas, WER **100% de erro**. Não se apresenta saída não vazia como transcrição correta.

WER = (substituições + deleções + inserções) / palavras da referência. O padrão preserva caixa, acentos e pontuação, normalizando Unicode em NFC e separando palavras por espaços. WER pode exceder 100%. O resultado do conjunto soma erros e palavras; não é a média simples das porcentagens das páginas. Texto vazio gera deleções, não acerto. Falhas ou referências ausentes impedem uma pontuação completa. `empty_output`, contagem de linhas e páginas com texto são diagnósticos de funcionamento, sem substituir WER.

O GT foi extraído do DOCX fornecido, um `.txt` por imagem. Continua provisório até revisão integral; veja `ground_truth/README.md`. Não ajuste o GT para aproximá-lo do OCR.

## Busca reproduzível

O tuning divide com seed 42: nove páginas de ajuste e três reservadas. A primeira grade contém 17 candidatos: nlbin 0,2/0,35/0,5/0,65/0,8; contraste 1,2/1,8 × limiar 0,4/0,6; denoise 3/12 × limiar 0,4/0,6; ampliação 1,5/2 × limiar 0,4/0,6. São 171 inferências incluindo os dois controles nas nove páginas. A grade de recuperação legada foi removida. Combinações opcionais usam busca aleatória. `SINGLE_SPACES` oferece também CLAHE, iluminação, deskew e dewarp.

Para testar a execução com menor custo, use `method="random", n_trials=1` e reduza explicitamente `tuning_pages`. Isso valida a execução desse orçamento, sem esgotar a grade. `random` amostra sem reposição; `grid` exige orçamento para toda a grade. Falhas/incompletos são excluídos da seleção. A imagem original pode vencer a busca neural; a baseline permanece em empates. `select_search_result()` compara resultados compatíveis pelo WER de ajuste; somente perfis neurais são aceitos.

A avaliação reservada fica desativada por padrão (`RUN_HELDOUT=False`). As três páginas já foram consultadas em campanhas anteriores: novas avaliações são exploratórias, não um teste final intocado. Não existe evidência de generalização ampla com esse conjunto pequeno do mesmo livro.

## Ambiente, erros e armazenamento

`requirements.txt` fixa o commit Kraken `bb487219e7d8289274443d1ad0989efb4924dc2a` e dependências diretas para Colab. `requirements.lock.txt` é um registro histórico macOS, não um lock do Colab. Cada execução grava versões efetivas, configuração, sementes e hashes. Nenhum arquivo upstream foi alterado.

Problemas identificados incluem pacote Kraken ausente, mistura de NumPy/Pillow/matplotlib carregados antes de instalar, seleção indevida de GPU, incompatibilidade de Transformers e polígonos inválidos. O setup instala/verifica versões e exige reinício quando necessário; o engine valida o dispositivo; falhas são registradas antes da visualização. Kraken conserva seu comportamento nativo de avisar e omitir polígonos inválidos: o texto omitido conta como erro. A saída legada vazia foi reproduzida na proteção de excesso de componentes conectados antes da extração de linhas; limiar baixo e ampliação foram testados como remediações explícitas.

Cada pasta em `outputs/` guarda configuração, resultados JSON/CSV, imagens intermediárias, segmentação, overlay, reconhecimento e transcrição por página. Modelos, dados e resultados são ignorados pelo Git e devem ser preservados separadamente. Downloads ficam em `models/`; a instalação de dependências pode ocupar muito mais espaço que os pesos. O ZIP de envio exclui ambientes e pesos.

A documentação metodológica está em `vault/00_index.md`; a nota 12 registra a migração para segmentação exclusivamente neural; a nota 10 preserva a validação anterior e a nota 11 registra a investigação atual, decisões, erros, modelos e escopo real da execução. Evidência bruta anterior: `outputs/colab_validation/`; investigação atual: `outputs/colab_repair/`. Sucesso de instalação ou célula sem exceção não prova reconhecimento correto.

## Evidência histórica anterior à migração neural

A versão foi entregue após o pedido de encerrar a exploração e preparar o material. Verificação concluída: 43 testes no Colab; oito células do notebook principal executadas na imagem 11, sem outputs de erro e com 29 figuras; execução separada de baseline/variante ampliada nas 12 páginas, com WER refeito a partir dos arquivos arquivados. O HTML dessa execução está em `outputs/colab_repair/kraken_experiments_executed.html` no workspace local e no pacote de evidências separado.

A validação reduzida do notebook de tuning foi iniciada, mas seu término não foi confirmado antes da entrega. A grade neural de 17 candidatos também não foi integralmente confirmada. O perfil daquela entrega provinha da grade legada de quatro candidatos concluída nas nove páginas de ajuste. Ele foi excluído do novo pacote e não deve ser reutilizado no fluxo neural. A aprovação para cancelar a busca longa foi recebida, mas a revisão automática bloqueou o sinal ao kernel; não se afirma que a execução remota tenha sido cancelada. Nenhuma dessas pendências muda os resultados concluídos acima.

## Verificação da versão somente neural

A engine chama exclusivamente o segmentador neural nativo. Busca e avaliação reservada usam apenas raw e baseline neural; importar perfil legado gera erro claro. Após esta mudança, 41 testes locais passaram e duas verificações de integração foram desmarcadas porque requerem Kraken instalado, ausente localmente. As células de ambos os notebooks compilam. Não foi executada nova campanha OCR no Colab nesta alteração; os 43 testes e escores remotos anteriores são evidência histórica.

Atualize os `.py` e os dois notebooks no Colab, reinicie a sessão para descarregar módulos antigos e execute desde a primeira célula. Isso não exige reenviar imagens ou baixar novamente pesos já presentes. Gere novo `best.json` neural antes de ativar `BEST_RESULTS`.
