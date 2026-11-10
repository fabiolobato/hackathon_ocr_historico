# Demonstração: do papel ao texto, pelos dois OCRs do projeto

Alguém escreve à caneta numa folha, a folha é fotografada com o celular, a foto vai para o computador e é lida pelos **dois OCRs do projeto**, sempre os mesmos, sem nada para escolher:

- **pipeline**: o pipeline final, local, sem LLM (`configs/final_v6.json`). A foto entra na mesma configuração que transcreve o livro, sem nenhum ajuste, e o cartão é o de `novas_imagens.ipynb`: operações de imagem, segmentação, texto depois de cada etapa e, linha a linha, o recorte da imagem ao lado do texto lido.
- **LLM**: o melhor LLM da avaliação do projeto (`Transcription_Evaluation.ipynb`, seção "Ranking"): **Gemini 3.8 Flash** com esforço de raciocínio `low`, que deu WER 11,8 % / CER 4,0 % nas páginas pré-processadas e 12,3 % / 4,3 % nas páginas recortadas. Aqui ele lê a própria foto. O envio é pelo motor `ocr_historico/motor_llm.py` (OpenRouter), com o prompt de transcrição do projeto. **É pago**: nada é enviado sem autorização explícita.

A única coisa que acontece com a foto antes dos OCRs é a **redução de escala**: o pipeline recebe a foto inteira com o lado maior em 880 px, a escala das 12 páginas do livro (~650 × 880); o LLM a recebe com o lado maior em 1600 px. Nada é recortado nem endireitado, então a folha deve preencher o quadro.

O cartão abre com **uma página do livro** e a pergunta "você consegue ler esta página?": dois botões revelam o que o pipeline final leu nela e a transcrição de referência, e um terceiro destaca as palavras que diferem. Depois vêm, nesta ordem: a foto e as duas entradas, as leituras lado a lado, o cartão completo do pipeline (suas operações de imagem, a segmentação, o texto por etapa, linha a linha) e a seção do LLM (prompt e resposta). Para a folha fotografada não há referência nem WER/CER: ninguém precisa anotar o que foi escrito.

A interface é o caderno `demo_ao_vivo.ipynb`, na raiz do repositório; a lógica está em `ocr_pipeline/demo.py`. As fotos usadas ficam em `dados/demo/`, dentro do projeto mas fora do git (só o README da pasta é versionado). Nada disso faz parte de `entrega_ocr/`.

## No VS Code

1. Abra `demo_ao_vivo.ipynb` com o kernel do `.venv` do projeto (no Windows, `.venv\Scripts\python.exe`).
2. Rode as células **Ambiente** (só confere a pasta) e **Os dois OCRs** (carrega os modelos do pipeline, 10 a 20 s). `USAR_LLM = True`, o padrão, faz o Gemini ler a mesma foto ao lado do pipeline (`False` roda só o pipeline); `PERMITIR_GASTO = False`, o padrão, o deixa em simulação, e `True` envia de verdade (pago). Para ensaiar a página sem gastar, `EXEMPLO_LLM = True` (com `PERMITIR_GASTO = False`) põe no cartão do LLM um texto de exemplo, marcado "exemplo", no lugar da marca de simulação.
3. Fotografe a folha de cima, preenchendo o quadro, sem a sombra do celular sobre o texto. Conecte o celular pelo cabo:
   - **Android**: escolha "transferência de arquivos" na notificação USB. No Explorador de Arquivos o celular aparece em "Este Computador"; a foto está em `DCIM\Camera`. Arraste-a para `dados/demo/`.
   - **iPhone**: toque em "Permitir" no celular; no Explorador, `Apple iPhone\Internal Storage\DCIM\...`. Fotos HEIC abrem do mesmo jeito (pillow-heif).
4. Rode a célula **Ler a foto**. Ela pega a foto mais recente de `dados/demo/`, imprime a leitura de cada OCR no caderno e abre o cartão completo numa aba do navegador (o arquivo fica em `outputs/demo/cartoes/<id>.html`). Deixe o navegador na tela do projetor e o caderno na sua.

Tempos medidos: uma folha com poucas linhas leva uns 6 a 14 s no pipeline com GPU; uma página inteira do livro, 40 a 80 s (a parte lenta é a decodificação do texto, em CPU). O LLM roda em paralelo com o pipeline.

## No Colab

Abra o caderno pelo GitHub (**Arquivo → Abrir notebook → GitHub**, repositório `fabiolobato/hackathon_ocr_historico`, ramo `demo-ao-vivo`), escolha um ambiente com GPU e rode as células em ordem. A célula **Ambiente** clona esse ramo e instala o Kraken e as dependências (alguns minutos; os modelos do pipeline vêm no clone). A foto entra pela célula de upload, que abre o seletor de arquivos e a guarda em `dados/demo/` do clone. Para o LLM, guarde a chave do OpenRouter no segredo `API_KEY` do Colab (ícone de chave na barra lateral). Como o navegador não enxerga os arquivos da máquina virtual do Colab, a célula **Ler a foto** baixa o cartão HTML para o seu computador: abra-o no navegador e deixe essa aba no projetor.

O ramo apontado em `REPOSITORIO`/`RAMO`, no início da célula **Ambiente**, precisa ter os arquivos da demo (a célula avisa se não tiver). O passo a passo curto, para quem só quer rodar, está em [`README_DEMO.md`](../README_DEMO.md). O caminho do Colab não foi executado neste computador, só o do VS Code.

## Pela linha de comando (alternativa)

```bash
python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg --sem-llm --abrir  # só o pipeline; o cartão abre no navegador
python -m ocr_pipeline.demo --vigiar --abrir                                  # dados/demo/: cada foto nova abre um cartão
python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg --permitir-gasto   # pipeline + LLM de verdade (pago)
python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg --exemplo-llm --abrir  # ensaio: cartão do LLM com texto de exemplo, nada enviado
```

`python -m ocr_pipeline.demo --help` lista as opções.

## A abertura: uma página do livro

A seção 1, "Você consegue ler esta página?", mostra a **página 10** do Livro de Registro de Leis nº 1 (a foto como veio da Fundação, `dados/Imagens Originais/10.…jpg`, 692×1024 px) ao lado de três cartões cobertos. A página é uma das três reservadas (7, 10 e 12), que não entraram em nenhuma escolha do pipeline. Clique na imagem para ampliá-la com o visualizador e deixe a plateia tentar ler; depois:

- **Revelar a leitura do pipeline** (azul) mostra o texto que o pipeline final produziu para essa página na entrega (`entrega_ocr/resultados/textos/pagina_10.txt`, `final_v6` executado para a entrega). Nada é calculado na hora.
- **Revelar a leitura do LLM** (âmbar) mostra o que o Gemini 3.8 Flash em esforço `low` leu na mesma página durante a avaliação do projeto, sobre a página pré-processada (`dados/LLM/imagens-pré-processadas-gemini-3.8-flash.json`, a melhor execução de `Transcription_Evaluation.ipynb`). O rodapé do cartão traz o custo que o OpenRouter cobrou por essa página (US$ 0,0020); o do pipeline diz "custo por página: nenhum".
- **Revelar a transcrição real** (verde) mostra a transcrição de referência da página (`entrega_ocr/ground_truth/10.…txt`).
- **Destacar diferenças**, liberado quando a referência e pelo menos um OCR estão abertos, marca as palavras que não batem, com a mesma regra do WER normalizado do projeto (minúsculas, sem pontuação): em vermelho sublinhado, nos cartões dos OCRs, as palavras que cada um leu errado ou inventou; no cartão da referência, fundo azul nas palavras que o pipeline não leu e âmbar nas que o LLM não leu (os dois, metade de cada). Uma legenda aparece ao lado dos botões. Clicar de novo tira as marcas.
- **Esconder de novo** cobre os três cartões para a próxima rodada.

Os três textos aparecem como texto corrido, sem as quebras de linha de origem (o pipeline quebra por linha do manuscrito, a referência por parágrafo, o LLM não quebra), para que as três colunas, de mesma largura, quebrem nos mesmos lugares e a comparação seja lado a lado; a leitura linha a linha do pipeline continua na seção 4 e no `.txt` da entrega.

Cada botão de revelar também esconde ao segundo clique. A página usada é a constante `BOOK_PAGE` de `ocr_pipeline/demo.py` (a célula **Os dois OCRs** imprime qual é); qualquer uma das 12 serve, desde que exista o `pagina_NN.txt` da entrega (a leitura do LLM e o custo vêm do mesmo registro da avaliação, que tem as 12). Se os arquivos não forem encontrados (fora do repositório, por exemplo), o cartão começa direto na foto e o caderno avisa.

## Como a foto aparece passando pelas etapas

O cartão é uma página única, com uma barra fixa no alto (o título "Uma folha escrita à mão, dois OCRs", a legenda das duas cores com o nome de cada OCR e atalhos para cada seção) e cinco seções numeradas: a página do livro, a foto, as leituras, dentro do pipeline e o LLM. **Cada seção é um slide**: ocupa a tela inteira abaixo da barra, e rolar (roda do mouse, setas, PageDown, espaço) salta direto para a seção seguinte em vez de descer aos poucos; os atalhos da barra também levam a uma seção. O que não cabe na tela (o cartão do pipeline, as leituras longas, os textos da página do livro) rola dentro da própria seção, com o título fixo; chegando ao fim desse conteúdo, a rolagem seguinte passa para o próximo slide. Em telas estreitas (celular) e na impressão as seções voltam a ser uma página corrida. Cada OCR tem uma cor, **azul para o pipeline e âmbar para o LLM**, que aparece na moldura da sua entrada na seção 2, na etiqueta e na borda do seu cartão de leitura, no atalho da barra e na borda e no número da sua seção, para seguir cada caminho ao longo da página. A seção 2, "A foto original vira duas entradas", é um garfo: a foto original à esquerda e, saindo dela, uma seta independente para cada OCR, cada uma com a sua redução feita a partir do original (nenhuma é feita a partir da outra):

1. **foto original**: como veio do celular (girada conforme a marca de orientação do arquivo, como qualquer visualizador faz), com a resolução original, por exemplo 3312×5888 px.
2. **→ redução à escala do pipeline → entrada do pipeline**: a foto inteira com o lado maior em 880 px (495×880 px no exemplo), o tamanho das páginas do livro com que o pipeline foi ajustado.
3. **→ redução para o LLM → imagem enviada ao LLM**: a foto inteira com o lado maior em 1600 px (900×1600 px no exemplo).

A seção 3 traz as leituras em cartões grandes, lado a lado quando há dois OCRs, com linhas, palavras e tempo de cada um. A seção 4, "Dentro do pipeline", é o cartão do pipeline, o mesmo de `novas_imagens.ipynb`, com as etapas internas dele, uma imagem por operação, cada uma aplicada sobre a anterior:

- **geometria (vale para os dois fluxos)**: original → + ampliação (2×) → + correção de inclinação (melhor ângulo entre −5° e +5°).
- **imagem de segmentação**: + contraste (×1,8) → + remoção de pauta → "= segmentação: N linhas (polígono + linha de base)", a imagem com cada linha encontrada pintada em uma cor, com seu polígono, sua linha de base e o número na ordem de leitura.
- **imagem de reconhecimento**: nenhuma operação a mais; o reconhecedor lê a imagem da geometria.
- **texto depois de cada etapa**: reconhecedor (guloso) → + reescore CTC com dicionário e contexto → + convenções (nº, §) → + hifenização → + correção léxica.
- **linha a linha** (dobrável): tabela com o recorte de cada linha na imagem de reconhecimento, o que o reconhecedor leu nela e o que o reescore mudou.

A seção 5, "O LLM", mostra o tamanho da imagem enviada, o prompt e a resposta (ou a marca de simulação).

**Cores e tema.** Cada seção tem uma cor, que aparece no número, numa faixa no alto do slide e nas molduras de dentro: terracota para a página do livro, violeta para a foto, azul e âmbar divididos nas leituras, azul no pipeline e âmbar no LLM; a barra do alto tem um filete com as cinco cores, na ordem dos slides, e os atalhos são pílulas nessas cores. O fundo não é chapado: tem brilhos suaves das três cores dos OCRs nos cantos. Há dois temas, o escuro (tinta, azul profundo) e o claro (papel, creme); a página segue o tema do sistema até o botão **tema claro/escuro** da barra ser usado, e a escolha fica guardada no navegador. No projetor, o tema claro costuma render mais.

**Ampliar e comparar.** Clique em qualquer imagem da página para abri-la em tela cheia. Roda do mouse aproxima e afasta em torno do cursor; arrastar move; duplo clique aproxima; `F` ajusta à tela e `1` mostra em tamanho real. As setas `←` `→` (ou os botões "anterior" e "seguinte") percorrem os passos em sequência: a foto original e as duas entradas na seção 2, e dentro do pipeline a cadeia original → ampliação → inclinação → contraste → remoção de pauta → segmentação. O botão **comparar** (`C`) põe o passo atual e o anterior lado a lado, com o mesmo zoom e a mesma posição, e tudo o que você fizer num lado acontece no outro; **alternar** (espaço) troca a imagem atual pela anterior no mesmo lugar, para ver só o que mudou. `Esc` fecha. As figuras do pipeline são embutidas com até 1200 px de largura para o zoom mostrar o traço de verdade.

## Custo e segurança do LLM

- **Nada é enviado sem autorização.** No caderno, `PERMITIR_GASTO = False` (o padrão) deixa o LLM em simulação: a requisição é montada e gravada em `outputs/demo/llm/<id>.json` (modelo, esforço, prompt, tamanho da imagem; a imagem em si fica em `outputs/demo/entrada/<id>.llm.png`), mas não sai da máquina, e o cartão marca "simulação: nada foi enviado". Pela linha de comando, o envio exige `--permitir-gasto`.
- Com `PERMITIR_GASTO = True`, a célula **Os dois OCRs** já falha se não houver chave (`OPENROUTER_API_KEY` no ambiente, segredo `API_KEY` do Colab ou `.env` na raiz, fora do git).
- Cada foto é uma requisição paga, de centavos com o Gemini 3.8 Flash em esforço `low`; o custo que o OpenRouter informar aparece em destaque no cartão da leitura do LLM (à direita do título, em âmbar), no rodapé do cartão e na seção 5. A leitura real de 05/11/2026 custou US$ 0,0033.
- **Ensaio sem gasto.** `EXEMPLO_LLM = True` no caderno (ou `--exemplo-llm [TEXTO]` na linha de comando) mantém a simulação, mas o cartão do LLM mostra um texto de exemplo no lugar de "[simulação: nenhuma requisição enviada]", com a etiqueta vermelha **exemplo** no cartão, "exemplo: texto ilustrativo, nada foi enviado" na seção do LLM e, como custo, o da leitura real de 05/11/2026 (US$ 0,0033), identificado como custo de referência. O texto padrão (`LLM_EXAMPLE_TEXT` em `ocr_pipeline/demo.py`) é a leitura real que o Gemini fez em 05/11/2026 da folha de teste `dados/demo/IMG20261105115033.jpg`. Com `PERMITIR_GASTO = True` o exemplo é ignorado e o LLM lê de verdade.
- Os testes (`tests/test_demo.py`) nunca enviam nada: rodam em simulação ou com um transporte falso, e qualquer tentativa de requisição real faz o teste falhar.

## O que funciona melhor

O reconhecedor do pipeline foi ajustado à letra cursiva do século XIX e o dicionário e o modelo de contexto vêm de leis e livros de 1889 a 1910. Por isso:

- **letra cursiva emendada** lê melhor do que letra de forma;
- **vocabulário de época e de lei** lê melhor do que texto do dia a dia: "A Camara Municipal de São Carlos do Pinhal decreta: Art. 1.º Fica o Intendente autorisado a despender a quantia de dous contos de reis com a construcção de uma ponte". Palavras modernas ou nomes próprios raros tendem a ser "corrigidos" para uma palavra do dicionário parecida. O LLM não tem essa limitação, o que torna a comparação interessante;
- **caneta escura, folha branca ou pautada** (a remoção de pauta do pipeline apaga as linhas horizontais), linhas com espaçamento normal, 5 a 10 linhas;
- **foto de cima, folha preenchendo o quadro, sem a sombra do celular** sobre o texto. Como nada é recortado, tudo o que estiver na foto entra no OCR.

Numa primeira foto real (folha de caderno pautada preenchendo o quadro, caneta esferográfica, três linhas de texto moderno: "olá, teste teste. texto em cursivo que, com sorte, será lido pelo pipeline."), o pipeline achou as linhas e manteve a estrutura, mas trocou as palavras modernas por palavras do dicionário de época ("olá, deste leitores / em ensino que, comporte, os será lido ou / pipe."). Recortar a folha só ao redor da escrita piorou muito (o segmentador quer uma imagem com forma de página), por isso a foto inteira é que entra. O LLM só foi exercitado em simulação: nenhuma foto foi enviada.

## Onde fica tudo

As fotos ficam em `dados/demo/` (fora do git, como `outputs/`). O resto vai para `outputs/demo/` (fora do git): `entrada/` (o que cada OCR recebeu), `textos/` (`<id>.pipeline.txt`, `<id>.llm.txt`), `cartoes/` (`<id>.html`), `llm/` (requisição e resposta de cada foto, sem a imagem) e `cache/` (segmentação e reconhecimento do pipeline). Cada foto recebe um nome único, então o cache nunca confunde duas fotos nem as páginas do livro.
