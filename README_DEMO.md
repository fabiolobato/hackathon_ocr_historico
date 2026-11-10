# Demonstração ao vivo: do papel ao texto, pelos dois OCRs do projeto

Alguém escreve à caneta numa folha, a folha é fotografada com o celular e a foto é lida pelos **dois OCRs do projeto**, sempre os mesmos, sem nada para escolher:

| OCR | o que é | custo |
|---|---|---|
| **pipeline** (azul) | o pipeline final do projeto (`configs/final_v6.json`): roda localmente, sem LLM, na mesma configuração que transcreve o livro | nenhum |
| **LLM** (âmbar) | o melhor LLM da avaliação do projeto: **Gemini 3.8 Flash**, esforço `low`, via OpenRouter, lendo a própria foto | centavos de dólar por foto (pago) |

A única preparação da foto é a redução de escala (880 px para o pipeline, 1600 px para o LLM). O resultado é um **cartão HTML em slides**, um por tela:

1. **Você consegue ler esta página?** Uma página do livro de 1892; botões revelam o que o pipeline leu, o que o LLM leu e a transcrição de referência, e destacam as diferenças.
2. **A foto original vira duas entradas**: a foto e as duas reduções, lado a lado.
3. **As leituras** dos dois OCRs, com o custo do LLM.
4. **Dentro do pipeline**: cada operação de imagem, a segmentação, o texto depois de cada etapa, linha a linha.
5. **O LLM**: prompt e resposta.

Clique em qualquer imagem para ampliar e comparar passos; o botão **tema** alterna entre escuro e claro (papel). O roteiro completo da apresentação, com dicas de escrita e de foto, está em [`docs/DEMO.md`](docs/DEMO.md).

## Rodar no Colab

1. **Abra o caderno pelo GitHub**: no Colab, *Arquivo → Abrir notebook → GitHub*, repositório `fabiolobato/hackathon_ocr_historico`, ramo **`demo-ao-vivo`**, arquivo `demo_ao_vivo.ipynb`. Ou use o link direto:
   `https://colab.research.google.com/github/fabiolobato/hackathon_ocr_historico/blob/demo-ao-vivo/demo_ao_vivo.ipynb`
2. **GPU**: *Ambiente de execução → Alterar o tipo de ambiente de execução → T4*. Em CPU também funciona, só demora mais.
3. **Chave do OpenRouter** (só para o LLM): guarde-a no segredo `API_KEY` do Colab (ícone de chave na barra lateral) e dê acesso ao caderno. Sem chave, use `USAR_LLM = False` (só o pipeline) ou `PERMITIR_GASTO = False` (o LLM só simula).
4. Rode as células em ordem:
   - **Ambiente** clona o ramo `demo-ao-vivo`, instala o Kraken e as dependências e confere os modelos (alguns minutos na primeira vez).
   - **Os dois OCRs** carrega o pipeline e confere a chave, se o envio estiver autorizado.
   - **A foto** abre o seletor de arquivos: envie a foto do celular (JPEG ou HEIC).
   - **Ler a foto** lê a foto mais recente, imprime as duas leituras e **baixa o cartão HTML** para o seu computador; abra-o no navegador e deixe essa aba no projetor.

Se a célula **Ambiente** disser que o ramo "ainda não tem a demo", o ramo apontado em `REPOSITORIO`/`RAMO` (no início da célula) está sem os arquivos da demo: aponte para o repositório e o ramo que os têm.

## Rodar no computador (VS Code ou Jupyter)

Precisa de Python 3.10 a 3.12 e, de preferência, uma GPU NVIDIA (em CPU uma folha com poucas linhas leva cerca de um minuto).

```bash
git clone -b demo-ao-vivo https://github.com/fabiolobato/hackathon_ocr_historico.git
cd hackathon_ocr_historico
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt -r requirements-demo.txt jupyter   # Windows; no Linux/macOS, .venv/bin/pip
```

Com GPU, instale antes o `torch` com CUDA da sua máquina (instruções em pytorch.org) e depois os requisitos acima. Os modelos do pipeline já vêm no clone (`models/`).

1. Abra `demo_ao_vivo.ipynb` no VS Code com o kernel do `.venv` (ou `jupyter notebook demo_ao_vivo.ipynb`).
2. Para o LLM, crie um arquivo `.env` na raiz do repositório com `OPENROUTER_API_KEY=...` (o `.env` está no `.gitignore`), ou defina a variável de ambiente.
3. Rode **Ambiente** e **Os dois OCRs**.
4. Fotografe a folha de cima, em pé (retrato), preenchendo o quadro e sem a sombra do celular sobre o texto. Conecte o celular pelo cabo e copie a foto para `dados/demo/`.
5. Rode **Ler a foto**: o cartão abre numa aba do navegador. Se mudar o código, reinicie o kernel antes de rodar de novo.

## Os três interruptores da célula "Os dois OCRs"

| variável | padrão | efeito |
|---|---|---|
| `USAR_LLM` | `True` | o Gemini lê a mesma foto ao lado do pipeline; `False` roda só o pipeline |
| `PERMITIR_GASTO` | `True` | envia a foto ao OpenRouter (pago; a célula falha logo se não houver chave); `False` deixa o LLM em simulação, nada é enviado |
| `EXEMPLO_LLM` | `False` | com `PERMITIR_GASTO = False`, o cartão do LLM mostra um texto de exemplo (marcado "exemplo") para ensaiar a página sem gastar |

Cada foto enviada é uma requisição paga; a leitura real de teste custou US$ 0,0033 e o cartão mostra o custo que o OpenRouter informar.

## Pela linha de comando

```bash
python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg --sem-llm --abrir       # só o pipeline
python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg --exemplo-llm --abrir   # ensaio: LLM com texto de exemplo, nada enviado
python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg --permitir-gasto --abrir  # pipeline + Gemini de verdade (pago)
python -m ocr_pipeline.demo --vigiar --abrir                                        # cada foto nova em dados/demo/ abre um cartão
```

## Onde fica tudo

- `demo_ao_vivo.ipynb`: o caderno; `ocr_pipeline/demo.py`: toda a lógica da demonstração.
- `dados/demo/`: as fotos usadas (ficam fora do git; só o README da pasta é versionado).
- `outputs/demo/` (fora do git): `cartoes/<id>.html`, `textos/`, `entrada/` (o que cada OCR recebeu), `llm/` (requisição e resposta, sem a imagem) e `cache/`.
- `tests/test_demo.py`: os testes; nenhum deles envia requisição real (`pytest tests/test_demo.py`).
