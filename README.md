# Demonstração ao vivo: como rodar

Este ramo (`demo-ao-vivo`) é a demonstração ao vivo do projeto: uma folha escrita à caneta é fotografada com o celular e lida pelos dois OCRs do projeto, o pipeline local e o Gemini 3.8 Flash; o resultado é um cartão HTML em slides para pôr no projetor. O projeto em si está descrito no [README do ramo `main`](https://github.com/fabiolobato/hackathon_ocr_historico/blob/main/README.md). Aqui, só o passo a passo para rodar.

## O que você vai precisar

- **Uma conta Google** (para rodar no Colab, sem instalar nada) **ou** um computador com Python 3.10 a 3.12, Git e VS Code (para rodar localmente).
- **Uma chave do OpenRouter**, só para a leitura pelo LLM, que é paga (centavos por foto). Sem chave, dá para rodar só o pipeline.
- **Um celular** para fotografar a folha e um jeito qualquer de passar a foto para o computador (cabo, WhatsApp Web, Drive, e-mail).

Dica para a foto: folha em pé (retrato), fotografada de cima, preenchendo a tela, sem a sombra do celular sobre o texto.

## Caminho 1: no Colab

1. **Abra o caderno** por este link, que já pega a versão atual do ramo:
   `https://colab.research.google.com/github/fabiolobato/hackathon_ocr_historico/blob/demo-ao-vivo/demo_ao_vivo.ipynb`
   (ou *Arquivo → Abrir notebook → GitHub*, repositório `fabiolobato/hackathon_ocr_historico`, ramo `demo-ao-vivo`). Não abra cópias antigas da lista "Recentes".
2. **Ligue a GPU**: *Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU → Salvar*.
3. **Guarde a chave do OpenRouter**: clique no **ícone de chave** na barra lateral esquerda → *Adicionar novo segredo* → em **Nome** escreva `API_KEY` → em **Valor** cole a chave → ligue o botão **Acesso do notebook** ao lado dela. A chave fica na sua conta Google, não no caderno.
   Sem chave: na célula **Os dois OCRs**, troque `PERMITIR_GASTO = True` por `False` (o LLM só simula).
4. **Rode a célula "Ambiente"** (botão ▶ à esquerda da célula). Demora de 3 a 6 minutos na primeira vez. No fim devem aparecer `GPU do Colab: Tesla T4`, `CUDA disponível: True` e `pasta de trabalho: /content/hackathon_ocr_historico`.
5. **Rode a célula "Os dois OCRs"** (10 a 30 s). Deve imprimir `dispositivo: cuda:0 (Tesla T4)` e `OCRs: configs/final_v6.json + google/gemini-3.8-flash (envio autorizado)`.
6. **Envie a foto**: rode a célula **A foto**; aparece o botão *Escolher arquivos*; escolha a foto (JPEG ou HEIC). Deve imprimir `recebida: dados/demo/<nome da foto>`.
7. **Rode a célula "Ler a foto"** (10 a 60 s). Ela imprime as duas leituras e **baixa um arquivo `.html`** para o seu computador (pasta de downloads do navegador).
8. **Abra o `.html` no navegador** (clique duplo). Esse é o cartão: leve essa aba ao projetor (F11 para tela cheia; rolar passa de slide em slide; o botão **tema** no canto alterna claro e escuro).

Para outra foto, repita os passos 6 a 8. Cada foto lida com `PERMITIR_GASTO = True` é uma requisição paga ao OpenRouter.

## Caminho 2: no VS Code, no seu computador

1. **Instale** Python 3.10 a 3.12, Git e VS Code com as extensões **Python** e **Jupyter** (da Microsoft).
2. **Clone o ramo** num terminal:
   ```bash
   git clone -b demo-ao-vivo https://github.com/fabiolobato/hackathon_ocr_historico.git
   cd hackathon_ocr_historico
   ```
3. **Crie o ambiente e instale as dependências** (alguns minutos):
   ```bash
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt -r requirements-demo.txt jupyter    # Windows
   .venv/bin/pip install -r requirements.txt -r requirements-demo.txt jupyter        # Linux ou macOS
   ```
   Se o computador tiver GPU NVIDIA, instale antes o torch com CUDA com o comando `pip` que o site pytorch.org gera para a sua máquina (usando o `pip` do `.venv`), e só depois os requisitos acima. Sem GPU também funciona, só demora mais (cerca de um minuto por foto).
4. **Guarde a chave do OpenRouter**: crie um arquivo chamado `.env` na raiz da pasta (ao lado de `demo_ao_vivo.ipynb`) com uma única linha, sem aspas e sem espaços:
   ```
   OPENROUTER_API_KEY=cole_a_sua_chave_aqui
   ```
   Esse arquivo é ignorado pelo git e nunca sai do seu computador. Sem chave: na célula **Os dois OCRs**, troque `PERMITIR_GASTO = True` por `False`.
5. **Abra a pasta no VS Code** (*Arquivo → Abrir pasta*), abra `demo_ao_vivo.ipynb` e, no canto superior direito do caderno, clique em **Selecionar kernel → Ambientes Python → `.venv`**.
6. **Rode a célula "Ambiente"** (só confere a pasta) e depois **"Os dois OCRs"** (10 a 20 s). Deve imprimir o dispositivo (`cuda:0 (nome da GPU)` ou `cpu`) e `OCRs: configs\final_v6.json + google/gemini-3.8-flash (envio autorizado)`.
7. **Passe a foto para a pasta `dados/demo/`** do projeto. Pelo cabo: no Android, toque na notificação USB e escolha *Transferência de arquivos*; no Explorador de Arquivos, *Este Computador → celular → DCIM → Camera*, e arraste a foto para `dados/demo/`. No iPhone, toque em *Permitir* no aparelho e vá em *Apple iPhone → Internal Storage → DCIM*. Qualquer outro caminho (WhatsApp Web, Drive, e-mail) serve, desde que a foto termine dentro de `dados/demo/`.
8. **Rode a célula "Ler a foto"** (10 a 60 s com GPU). Ela imprime as duas leituras e **abre o cartão numa aba do navegador**. Leve essa aba ao projetor (F11 para tela cheia).

Para outra foto, repita os passos 7 e 8: a célula sempre lê a foto mais recente de `dados/demo/`. Se mudar algo no código, use *Reiniciar o kernel* antes de rodar de novo.

## As três chaves da célula "Os dois OCRs"

| variável | padrão | efeito |
|---|---|---|
| `USAR_LLM` | `True` | o Gemini lê a mesma foto ao lado do pipeline; `False` roda só o pipeline |
| `PERMITIR_GASTO` | `True` | envia a foto ao OpenRouter (pago; exige a chave); `False` deixa o LLM em simulação, nada é enviado |
| `EXEMPLO_LLM` | `False` | com `PERMITIR_GASTO = False`, o cartão do LLM mostra um texto de exemplo (marcado "exemplo") para ensaiar a página sem gastar |

## Pela linha de comando (opcional)

Na raiz do projeto, com o `.venv` ativado e a foto já em `dados/demo/`:

```bash
python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg --permitir-gasto --abrir   # pipeline + Gemini (pago); o cartão abre no navegador
python -m ocr_pipeline.demo --foto dados/demo/IMG_1234.jpg --sem-llm --abrir          # só o pipeline
python -m ocr_pipeline.demo --vigiar --abrir                                           # cada foto nova em dados/demo/ abre um cartão
```

## Onde fica cada coisa

- `demo_ao_vivo.ipynb`: o caderno. `ocr_pipeline/demo.py`: a lógica da demonstração. `docs/DEMO.md`: o roteiro da apresentação, com o que cada slide mostra.
- `dados/demo/`: as fotos (ficam fora do git).
- `outputs/demo/cartoes/`: os cartões `.html` gerados (fora do git); `outputs/demo/textos/`: os textos lidos.
