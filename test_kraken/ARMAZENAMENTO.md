# Dados, modelos e resultados

O conteúdo do ZIP atualizado foi extraído nesta pasta. Para manter dados e pesos fora do Git, `Imagens/`, transcrições/DOCX/metadados de GT em `ground_truth/`, `models/`, `outputs/` e perfis JSON de ajuste ficam ignorados. As 12 imagens e o GT foram preservados no clone local ao extrair o ZIP; não são adicionados por este commit.

Após clonar em outra máquina ou no Colab, copie do pacote de dados original as pastas `Imagens/` e `ground_truth/` para esta pasta, preservando os nomes. O notebook espera um `.txt` por imagem e `ground_truth/review.json`. O pacote completo preparado para envio ao Colab é mantido separadamente, fora do Git. Não use o Git como armazenamento de pesos ou resultados.

Execute os notebooks a partir de `test_kraken/`. No Colab, use `%cd /content/hackathon_ocr_historico/test_kraken` se esse for o local do clone. Os pesos PP-OCRv6 medium são baixados automaticamente pelo notebook e verificados por checksum; não precisam ser enviados. A instalação inicial pode solicitar reinício por módulos binários antigos na memória.

Guarde os resultados de execução separadamente antes de encerrar o runtime. Cada execução registra configurações, sementes, versões, hashes, previsões, segmentação e WER. O código usa somente segmentação neural nativa do Kraken; resultados e perfis antigos com segmentação legada não representam esta versão.
