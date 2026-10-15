# Perfis de ajuste

O pipeline atual aceita somente segmentação neural. O arquivo local `best_preprocessing.json` de campanhas anteriores seleciona legado: foi preservado como histórico, mas não é distribuído no ZIP nem aplicado pelo notebook. Não troque apenas seu campo `segmenter`: os escores deixariam de corresponder ao método executado. Gere um novo `best.json` usando `kraken_tuning.ipynb`, envie-o para esta pasta e configure `BEST_RESULTS` explicitamente.
