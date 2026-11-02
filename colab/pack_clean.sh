#!/bin/bash
# Bundle for colab/clean_train.py (E16). Only the 9 tuning pages of imagens/ and their references travel; held-out
# pages, Gemini test pages and their transcriptions are never packed.
# usage: colab/pack_clean.sh BRANCH [OUT_ZIP]   (BRANCH m or s; run from the repository root)
set -euo pipefail
BRANCH=$1; ZIP=${2:-outputs/clean_$BRANCH.zip}
POOL=$([ "$BRANCH" = m ] && echo imagens_extra/treino_crop || echo imagens_extra/ldl1_clean)
NAMED=$(.venv/bin/python -c "import json; print(' '.join(sorted({v for c in ('configs/v4_pass1.json', 'configs/final_v5.json') \
for k, v in json.load(open(c))['post'].items() if k in ('lexicon','period_lexicon','context_lm','edit_costs') and v})))")
TUNING=$(.venv/bin/python -c "import re; from pathlib import Path; tuning = {5, 6, 8, 9, 11, 13, 14, 15, 16}; \
print('\n'.join(str(p) for d in ('imagens', 'test_kraken/ground_truth', 'test_kraken/ground_truth_gemini') \
for p in sorted(Path(d).iterdir()) if (m := re.match(r'\d+', p.name)) and int(m[0]) in tuning))")
rm -f "$ZIP"; mkdir -p "$(dirname "$ZIP")"
zip -qr "$ZIP" ocr_pipeline -x '*/__pycache__/*'
zip -q "$ZIP" colab/eval_clean.py requirements.txt configs/v4_pass1.json configs/v4_pass1_st_c1m.json \
  configs/final_v5.json $NAMED
echo "$TUNING" | zip -q "$ZIP" -@
zip -qr "$ZIP" "$POOL" imagens_extra/transcricoes/treino
if [ "$BRANCH" = m ]; then zip -q "$ZIP" models/finetuned/st_g3m.safetensors; fi  # reference, same GPU numerics
ls -la "$ZIP"; unzip -l "$ZIP" | tail -1
