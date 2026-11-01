#!/bin/bash
# Bundle for colab/final_job.py: the tracked repository (committed state), the recognizer and lexicon/LM files the
# final configuration names, and the 10 Gemini test pages (images_extra is not in git).
# usage: colab/pack_final.sh [CONFIG] [OUT_ZIP]   (run from the repository root)
set -euo pipefail
CONFIG=${1:-configs/final_v6.json}; ZIP=${2:-outputs/final.zip}
NAMED=$(.venv/bin/python -c "import json; from ocr_pipeline.engine import MODELS; c = json.load(open('$CONFIG')); \
print(' '.join([str(MODELS[c['rec']['model']])] + [v for k, v in c['post'].items() \
if k in ('lexicon', 'period_lexicon', 'context_lm', 'edit_costs') and v]))")
rm -f "$ZIP"; mkdir -p "$(dirname "$ZIP")"
git ls-files -z | grep -zv '^entrega_ocr/' | xargs -0 zip -q "$ZIP"
zip -q "$ZIP" $NAMED
zip -qr "$ZIP" imagens_extra/teste
ls -la "$ZIP"; unzip -l "$ZIP" | tail -1
