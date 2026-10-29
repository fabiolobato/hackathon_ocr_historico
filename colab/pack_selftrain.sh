#!/bin/bash
# Bundle for colab/self_train.py: code, labelling config, the lexicon/LM it names, cropped extra pages of the book.
# usage: colab/pack_selftrain.sh CONFIG IMAGES [OUT_ZIP]   (run from the repository root)
set -euo pipefail
CONFIG=$1; IMAGES=$2; ZIP=${3:-outputs/selftrain.zip}
NAMED=$(.venv/bin/python -c "import json,sys; p=json.load(open('$CONFIG'))['post']; \
print(' '.join(v for k in ('lexicon','period_lexicon','context_lm','edit_costs') if (v:=p.get(k))))")
rm -f "$ZIP"; mkdir -p "$(dirname "$ZIP")"
zip -qr "$ZIP" ocr_pipeline -x '*/__pycache__/*'
zip -q "$ZIP" "$CONFIG" requirements.txt $NAMED
zip -qr "$ZIP" "$IMAGES"
ls -la "$ZIP"; unzip -l "$ZIP" | tail -1
