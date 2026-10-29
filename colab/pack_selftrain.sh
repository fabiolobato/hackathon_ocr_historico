#!/bin/bash
# Bundle for colab/self_train.py: code, labelling config, the lexicon/LM it names, cropped extra pages of the book.
# usage: colab/pack_selftrain.sh CONFIG IMAGES [OUT_ZIP]   (run from the repository root)
set -euo pipefail
CONFIG=$1; IMAGES=$2; ZIP=${3:-outputs/selftrain.zip}
NAMED=$(.venv/bin/python -c "import json,sys; p=json.load(open('$CONFIG'))['post']; \
print(' '.join(v for k in ('lexicon','period_lexicon','context_lm','edit_costs') if (v:=p.get(k))))")
# a fine-tuned labelling recognizer travels in the bundle (the medium base is downloaded on the VM)
NAMED="$NAMED $(.venv/bin/python -c "import json; from ocr_pipeline.engine import MODELS; \
m=MODELS[json.load(open('$CONFIG'))['rec']['model']]; print(m if str(m).startswith('models/finetuned') else '')")"
rm -f "$ZIP"; mkdir -p "$(dirname "$ZIP")"
zip -qr "$ZIP" ocr_pipeline -x '*/__pycache__/*'
zip -q "$ZIP" "$CONFIG" requirements.txt $NAMED
zip -qr "$ZIP" "$IMAGES"
ls -la "$ZIP"; unzip -l "$ZIP" | tail -1
