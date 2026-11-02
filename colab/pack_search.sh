#!/bin/bash
# Bundle for colab/search_job.py: code, spec, --init decisions, the recognizer weights and lexicon/LM of the initial
# configuration, the 12 page images and their GT.
# usage: colab/pack_search.sh SPEC INIT_DECISIONS BEST_CONFIG [OUT_ZIP]   (run from the repository root)
#   BEST_CONFIG: best_config.json of the --init search (names the lexicon/LM files the frozen stages keep)
set -euo pipefail
SPEC=$1; INIT=$2; BEST=$3; ZIP=${4:-outputs/search.zip}
# lexicon/LM files: those of the initial configuration and every one a spec candidate can switch to
NAMED=$(.venv/bin/python -c "import json; p=json.load(open('$BEST'))['post']; \
s=json.dumps(json.load(open('$SPEC'))); import re; \
files={v for k in ('lexicon','period_lexicon','context_lm','edit_costs') if (v:=p.get(k))}; \
files|=set(re.findall(r'models/lexicon/[^\"]+', s)); print(' '.join(sorted(files)))")
MODEL=$(.venv/bin/python -c "import json; from ocr_pipeline.engine import MODELS; \
print(MODELS[json.load(open('$SPEC'))['base']['rec.model']])")
rm -f "$ZIP"; mkdir -p "$(dirname "$ZIP")"
zip -qr "$ZIP" ocr_pipeline -x '*/__pycache__/*'
zip -q "$ZIP" "$SPEC" "$INIT" requirements.txt $NAMED "$MODEL"
zip -qr "$ZIP" imagens test_kraken/ground_truth
ls -la "$ZIP"; unzip -l "$ZIP" | tail -1
