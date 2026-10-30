#!/bin/bash
# Upload a large bundle to a Colab session in 30 MB parts (one-shot uploads of ~90 MB get HTTP 400), reassemble
# it on the VM and check the MD5.
# usage: colab/upload_bundle.sh SESSION [ZIP] [REMOTE]
set -euo pipefail
SESSION=$1; ZIP=${2:-outputs/selftrain.zip}; REMOTE=${3:-/content/selftrain.zip}
PARTS=$(mktemp -d)
split -b 30m "$ZIP" "$PARTS/part_"
for p in "$PARTS"/part_*; do colab upload -s "$SESSION" "$p" "/content/$(basename "$p")" > /dev/null; done
rm -rf "$PARTS"
REMOTE_MD5=$(printf 'import subprocess,hashlib\nsubprocess.run("cat /content/part_* > %s && rm /content/part_*", shell=True, check=True)\nprint(hashlib.md5(open("%s","rb").read()).hexdigest())\n' "$REMOTE" "$REMOTE" \
  | colab exec -s "$SESSION" 2>/dev/null | tail -1)
[ "$REMOTE_MD5" = "$(md5 -q "$ZIP")" ] && echo "uploaded $ZIP ($REMOTE_MD5)" || { echo "MD5 mismatch: $REMOTE_MD5"; exit 1; }
