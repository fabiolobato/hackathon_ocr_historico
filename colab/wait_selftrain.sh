#!/bin/bash
# Wait for /content/RUN/READY on Colab session SESSION, fetch result.zip, unpack it, mark FETCHED.
# usage: colab/wait_selftrain.sh SESSION RUN [MAX_MIN]
SESSION=$1; RUN=$2; MAX=${3:-360}; DEST=outputs/finetune/runs/$RUN
mkdir -p "$DEST"
for i in $(seq 1 $((MAX * 2))); do
  sleep 30
  if colab ls -s "$SESSION" "/content/$RUN" 2>/dev/null | grep -q '^READY$'; then
    colab download -s "$SESSION" "/content/$RUN/result.zip" "$DEST/result.zip" > /dev/null 2>&1 \
      && unzip -oq "$DEST/result.zip" -d "$DEST" \
      && touch "$DEST/FETCHED" && colab upload -s "$SESSION" "$DEST/FETCHED" "/content/$RUN/FETCHED" > /dev/null 2>&1
    echo "fetched $RUN after $((i / 2)) min: READY=$(cat "$DEST/READY" 2>/dev/null)"; ls "$DEST"; exit 0
  fi
done
echo "timeout waiting for $RUN"; exit 1
