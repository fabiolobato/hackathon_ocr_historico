#!/bin/bash
# Wait until /content/RUN/READY exists on the Colab VM, then fetch results; usage: wait_ready.sh RUN [MAX_MIN]
RUN=$1; MAX=${2:-120}; DEST=outputs/finetune/runs/$RUN
mkdir -p $DEST
for i in $(seq 1 $((MAX * 2))); do
  sleep 30
  if colab ls -s ft /content/$RUN 2>/dev/null | grep -q '^READY$'; then
    for f in result.zip metrics.json train.log cmd.txt files.txt READY; do
      colab download -s ft /content/$RUN/$f $DEST/$f > /dev/null 2>&1
    done
    echo "fetched $RUN after $((i / 2)) min"; ls -la $DEST; exit 0
  fi
done
echo "timeout waiting for $RUN"; exit 1
