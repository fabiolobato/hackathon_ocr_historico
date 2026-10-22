#!/bin/bash
# Poll a Colab training log until it shows a terminal state or a new validation line; usage: poll_log.sh RUN [MAX_MIN]
RUN=$1; MAX=${2:-60}; LOCAL=outputs/${RUN}_train.log
for i in $(seq 1 $((MAX * 60 / 90))); do
  sleep 90
  colab download -s ft /content/$RUN/train.log $LOCAL > /dev/null 2>&1
  if tr '\r' '\n' < $LOCAL | grep -qE "Error|Traceback|max_epochs=|Trainer.fit. stopped"; then break; fi
done
tr '\r' '\n' < $LOCAL | grep -v "^\s*$" | grep -iE "val_|stopped|Error" | tail -15
