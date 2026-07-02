#!/bin/bash
# ON-vs-OFF comparison v3: control then manager, sequentially (single GPU).
# Both arms: 6 segments x 50 iters x 256 envs, seed 42, starting from the
# 10k baseline's model_step_002000.pt (the live region of the eval curve:
# progress_rate 0.003 @ 2k -> 0.050 @ 4k; a fresh start would leave the
# eval scoreboard degenerate-zero for the whole run).
# Per-segment im_eval at fixed relaxed thresholds after every segment.
set -u
cd "$(dirname "$0")"
PY=~/.local/bin/python3.10
CKPT=/workspace/wbc-training-logs/baseline/wbc_baseline_10k-20260701_232851/model_step_002000.pt
SEGS=6
ITERS=50

# base knobs = the run's ACTUAL stock values (read from the control run's
# saved config.yaml). Passing them (a) seeds the registry belief so a
# "one-notch" change really is one notch (v2 applied 0.15->0.35 believing
# it was 0.30->0.35), and (b) as identical-value Hydra overrides they leave
# the resolved config byte-equivalent to control (prefix identity holds).
BASE_KNOBS='{"termination_threshold.anchor_pos": 0.15, "termination_threshold.ee_body_pos": 0.15, "termination_threshold.foot_pos_xyz": 0.2}'

for ARM in "$@"; do
  echo "=== $(date +%H:%M:%S) arm=$ARM start ==="
  $PY smoke_driver.py --arm $ARM --segments $SEGS --iters $ITERS \
    --num-envs 256 --eval-envs 64 --seed 42 \
    --initial-checkpoint $CKPT \
    --project cmp_${ARM} \
    --base-knobs "$BASE_KNOBS" \
    --journal-out ${ARM}_journal_v3.json \
    > ${ARM}_summary_v3.json 2> ${ARM}_driver_v3.log
  echo "=== $(date +%H:%M:%S) arm=$ARM done rc=$? ==="
done
