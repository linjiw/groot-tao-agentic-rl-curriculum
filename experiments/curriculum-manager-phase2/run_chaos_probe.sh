# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
#!/bin/bash
# E5b chaos-floor probe, take 3. Takes 1-2 were behaviorally inert:
#  take1: anchor_pos 0.15->0.15000001 rounds back to fp32 0.15 (free bit-repro evidence).
#  take2: 0.15->0.1500001 IS fp32-distinct but threshold is a comparison knob;
#         no height_diff fell in (0.15, 0.1500001) in 50 iters -> 1250 metric
#         lines bit-identical to reference.
# take3 injects into a CONTINUOUS path: entropy_coef 0.01 -> 0.0100001
# (fp32-distinct, multiplies the loss every gradient step). Divergence of
# the 50-iter series vs the deterministic reference (rep3-s8, bit-verified)
# measures the pure chaotic divergence floor; tau gate #1 sits above it.
set -u
cd "$(dirname "$0")"
PY="${PY:-$HOME/.local/bin/python3.10}"
CKPT="/workspace/wbc-training-logs/cmp_control_seed42_rep4/control_s7-20260708_174150/snapshot_control_s7.pt"
CHAOS_KNOBS='{"entropy_coef": 0.0100001}'
echo "=== $(date +%H:%M:%S) chaos probe start ==="
"$PY" smoke_driver.py --arm control --segments 1 --iters 50 \
  --num-envs 256 --eval-envs 64 --seed 42 \
  --initial-checkpoint "$CKPT" \
  --project "cmp_control_seed42_chaos3" \
  --base-knobs "$CHAOS_KNOBS" \
  --journal-out "control_journal_chaos3.json" \
  --heldout-manifest heldout/manifest.json \
  --heldout-motion-file data/motion_lib_bones_seed/robot_heldout_eval64 \
  --curriculum-motion-file data/motion_lib_bones_seed/robot_curriculum \
  --eval-motion-file data/motion_lib_bones_seed/robot_curriculum_eval64 \
  > "control_summary_chaos3.json" 2> "control_driver_chaos3.log"
echo "=== $(date +%H:%M:%S) chaos probe done rc=$? ==="
