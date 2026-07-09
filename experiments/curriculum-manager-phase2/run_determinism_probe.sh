# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
#!/bin/bash
# E5c determinism probe: rerun ONE segment twice from the same rep4-s7
# snapshot with an identical config, to discriminate between:
#   (a) probe_a == probe_b  -> process is deterministic given identical
#       inputs; the rep3-vs-rep4 s8 divergence must come from some input
#       string (e.g. rep-suffixed paths) leaking into the compute graph.
#   (b) probe_a != probe_b  -> sporadic infrastructure nondeterminism;
#       sigma_rep=0 conclusion must be weakened to "usually deterministic".
# Single segment, 50 iters, 256 envs, eval on -> ~12 min each, serial.
set -u
cd "$(dirname "$0")"
PY="${PY:-$HOME/.local/bin/python3.10}"
CKPT="/workspace/wbc-training-logs/cmp_control_seed42_rep4/control_s7-20260708_174150/snapshot_control_s7.pt"
BASE_KNOBS='{"termination_threshold.anchor_pos": 0.15, "termination_threshold.ee_body_pos": 0.15, "termination_threshold.foot_pos_xyz": 0.2}'
for P in a b; do
  echo "=== $(date +%H:%M:%S) probe_$P start ==="
  "$PY" smoke_driver.py --arm control --segments 1 --iters 50 \
    --num-envs 256 --eval-envs 64 --seed 42 \
    --initial-checkpoint "$CKPT" \
    --project "cmp_control_seed42_probe${P}" \
    --base-knobs "$BASE_KNOBS" \
    --journal-out "control_journal_probe_${P}.json" \
    --heldout-manifest heldout/manifest.json \
    --heldout-motion-file data/motion_lib_bones_seed/robot_heldout_eval64 \
    --curriculum-motion-file data/motion_lib_bones_seed/robot_curriculum \
    --eval-motion-file data/motion_lib_bones_seed/robot_curriculum_eval64 \
    > "control_summary_probe_${P}.json" 2> "control_driver_probe_${P}.log"
  echo "=== $(date +%H:%M:%S) probe_$P done rc=$? ==="
done
echo "=== probe complete ==="
