#!/bin/bash
# ON-vs-OFF comparison v4 (multi-seed, longer horizon): for each seed run
# control then manager SEQUENTIALLY (single 23GB A10G — never parallel).
# Protocol is v3's (see run_comparison_v3.sh / COMPARISON_V3_RESULTS.md):
# per-segment im_eval both arms at fixed relaxed thresholds, eval-side
# tripwire, --base-knobs registry seeding, resume from the overnight
# baseline's model_step_002000.pt (live region of the eval curve).
#
# E1 (scripted ablation arm): ARMS=scripted replays the fixed v4 manager
# knob ladder open-loop (smoke_driver.py --arm scripted) with the SAME
# baseline checkpoint, seeds, segment count and eval config as v4, so its
# runs are directly comparable to the four existing v4 journals. Its
# artifacts carry a v5 suffix (scripted_journal_v5_seed42.json etc.) so
# nothing v4 is ever overwritten.
#
# Usage:
#   bash run_comparison_multiseed.sh                 # all seeds, both arms
#   bash run_comparison_multiseed.sh --dry-run       # print commands only
#   SEEDS="42 1337 7" SEGS=12 bash run_comparison_multiseed.sh
#   ARMS="manager" bash run_comparison_multiseed.sh  # subset of arms
#   ARMS="scripted" bash run_comparison_multiseed.sh # E1 ablation (v5 artifacts)
#
# Journals land beside this script as {arm}_journal_v4_seed{SEED}.json
# (v5 for the scripted arm); summaries as {arm}_summary_v4_seed{SEED}.json;
# driver logs as {arm}_driver_v4_seed{SEED}.log. Container artifacts under
# /workspace/wbc-training-logs/cmp_{arm}_seed{SEED}_* (per-segment train
# logs, snapshots, eval dirs with metrics_eval.json).
#
# Wall-clock estimate: v3 measured ~80 min for 2 arms x 6 segments
# (256 envs, 50 iters, eval 64 envs) => ~6.7 min/segment. Defaults below
# (2 seeds x 2 arms x 10 segments = 40 segments) => ~4.5 h. With SEGS=12:
# ~5.4 h. Plan A10G occupancy accordingly.
set -u
cd "$(dirname "$0")"
PY="${PY:-$HOME/.local/bin/python3.10}"

# Same checkpoint as v3 (verified against run_comparison_v3.sh):
CKPT="${CKPT:-/workspace/wbc-training-logs/baseline/wbc_baseline_10k-20260701_232851/model_step_002000.pt}"
SEEDS="${SEEDS:-42 1337}"
SEGS="${SEGS:-10}"
ITERS="${ITERS:-50}"
NUM_ENVS="${NUM_ENVS:-256}"
EVAL_ENVS="${EVAL_ENVS:-64}"
ARMS="${ARMS:-control manager}"

# held-out protected metric (doc 08 §5): training pinned to the curriculum-
# only split; per-segment second eval pass on the 64-motion held-out eval
# subset (~1-2 min/pass at 64 envs; the full 12,861-key held-out dir is an
# end-of-run measurement, not a per-segment one). Set HELDOUT_MANIFEST=""
# to disable and reproduce the v3 protocol exactly.
HELDOUT_MANIFEST="${HELDOUT_MANIFEST:-heldout/manifest.json}"
HELDOUT_MOTION="${HELDOUT_MOTION:-data/motion_lib_bones_seed/robot_heldout_eval64}"
CURRICULUM_MOTION="${CURRICULUM_MOTION:-data/motion_lib_bones_seed/robot_curriculum}"
# fixed 64-motion curriculum-side eval subset (v4 post-mortem: the standard
# eval pass must be explicitly pinned or it inherits the training config's
# motion_file — with CURRICULUM_MOTION pinned that was a 116,924-motion eval)
EVAL_MOTION="${EVAL_MOTION:-data/motion_lib_bones_seed/robot_curriculum_eval64}"

# base knobs = the run's ACTUAL stock values (read from the control run's
# saved config.yaml; see COMPARISON_V3_RESULTS.md "Bugs caught" #1).
# Seeds the registry belief AND, as identical-value Hydra overrides,
# leaves the resolved config equivalent to control (prefix identity).
BASE_KNOBS='{"termination_threshold.anchor_pos": 0.15, "termination_threshold.ee_body_pos": 0.15, "termination_threshold.foot_pos_xyz": 0.2}'

DRY_RUN=0
for a in "$@"; do
  case "$a" in
    --dry-run|-n) DRY_RUN=1 ;;
    --help|-h)
      sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a (arms/seeds are set via ARMS=/SEEDS= env vars)" >&2; exit 2 ;;
  esac
done

echo "=== multiseed v4: seeds=[$SEEDS] arms=[$ARMS] segs=$SEGS iters=$ITERS envs=$NUM_ENVS eval_envs=$EVAL_ENVS dry_run=$DRY_RUN ==="

for SEED in $SEEDS; do
  for ARM in $ARMS; do
    # E1: scripted-arm artifacts are v5 (a NEW experiment; the v4 journal
    # set — control/manager x 42/1337 — must never be overwritten). The
    # cmp_{arm}_seed{SEED} container prefix already embeds the arm name, so
    # container artifacts can't collide with v4's either.
    if [ "$ARM" = "scripted" ]; then VER="v5"; else VER="v4"; fi
    # E5 (noise-floor replicates): REP=N appends a _repN suffix to all
    # artifacts (journal/summary/log + container project prefix) so exact
    # re-runs of an existing arm/seed config never overwrite replicate 1
    # (the suffix-less v4 artifacts). Config is otherwise IDENTICAL.
    REPSFX=""
    if [ -n "${REP:-}" ]; then REPSFX="_rep${REP}"; fi
    JOURNAL="${ARM}_journal_${VER}_seed${SEED}${REPSFX}.json"
    SUMMARY="${ARM}_summary_${VER}_seed${SEED}${REPSFX}.json"
    DRVLOG="${ARM}_driver_${VER}_seed${SEED}${REPSFX}.log"
    CMD=("$PY" smoke_driver.py --arm "$ARM" --segments "$SEGS" --iters "$ITERS"
         --num-envs "$NUM_ENVS" --eval-envs "$EVAL_ENVS" --seed "$SEED"
         --initial-checkpoint "$CKPT"
         --project "cmp_${ARM}_seed${SEED}${REPSFX}"
         --base-knobs "$BASE_KNOBS"
         --journal-out "$JOURNAL")
    if [ -n "$HELDOUT_MANIFEST" ]; then
      CMD+=(--heldout-manifest "$HELDOUT_MANIFEST"
            --heldout-motion-file "$HELDOUT_MOTION"
            --curriculum-motion-file "$CURRICULUM_MOTION"
            --eval-motion-file "$EVAL_MOTION")
    fi
    if [ "$DRY_RUN" = "1" ]; then
      echo "DRY: ${CMD[*]} > $SUMMARY 2> $DRVLOG"
      continue
    fi
    if [ -s "$JOURNAL" ]; then
      echo "=== SKIP seed=$SEED arm=$ARM: $JOURNAL already exists (delete to re-run) ==="
      continue
    fi
    echo "=== $(date +%H:%M:%S) seed=$SEED arm=$ARM start ==="
    "${CMD[@]}" > "$SUMMARY" 2> "$DRVLOG"
    RC=$?
    echo "=== $(date +%H:%M:%S) seed=$SEED arm=$ARM done rc=$RC ==="
    if [ $RC -ne 0 ]; then
      echo "!!! seed=$SEED arm=$ARM FAILED (rc=$RC) — continuing to next run; see $DRVLOG" >&2
    fi
  done
done
echo "=== multiseed v4 complete $(date +%H:%M:%S) ==="
