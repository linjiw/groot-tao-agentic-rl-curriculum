#!/bin/bash
# G0 — library-native warm start + tier-0 insertion bit-identity gate-0.
# doc 10 §4-G0; pre-registered in G0_G1_G2_PREREG.md.
#
# Three steps, SEQUENTIAL (single shared A10G):
#   1. WARM: train ~2000 iters from scratch on robot_curriculum (seed 42).
#   2. SMOKE: from WARM's last.pt, two short runs — `stock` (no shim) and
#      `noop` (tier-0 shim inserted, SONIC_TIER0_ACTIVE=0).
#   3. GATE: check_gate0.py asserts stock ≡ noop (bit_identical). PASS ->
#      insertion is inert, proceed to G1. FAIL -> STOP (debug the shim).
#
# PREREQUISITES (this script refuses to start without them):
#   - GPU effectively free (no foreign training job; a shared box risks OOM).
#   - >= MIN_FREE_GB free on /workspace (warm start + smokes write ckpts).
#   - the tier-0 shim deployed to $TIER0_PP (copy, not symlink — I1 F10).
#
# Usage:
#   bash run_g0.sh --dry-run          # print the 3 commands, deploy shim, exit
#   bash run_g0.sh                     # run for real (long; ~2-4 GPU-h warm)
#   WARM_ITERS=2000 SMOKE_ITERS=10 bash run_g0.sh
#
# Artifacts (beside this script): g0_warm.log, g0_{stock,noop}_journal.json,
# g0_{stock,noop}_summary.json, g0_gate0.json. Container: cmp_g0_* prefixes.
set -u
cd "$(dirname "$0")"
PY="${PY:-$HOME/.local/bin/python3.10}"
RMC="../run-manager-core"
TIER0_SRC="$RMC"                         # host source of core/ + adapters/
TIER0_PP="${TIER0_PP:-/workspace/rmc_tier0}"   # container-visible deploy dir
CURRICULUM_MOTION="${CURRICULUM_MOTION:-data/motion_lib_bones_seed/robot_curriculum}"
EVAL_MOTION="${EVAL_MOTION:-data/motion_lib_bones_seed/robot_curriculum_eval64}"
BASE_DIR="/workspace/wbc-training-logs"
WARM_ITERS="${WARM_ITERS:-2000}"
SMOKE_ITERS="${SMOKE_ITERS:-10}"
NUM_ENVS="${NUM_ENVS:-256}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
TIER0_TERM="${TIER0_TERM:-tracking_anchor_pos}"
BASE_KNOBS='{"termination_threshold.anchor_pos": 0.15, "termination_threshold.ee_body_pos": 0.15, "termination_threshold.foot_pos_xyz": 0.2}'

DRY_RUN=0
for a in "$@"; do case "$a" in
  --dry-run|-n) DRY_RUN=1 ;;
  --help|-h) sed -n '2,30p' "$0"; exit 0 ;;
  *) echo "unknown arg: $a" >&2; exit 2 ;;
esac; done

# ── deploy the tier-0 shim into a container-visible dir (I1 F10) ────────
echo "=== deploy tier-0 shim -> $TIER0_PP ==="
mkdir -p "$TIER0_PP/core" "$TIER0_PP/adapters/sonic_tier0"
cp "$TIER0_SRC"/core/*.py "$TIER0_PP/core/"
cp "$TIER0_SRC"/adapters/__init__.py "$TIER0_PP/adapters/"
cp "$TIER0_SRC"/adapters/sonic_tier0/*.py "$TIER0_PP/adapters/sonic_tier0/"
echo "deployed: $(ls "$TIER0_PP"/adapters/sonic_tier0/*.py | wc -l) shim files"

# ── preflight gates ─────────────────────────────────────────────────────
preflight() {
  local fail=0
  local free_gb
  free_gb=$(df -BG --output=avail "$BASE_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -z "$free_gb" ] || [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "PREFLIGHT FAIL: only ${free_gb}G free on $BASE_DIR (need >= ${MIN_FREE_GB}G)." >&2
    echo "  free space first (purge old wbc-training-logs runs)." >&2; fail=1
  fi
  # foreign training job on the GPU?  a from-scratch warm start needs headroom.
  local foreign
  foreign=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$foreign" -gt 0 ]; then
    echo "PREFLIGHT WARN: $foreign process(es) already on the GPU:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >&2
    echo "  a from-scratch SONIC warm start (~4-6 GB) may OOM or disturb them." >&2
    echo "  proceed only if that memory + headroom is genuinely free." >&2
  fi
  return $fail
}

warm_ckpt() {  # newest last.pt under the g0_warm experiment dir
  ls -t "$BASE_DIR"/cmp_g0_warm_g0_warm*/last.pt 2>/dev/null | head -1
}

if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN: the 3 G0 steps ==="
  echo "1. WARM: $PY smoke_driver.py --arm control --segments 1 --iters $WARM_ITERS \\"
  echo "     --num-envs $NUM_ENVS --seed 42 --no-eval --project cmp_g0_warm \\"
  echo "     --base-knobs '$BASE_KNOBS' \\"
  echo "     --curriculum-motion-file $CURRICULUM_MOTION  (from scratch, no --initial-checkpoint)"
  echo "   NOTE: warm start uses a bare training launch pinned to robot_curriculum;"
  echo "   if smoke_driver requires eval wiring, launch via job_adapter.build_train_command directly."
  echo "2a. STOCK smoke: $PY smoke_driver.py --arm control --segments 2 --iters $SMOKE_ITERS \\"
  echo "      --num-envs $NUM_ENVS --seed 42 --initial-checkpoint <WARM last.pt> \\"
  echo "      --project cmp_g0_stock --base-knobs '$BASE_KNOBS' \\"
  echo "      --eval-motion-file $EVAL_MOTION --journal-out g0_stock_journal.json"
  echo "2b. NOOP smoke: (same) + --tier0 noop --tier0-term $TIER0_TERM \\"
  echo "      --tier0-pythonpath $TIER0_PP --journal-out g0_noop_journal.json"
  echo "3. GATE: $PY check_gate0.py --stock g0_stock_journal.json --noop g0_noop_journal.json"
  preflight || true
  exit 0
fi

preflight || { echo "aborting on preflight failure." >&2; exit 1; }

echo "=== $(date +%H:%M:%S) G0 step 1/3: WARM start ($WARM_ITERS iters, from scratch) ==="
"$PY" smoke_driver.py --arm control --segments 1 --iters "$WARM_ITERS" \
  --num-envs "$NUM_ENVS" --seed 42 --no-eval --project cmp_g0_warm \
  --base-knobs "$BASE_KNOBS" --curriculum-motion-file "$CURRICULUM_MOTION" \
  > g0_warm_summary.json 2> g0_warm.log || { echo "WARM failed; see g0_warm.log" >&2; exit 1; }
WCKPT="$(warm_ckpt)"
[ -n "$WCKPT" ] || { echo "no warm checkpoint found under cmp_g0_warm*; see g0_warm.log" >&2; exit 1; }
echo "warm checkpoint: $WCKPT"

echo "=== $(date +%H:%M:%S) G0 step 2a/3: STOCK smoke ($SMOKE_ITERS iters x2) ==="
"$PY" smoke_driver.py --arm control --segments 2 --iters "$SMOKE_ITERS" \
  --num-envs "$NUM_ENVS" --seed 42 --initial-checkpoint "$WCKPT" \
  --project cmp_g0_stock --base-knobs "$BASE_KNOBS" \
  --curriculum-motion-file "$CURRICULUM_MOTION" \
  --eval-motion-file "$EVAL_MOTION" --no-eval \
  --journal-out g0_stock_journal.json > g0_stock_summary.json 2> g0_stock.log \
  || { echo "STOCK smoke failed; see g0_stock.log" >&2; exit 1; }

echo "=== $(date +%H:%M:%S) G0 step 2b/3: NOOP smoke (tier-0 inert) ==="
"$PY" smoke_driver.py --arm control --segments 2 --iters "$SMOKE_ITERS" \
  --num-envs "$NUM_ENVS" --seed 42 --initial-checkpoint "$WCKPT" \
  --project cmp_g0_noop --base-knobs "$BASE_KNOBS" \
  --curriculum-motion-file "$CURRICULUM_MOTION" \
  --eval-motion-file "$EVAL_MOTION" --no-eval \
  --tier0 noop --tier0-term "$TIER0_TERM" --tier0-pythonpath "$TIER0_PP" \
  --journal-out g0_noop_journal.json > g0_noop_summary.json 2> g0_noop.log \
  || { echo "NOOP smoke failed; see g0_noop.log" >&2; exit 1; }

echo "=== $(date +%H:%M:%S) G0 step 3/3: GATE-0 bit-identity check ==="
"$PY" check_gate0.py --stock g0_stock_journal.json --noop g0_noop_journal.json \
  | tee g0_gate0.json
RC=${PIPESTATUS[0]}
if [ "$RC" -eq 0 ]; then
  echo "=== G0 PASS: tier-0 insertion is inert. Proceed to G1. ==="
else
  echo "=== G0 FAIL: insertion perturbs training. STOP — debug before G1/G2. ===" >&2
fi
exit "$RC"
