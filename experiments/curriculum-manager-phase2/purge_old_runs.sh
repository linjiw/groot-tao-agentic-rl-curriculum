#!/bin/bash
# Reclaim /workspace disk for the G-series by dropping REGENERABLE checkpoints
# from OLD completed runs (doc 10 E0 hygiene, generalized). DRY-RUN BY DEFAULT.
#
# What it drops:  last.pt and model_step_*.pt  (regenerable / redundant with
#                 the eval-required snapshot_*.pt kept beside them).
# What it KEEPS:  snapshot_*.pt (rollback + re-eval points), every *.json
#                 (metrics_eval, journals), config.yaml / meta.yaml, eval dirs.
#
# Scope: only run-dir prefixes listed in PURGE_PREFIXES (default = the v2 smoke
# runs, already superseded by v3/v4/E1). It NEVER touches the current baseline,
# the v4 journals' source dirs unless you add them, or anything outside
# /workspace/wbc-training-logs.
#
# Some files are root-owned (training ran in the container as root); this
# script uses `docker exec isaac-lab-base rm` for those, matching the E0
# root-owned-file pattern already proven in the driver.
#
# Usage:
#   bash purge_old_runs.sh                 # DRY RUN: list what WOULD be freed
#   bash purge_old_runs.sh --apply         # actually delete (asks once)
#   PURGE_PREFIXES="smoke_manager smoke_control" bash purge_old_runs.sh --apply
set -u
BASE_DIR="/workspace/wbc-training-logs"
CONTAINER="isaac-lab-base"
# default: the July-1 v2 smoke runs (28 GB), fully superseded. Add more
# prefixes explicitly — this script will not guess which results matter.
PURGE_PREFIXES="${PURGE_PREFIXES:-smoke_manager smoke_control}"
APPLY=0
for a in "$@"; do case "$a" in
  --apply) APPLY=1 ;;
  --help|-h) sed -n '2,28p' "$0"; exit 0 ;;
  *) echo "unknown arg: $a" >&2; exit 2 ;;
esac; done

echo "=== purge_old_runs (apply=$APPLY) — prefixes: $PURGE_PREFIXES ==="
echo "KEEP: snapshot_*.pt, *.json, *.yaml, eval dirs.  DROP: last.pt, model_step_*.pt"
total=0
files_list="$(mktemp)"
for pref in $PURGE_PREFIXES; do
  d="$BASE_DIR/$pref"
  [ -d "$d" ] || { echo "  (skip: $d not found)"; continue; }
  # collect regenerable checkpoints under this prefix
  while IFS= read -r f; do
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    total=$((total + sz))
    echo "$f" >> "$files_list"
  done < <(find "$d" \( -name "last.pt" -o -name "model_step_*.pt" \) -type f 2>/dev/null)
done
n=$(wc -l < "$files_list")
awk -v n="$n" -v t="$total" 'BEGIN{printf "would free: %d files, %.1f GB\n", n, t/1e9}'

if [ "$APPLY" != "1" ]; then
  echo "--- sample (first 10) ---"; head -10 "$files_list"
  echo "DRY RUN — re-run with --apply to delete. rm -f "$files_list" when done reviewing."
  exit 0
fi

echo "APPLYING in 5s (Ctrl-C to abort)…"; sleep 5
freed=0
while IFS= read -r f; do
  if rm -f "$f" 2>/dev/null; then
    freed=$((freed+1))
  else
    # root-owned -> delete inside the container (same path, bind mount)
    docker exec "$CONTAINER" rm -f "$f" 2>/dev/null && freed=$((freed+1)) \
      || echo "  could not remove: $f" >&2
  fi
done < "$files_list"
rm -f "$files_list"
echo "removed $freed/$n files. Now free:"; df -h "$BASE_DIR" | tail -1
