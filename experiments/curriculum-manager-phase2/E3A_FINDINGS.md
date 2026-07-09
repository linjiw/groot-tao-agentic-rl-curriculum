# E3(a) Findings — per-frame data availability in persisted v4 eval artifacts

Date: 2026-07-07. All inspection read-only. [verified] items include the exact
commands run.

## Verdict

**Per-frame tracking series: NO. Per-motion arrays (incl. per-motion
progress = normalized truncation length): YES.**

- No per-frame mpjpe series, per-frame survival masks, or per-frame counts
  exist anywhere in the persisted eval outputs. Each eval dir contains exactly
  one file, `metrics_eval.json`, whose finest granularity is **per-motion**
  (length-64 arrays / dicts).
- Per-motion **truncation lengths are recoverable at 0 GPU-h**: each
  `metrics_eval.json` carries `eval/all_metrics_dict.progress` (length-64,
  fraction of the motion completed before termination) plus
  `eval/all_metrics_dict.terminated` (length-64 bool) and `motion_keys`.
  `progress[i] * n_frames(motion_i)` gives the survived frame count per motion
  (motion frame counts come from the eval subset manifest / motion dataset,
  not from these files).
- Consequence for E3(b): survival-**matched per-frame** metrics (e.g. mpjpe
  over the first K frames common to both arms) are **NOT** computable from
  persisted artifacts → that specific design costs ~1.3 GPU-h (re-eval of the
  40 snapshots, which ARE persisted — see below). Survival-**aware per-motion**
  analyses (pairing motions by survival length, per-motion progress deltas,
  truncation-length comparisons) cost **0 GPU-h** — the data is already there.

## Where the artifacts live

`/workspace` is **host-visible** (no `docker exec` needed; container
`isaac-lab-base` mounts the same tree). Eval outputs live under
`/workspace/wbc-training-logs/` — 201 `*eval*` entries, including for each of
the 40 v4 segments (arm ∈ {control, manager} × seed ∈ {42, 1337} × s1..s10):

- `cmp_<arm>_seed<seed>_<arm>_sN_eval/` — in-distribution eval dir
- `cmp_<arm>_seed<seed>_<arm>_sN_eval.log` — driver log (tqdm progress only)
- `cmp_<arm>_seed<seed>_<arm>_sN_heldout_eval/` — held-out eval dir (same schema)
- `cmp_<arm>_seed<seed>_<arm>_sN_heldout_eval.log`

No `*_eval_pinned/` dirs exist for the seed42/seed1337 v4 runs (the scorer's
pinned-preference fallback correctly falls through to `_eval`).

Run dirs with snapshots: `/workspace/wbc-training-logs/cmp_manager_seed1337/
manager_sN-<timestamp>/` containing `snapshot_manager_sN.pt`, `last.pt`,
`model_step_000050.pt`, `config.yaml`, `meta.yaml`, `.hydra/` logs — so
re-eval of all 40 snapshots is possible if E3(b) needs true per-frame data.

## [verified] Directory listing of one segment's eval output

Command:

```
ls -laR /workspace/wbc-training-logs/cmp_manager_seed1337_manager_s10_eval \
        /workspace/wbc-training-logs/cmp_manager_seed1337_manager_s10_heldout_eval
```

Result: each dir contains **exactly one file**:

```
cmp_manager_seed1337_manager_s10_eval/metrics_eval.json          (122560 bytes)
cmp_manager_seed1337_manager_s10_heldout_eval/metrics_eval.json  (122607 bytes)
```

## [verified] Schema of metrics_eval.json

Command (python3, read-only):

```
d = json.load(open("/workspace/wbc-training-logs/cmp_manager_seed1337_manager_s10_eval/metrics_eval.json"))
# 56 top-level keys: 25× eval/all/*, 27× eval/success/* (scalars),
# eval/all_metrics_dict, eval/failed_metrics_dict (dicts of arrays),
# failed_keys, failed_idxes (length-63 lists)
```

Top-level key groups:

| group | count | granularity |
|---|---|---|
| `eval/success/*` (mpjpe_g/l/pa, accel/vel_dist, body-part variants, success_rate, progress_rate) | 27 | scalar |
| `eval/all/*` (same metric families) | 25 | scalar |
| `eval/all_metrics_dict` | 1 | dict of 29 length-64 per-**motion** arrays |
| `eval/failed_metrics_dict` | 1 | dict of 27 length-63 per-motion arrays (failed motions only) |
| `failed_keys` / `failed_idxes` | 2 | length-63 lists (motion names / indices) |

`eval/all_metrics_dict` keys (all length-64, one entry per eval motion):

```
motion_keys, progress, terminated, sampling_prob,
mpjpe_g, mpjpe_l, mpjpe_pa, accel_dist, vel_dist,
{mpjpe_g,mpjpe_l,mpjpe_pa,accel_dist,vel_dist}_{legs,vr_3points,other_upper_bodies,foot}
```

Schema excerpt (first entries):

```
"motion_keys": ["dance_basic_slide_180_R_loop_002__A323", "walk_big_dog_ff_315_loop_R_002__A494", ...]
"progress":    [0.06813627481460571, 0.02994011901319027, 0.16981132328510284, 0.12631578743457794, ...]
"terminated":  [true, true, true, true, ...]
"mpjpe_g":     [64.37951040436836, 56.414203689759034, 79.32449085005904, ...]
"mpjpe_l":     [45.78070118801653, 32.57332454819277, ...]
```

Every array is one **scalar per motion** (already averaged over that motion's
executed frames). There is no key whose value is a per-frame time series, a
per-frame mask, or a (motion × frame) matrix. The heldout eval's
`metrics_eval.json` has the identical schema (verified by key-set comparison).

The `*_eval.log` files contain only tqdm progress lines (e.g.
`Terminated: 63 | max frames: 242 | steps 241 ...`) — max-frame counts per
eval batch are visible there but not per-motion per-frame data.

## E3(b) cost decision

| E3(b) design | data source | GPU cost |
|---|---|---|
| Survival-matched **per-motion** metrics (pair by per-motion progress/truncation, LOO, paired deltas) | persisted `metrics_eval.json` (`progress`, `terminated`, per-motion metric arrays) | **0 GPU-h** |
| Survival-matched **per-frame** metrics (mpjpe over common frame prefix) | requires re-eval of 40 persisted snapshots with per-frame dumping enabled | **~1.3 GPU-h** |
