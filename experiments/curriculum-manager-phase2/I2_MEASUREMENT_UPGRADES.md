# I2 — Measurement upgrades (doc 10 §3-I2)

Three sub-items; two done now (CPU, this commit), one folded into G0.

## I2.1 — Resolving held-out metric — **DONE, 0 GPU-h**

**Finding (recon, [verified] against real artifacts):** the resolving held-out
metric doc 10 §2 asked for ALREADY EXISTS in every persisted `metrics_eval.json`;
no re-eval needed. E3A established `eval/all_metrics_dict.progress` is per-motion
progress (fraction of clip completed before termination), continuous in [0,1]. The v4
held-out `success_rate` was 0.0 everywhere (all-or-nothing full-clip completion), but
per-motion progress on the same held-out eval is fully resolving:

Real v4 held-out artifact
(`cmp_manager_seed1337_manager_s10_heldout_eval/metrics_eval.json`) [measured]:
- `heldout_success_rate` = **0.0** (useless — the doc-10 "zero resolution" problem)
- `heldout_progress_rate` = **0.1056**  ← the resolving scalar
- per-motion progress: mean 0.1056, min 0.0093, max 0.3196, **64/64 nonzero**, stdev 0.066

**Change:** `holdout.heldout_record_from_metrics_eval` now additionally emits
`heldout_progress_rate` (scalar) and `heldout_progress_per_motion`
(mean/min/max/nonzero/n), with the SAME strict integrity guard as `failed_keys`
(per-motion `motion_keys` must lie inside the held-out subset, else refuse — a wrong
motion set cannot silently feed a wrong resolving metric). Additive/back-compatible:
absent progress data => keys simply omitted. 3 new tests + a real-artifact check.

**Consequence for the plan:** doc 10 §2's "held-out metric with resolution" gap is
closed on the read path at 0 GPU-h. The G2/G4 driver should journal
`heldout_progress_rate` (via the held-out hook) as the protected metric instead of
success_rate. The `heldout_success_rate` 0.0 in v4 was never evidence of "no transfer"
— it was a saturated metric; progress is the honest one. **True per-FRAME series still
do not persist** (E3A) and would cost ~1.3 GPU-h of re-eval — not needed, per-motion
progress resolves.

## I2.2 — Effective-replicate collapse — **DONE, CPU**

New module `replicate_collapse.py` (17 tests incl. real v4 data). Makes doc 10 §0.2
structural: fixed-seed replicates are bit-deterministic modulo kernel events, so a
spread statistic over launch count overstates n.

- `series_bit_identical` / `mapping_bit_identical` — exact (no-tolerance) equality on
  shared segments (by list index / by SEGMENT NAME resp.), with a `min_overlap` guard
  so warm-start-only stubs don't merge.
- `collapse_by_segment` — the real-run path: keys on segment NAME (a dropped journal
  entry shifts list positions; v4 rep2 is missing `control_s2`). **Confirmed on real
  data: v4 seed-42 control collapses 4 launches -> 3 distinct trajectories** (rep2 ==
  rep3 bit-exactly on all 9 shared segments).
- `noise_band_from_replicates` — the G1 helper: collapse, then endpoint band
  (mean/range/rel_range/sigma) over DISTINCT trajectories only; REFUSES a band with < 2
  distinct (returns sigma=None + note, not a fabricated 0). On real v4 seed-42:
  rel_range ≈ 0.094 over 3 distinct — matching doc 10 §0.2's ±5% band.

`collapse_replicates` (raw-index) is retained but documented as conservative (won't
merge ragged pairs) — use `collapse_by_segment` on real journals.

## I2.3 — Library-native baseline training script — **folded into G0**

The fresh ~2,000-iter `robot_curriculum` baseline is a GPU launch; its script belongs
with the G0 driver (needs the box free). Deferred to the G0 prep task, not a CPU
deliverable. The launch command is a `job_adapter.build_train_command` with
`extra_overrides` pinning `motion_file=data/motion_lib_bones_seed/robot_curriculum`,
no checkpoint (from scratch), 2000 iters — same chain as v4, different motion pin and
no warm start.

## Tests
- `test_replicate_collapse.py` — 17 (incl. 3 real-v4-data).
- `skills/agentic/sonic-heldout-watcher/test_holdout.py` — +3 (resolving metric,
  foreign-key refusal, optionality). 16 total.
