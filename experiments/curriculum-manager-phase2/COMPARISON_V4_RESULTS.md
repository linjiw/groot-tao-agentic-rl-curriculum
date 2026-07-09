# ON-vs-OFF comparison v4 — multi-seed, longer horizon, held-out protected metric

**Status: COMPLETE (2026-07-07). 2 seeds × 2 arms × 10 segments, all 40
segments trained + eval-scored; report rebuilt by
`score_comparison_multiseed.py --seeds 42 1337` → `multiseed_v4_report.json`.
Verdict (REVISED post-review, see "Post-review amendment" below): the
mechanism is fully validated at campaign scale, but per-motion decomposition
shows the apparent arm-level signal (manager +5.4% relative on final
progress_rate, cross-seed mean; the "reproducible s7 jump") is driven by ONE
motion out of 64 undergoing a bistable breakthrough that the control arm also
reaches unaided. NO curriculum-side value signal survives decomposition.
Held-out metrics show no leakage and no measurable generalization gain.**

## What v4 changes over v3

v3 (COMPARISON_V3_RESULTS.md) demonstrated the eval-scored mechanism at
anecdote scale (1 seed, 6 segments, 1 decision) and explicitly deferred any
value claim to "multi-seed, longer runs". v4 delivers that step plus one
protocol upgrade:

1. **Multi-seed**: seeds 42 and 1337, control + manager arms each, run
   sequentially on the single A10G (`run_comparison_multiseed.sh`, env-var
   selectable `SEEDS`/`ARMS`/`SEGS`).
2. **Longer horizon**: 10 segments × 50 iters = 500 iters per arm
   (v3: 300), resumed from the same overnight baseline
   `model_step_002000.pt` (live region of the eval curve).
3. **Held-out protected metric** (doc 08 §5): training pinned to a
   64-motion curriculum-only split (`curriculum_eval64`); after every
   segment a SECOND eval pass runs on a disjoint 64-motion held-out subset.
   The manager can only see/optimize the curriculum side; the held-out pass
   detects both leakage and (in principle) generalization transfer.
4. **Eval motion_file pin** (bug fixed pre-run, unit-tested): the standard
   per-segment eval pass explicitly pins `motion_file` to the
   curriculum_eval64 subset, closing a config-inheritance leak analogous to
   v3's M1 finding.

## Protocol

- Both arms start from `wbc_baseline_10k-20260701_232851/model_step_002000.pt`.
- Per seed: control (no changes ever) and manager
  (`TrainSideBandPolicy`, band len_low=20, sustain=2, binding-axis
  selection; eval-side tripwire: 30% relative AND >0.002 absolute drop on
  `eval/progress_rate`, 2 consecutive) — identical base config,
  `--base-knobs` seeding from the run's real stock values.
- 256 train envs, 64 eval envs, eval at fixed relaxed thresholds with the
  foot_pos_xyz re-pin (v3 M1 fix) plus the v4 motion_file pin.
- Journals/summaries/driver logs beside this file as
  `{arm}_{journal,summary,driver}_v4_seed{SEED}.{json,log}`.

### Run integrity [verified]

- 40/40 segments completed; `eval_sources` = `_eval` for every segment
  (all numbers from real per-segment eval passes, no train-side fallback).
- `prefix_identity_all_seeds: true` — within each seed, both arms'
  segment-1 configs and metrics are identical until the first manager
  decision (tick 2); arms differ only by the decision stream.
- 0 rollbacks, 0 validator rejections, 0 eval errors.
- One infra failure, not a method failure: the seed1337/manager run died
  at segment 9 the first time because /workspace hit 100% (each segment
  leaves ~3.6 GB of checkpoints; 4 runs × 10 segments ≈ 144 GB written).
  Artifacts archived in `archive_v4_failed/`; intermediate checkpoints
  (`last.pt`, `model_step_*.pt`) were purged (keeping eval-required
  `snapshot_*.pt`) and the run repeated cleanly
  (`SEEDS=1337 ARMS=manager bash run_comparison_multiseed.sh`, exit 0,
  wall time ~2 h [observed live, not artifact-backed: driver logs are
  0 bytes]). Lesson recorded in "Next steps".

## Measured results (source: `multiseed_v4_report.json`)

### PRIMARY — eval `progress_rate` on curriculum_eval64 (fixed thresholds)

Per-segment, per seed:

| Seg | ctrl s42 | ctrl s1337 | mgr s42 | mgr s1337 |
|---|---|---|---|---|
| 1 | 0.0869 | 0.0809 | 0.0869 | 0.0809 |
| 2 | 0.0882 | 0.0810 | 0.0862 | 0.0810 |
| 3 | 0.0877 | 0.0823 | 0.0863 | 0.0815 |
| 4 | 0.0882 | 0.0833 | 0.0852 | 0.0814 |
| 5 | 0.0894 | 0.0837 | 0.0851 | 0.0822 |
| 6 | 0.0908 | 0.0841 | 0.0870 | 0.0858 |
| 7 | 0.0938 | 0.0852 | **0.0993** | **0.1028** |
| 8 | 0.1091 | 0.0875 | 0.0998 | 0.1059 |
| 9 | 0.1109 | 0.0902 | 0.0977 | 0.1055 |
| 10 | 0.0988 | 0.0916 | 0.0951 | 0.1056 |

Final-segment cross-seed:

| arm | s42 final | s1337 final | mean | min..max |
|---|---|---|---|---|
| control | 0.0988 | 0.0916 | 0.0952 | 0.0916–0.0988 |
| manager | 0.0951 | 0.1056 | **0.1004** | 0.0951–0.1056 |

Key observations:

- **The s7 jump reproduces across both seeds — but see the Post-review
  amendment** [measured]: manager progress_rate steps up sharply at
  segment 7 in both seeds (s42: 0.0870→0.0993; s1337: 0.0858→0.1028),
  immediately following the tick-6 decision `foot_pos_xyz 0.25→0.30`
  (applied at iter 300) completing its watch window. Per-motion
  decomposition shows this jump is dominated by a single motion (51% of
  the s42 delta, 89% of s1337's) that control also breaks through on its
  own — see amendment before citing this as a decision effect.
- **Cross-seed mean favors the manager (+5.4% relative)** but the ranges
  overlap, **seed 42's final flips** (manager 0.0951 < control 0.0988),
  and the entire mean advantage disappears when the single breakthrough
  motion is excluded (amendment). With n=2 no significance is claimed.
- Manager pays an early cost: segments 2–6 manager ≤ control in 9 of 10
  seed-segment pairs (exception: s1337 seg 6, manager 0.0858 > control
  0.0841).

### SECONDARY — eval `mpjpe_l` (curriculum_eval64, mm)

| arm | s42 final | s1337 final | cross-seed mean |
|---|---|---|---|
| control | 43.55 | 43.27 | 43.41 |
| manager | 42.52 | 45.50 | 44.01 |

Manager mpjpe_l worsens in step with the s7 progress jump (s1337:
41.0→44.8 across s6→s7). This co-movement is a **survivor-composition
effect**: surviving deeper into 2002-frame clips adds harder late-clip
frames to the average. Do not read it as tracking degradation without a
survival-matched comparison (see Next steps).

### `mpjpe_g` — reported, survivor-biased (ANTI-correlated with survival)

control final mean 56.62 (55.07/58.18), manager 58.47 (56.52/60.43).
Same composition caveat, stronger; never read lower-is-better across
different survival lengths.

### Held-out protected metric (disjoint 64-motion subset) [measured]

- `heldout_success_rate`: 0.0 for all arms/seeds/segments — zero
  resolution at this scale (full-clip all-or-nothing), by construction.
- `heldout_mpjpe_g` s1→s10: control 51.5→50.8 (s42), 52.1→52.9 (s1337);
  manager 51.5→50.8 (s42), 52.1→52.6 (s1337). Flat, arms
  indistinguishable.
- Reading: **no leakage** — the manager's curriculum-side gains did not
  come at held-out expense; but also **no measurable generalization
  transfer** at 500 iters. The protected-metric harness works; its current
  metrics lack resolution to reward transfer (see Next steps).

### Training-side context (manager's own loop, NOT evidence)

Final len_mean: manager 18.6 vs control 13.5; rew_mean 1.16 vs 0.93.
Directionally consistent with the eval story, but longer episodes under
loosened terminations are partly definitional — context only.

### Manager decision streams [verified, journaled]

- seed 42: foot_pos_xyz 0.20→0.25 (t2), ee_body_pos 0.15→0.20 (t4),
  foot_pos_xyz →0.30 (t6), ee_body_pos →0.25 (t8), foot_pos_xyz →0.35
  (t10, pending at run end).
- seed 1337: same first three decisions at the same ticks; t6's
  foot_pos_xyz →0.30 scored `survived_effect_confirmed` (the only
  confirmed-effect outcome in the study); ee_body_pos →0.25 at t10
  (pending).
- Both seeds independently walked the same knob ladder — the band policy
  is deterministic given similar digests, so this is policy-consistency,
  not seed-agreement evidence; but the *effect* at s7 reproducing is.

## What this DOES show

1. The full closed loop now survives a 4-run, 40-segment, ~8 h multi-seed
   campaign with zero mechanism failures (no rollbacks needed, no
   validator rejections, no eval errors, prefix identity intact).
2. A specific journaled decision (foot_pos_xyz →0.30 @ iter 300) is
   followed by a curriculum-side progress_rate jump in both seeds — but
   per-motion decomposition attributes it to a single breakthrough motion
   that control also reaches (see amendment); it is NOT established as a
   decision effect.
3. The held-out protected-metric harness works end-to-end and shows no
   leakage from manager actions into the protected split.
4. Cross-seed mean final progress_rate: manager 0.1004 vs control 0.0952
   — an aggregate that does not survive per-motion decomposition
   (amendment).

## What this does NOT show (do not cite otherwise)

- **Not a value claim.** n=2 seeds, ranges overlap
  (0.0951–0.1056 vs 0.0916–0.0988), seed 42's final flips. Cross-seed
  spread is a range, not a CI. progress_rate quantum is
  1/(2·2002) ≈ 0.00025 — though the s7 jumps (0.0123–0.0170, ≈49–68
  quanta) are far above quantization noise, arm-level final differences are not
  distinguishable from seed noise at n=2.
- **No generalization claim.** Held-out metrics are flat for everyone;
  success_rate has zero resolution.
- **mpjpe_l/g degradation is not established** — survivor-composition
  confound is unresolved.
- **The tripwire still never faced a real test** (0 rollbacks); its guard
  behavior remains unit-test-only evidence.
- ~~Both arms remain on 2 curriculum motions resumed from one shared
  baseline checkpoint — library-scale behavior is untested.~~
  **CORRECTED 2026-07-09 [measured from run logs + resolved configs]:** v4
  training ran on the full 116,924-motion `robot_curriculum` split ("Loaded
  116924 motions" in every v4 segment log). What remains true: the shared
  warm-start checkpoint (`baseline_10k` step 2000) had only ever seen 2
  motions, and the termination-threshold action space does not shape the
  training distribution. See `PHASE2_FINAL_REPORT_DRAFT.md` amendment +
  `docs/design/10-library-scale-gate-program.md` §0.1.

## Bugs caught by/for this run

1. **Eval motion_file inheritance leak** (pre-run): the standard eval
   pass inherited the training config's motion_file; pinned to
   curriculum_eval64 + structural unit test (same class of bug as v3 M1 —
   scoreboard boundaries must be pinned, not assumed).
2. **Disk exhaustion kills runs at segment granularity** (during):
   per-segment checkpoints (~3.6 GB/segment) filled /workspace at run 4
   of 4, failing s9's eval (the retained tick-9 error records only
   "no metrics_eval.json … No such file or directory"; disk-full is the
   [speculative] cause inferred from the observed 100% /workspace — no
   retained artifact contains an explicit ENOSPC). The failure was
   cleanly journaled (`segment_failed`, tick 10) and detected by
   verification (missing s10 snapshot), not by the exit code alone —
   run-level rc checking is insufficient; per-segment artifact
   verification caught it.
3. **Initial "all 4 runs rc=0" conclusion was wrong** (during): the
   summary JSON existed for the failed run; only journal-event +
   snapshot-presence checks exposed the failure. Verification protocol
   updated accordingly.

## Next steps (input to the v5 planning cycle)

1. Seeds ≥ 3–5 before any arm-level claim; consider longer horizon past
   s10 (control's late surge at s8–s9 in seed 42 suggests the two arms'
   trajectories cross repeatedly at this scale).
2. Survival-matched tracking metric (mpjpe on frames both arms survive,
   or per-frame-index curves) to break the survivor-composition confound.
3. Held-out metrics with resolution: partial-progress / per-frame success
   on the held-out split instead of all-or-nothing success_rate.
4. Ablation for the s7 effect: control arm + a hard-coded foot_pos_xyz
   →0.30 at iter 300 (no manager) — is the gain from *adaptive* choice or
   from that one relaxation applied to anyone?
5. Disk hygiene in the driver: delete `last.pt`/`model_step_*.pt` at
   segment end (keep `snapshot_*.pt`), assert free-space headroom before
   each segment.
6. Scale curriculum beyond 2 motions toward the real library.

## Post-review amendment (2026-07-07, after adversarial review — all numbers independently re-derived from the four journals' `per_motion_progress` payloads)

The adversarial methodology review decomposed the eval payloads per motion;
the parent independently reproduced every number below [measured].

1. **The s7 jump is one motion, not a distribution shift.**
   `postmortem_convulsions_stomach_loop_R_001__A471_M` (1 of 64 eval
   motions) accounts for **51%** of manager-s42's s6→s7 delta
   (motion progress 0.025→0.426) and **89%** of manager-s1337's
   (0.029→1.000). The same motion has a **bistable breakthrough in
   control s42 with no manager at all**: 0.054→1.000 at s7→s8, which is
   **96%** of control's own +0.0153 jump. Full per-segment trajectories:
   - control s42: 0.021 … 0.054, **1.0, 1.0**, 0.05 (regresses at s10)
   - manager s42: 0.021 … 0.025, **0.426, 0.570, 0.566, 0.426**
   - control s1337: 0.017 … 0.054, 0.054 (never breaks through)
   - manager s1337: 0.017 … 0.029, **1.0, 1.0, 1.0, 1.0**
   The design cannot distinguish "the decision caused the breakthrough"
   from "the shared baseline checkpoint was near a bistable breakthrough
   that any continued training can reach".
2. **The +5.4% cross-seed mean advantage is the same single motion.**
   Excluding it, manager final progress is ≤ control in **both** seeds:
   s42 0.0899 vs 0.0996; s1337 0.0914 vs 0.0922. Paired per-motion final
   deltas: median exactly 0.0000 in both seeds; win/loss/tie
   s42 = 22/22/20, s1337 = 26/18/20.
3. **Revised verdict**: v4 establishes the *mechanism* (closed loop,
   guardrails, journaling, verification protocol) at campaign scale, and
   establishes that the *measurement/analysis pipeline must decompose
   per-motion before any arm-level claim*. It does **not** provide a
   directionally positive value signal for the manager. The v5 plan's
   tier-1 gate (scripted-decision replay ablation + survivor-confound
   resolution before any seed-scaling) is the correct next move; add
   per-motion decomposition to the scorer as a standing requirement.

Related reports: `review_v4_data_audit.md`, `review_v4_methodology.md`,
`plan_v5_draft.md`.
