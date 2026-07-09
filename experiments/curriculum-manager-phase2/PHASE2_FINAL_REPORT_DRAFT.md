# Phase-2 Final Report (DRAFT) — Adaptive Curriculum Manager for SONIC WBC Training

**Status: DRAFT, 2026-07-08. Not yet through adversarial review; do not cite as final.**

> **AMENDMENT (2026-07-09, recomputed from run logs — supersedes the "2 curriculum
> motions" scale caveat everywhere it appears below):**
>
> 1. **v4 and E1 trained at library scale, not on 2 motions.** Training logs
>    (`/workspace/wbc-training-logs/cmp_control_seed42_control_s1.log`,
>    `cmp_scripted_seed1337_scripted_s1.log`) show "Loaded 116924 motions"
>    (`motion_file: data/motion_lib_bones_seed/robot_curriculum` in every v4-era
>    resolved config.yaml) [measured]. Only v3 and the shared `baseline_10k`
>    warm-start trained on 2 motions. The correct scale caveat is: *the warm-start
>    checkpoint had only ever seen 2 motions, the horizon was 500 iters, and the
>    action space (termination thresholds) did not grip the 116k-motion training
>    distribution* — not that the curriculum was 2 motions.
> 2. **The seed-42 E5 replicates are not 4 independent runs.** rep2 ≡ rep3
>    bit-exactly on all 9 shared segments (identical rew_series; rep2 is missing its
>    control_s2 journal entry) [measured]. Distinct trajectories: rep1, rep2/3,
>    rep4 → effective n=3, final progress_rate {0.0988, 0.1063, 0.0969} — a ±5%
>    relative fixed-seed band, consistent with the E5b window-mean chaos floor.
>    Any σ_rep computed from these four files without bit-identity collapse
>    overstates n.
>
> Downstream consequence and the successor program:
> `docs/design/10-library-scale-gate-program.md`.

Scope: this report closes out phase 2 of the curriculum-manager line — the arc
from the first eval-scored closed-loop run (v3), through the multi-seed
campaign and its post-review amendment (v4), to the scripted-replay ablation
(E1 / v5). Every quantitative claim below marked [measured] was **recomputed
for this report directly from the raw per-segment journal JSONs**
(`{control,manager}_journal_v4_seed{42,1337}.json`,
`scripted_journal_v5_seed{42,1337}.json` — segment entries carry
`eval.progress_rate` and `eval.per_motion_progress`, n=64 motions), not copied
from the intermediate markdown reports. Recomputation script logic: final/per-
segment `eval_progress_rate`, per-motion paired win/loss/tie at the final
segment, prefix-determinism checks (exact float equality), and single-motion
jump attribution shares.

**Reconciliation note:** all recomputed numbers agree with
`COMPARISON_V4_RESULTS.md` (incl. its post-review amendment) and
`E1_SCRIPTED_ABLATION_RESULTS.md` to the precision those documents print; the
only deviations are last-digit rounding (e.g. the v4 doc's "s7 jumps
0.0123–0.0170" are exactly 0.012220 and 0.016996 recomputed; the amendment's
attribution shares "51% / 89% / 96%" are 51.2% / 89.3% / 96.5%). No
substantive discrepancy was found. One structural check the earlier docs
state loosely: "seed42 s1–s3 / seed1337 s1–s8 match manager to 4 decimals"
(E1 doc) is in fact **bit-exact float equality** in the journals [measured].

Claim labels: **[verified]** = structural/protocol fact confirmed against
artifacts or code in this repo; **[measured]** = number recomputed from raw
journals for this report; **[design]** = design intent / plan, not yet
evidenced.

---

## 1. Headline verdict

Across v3 → v4 → E1, the phase-2 evidence chain converges on a **null result
for the manager's core value proposition** at the tested scale (2 curriculum
motions, one shared baseline checkpoint, 10 segments × 50 iters, n=2 seeds):

1. **Closed-loop adaptivity contributes nothing measurable** [measured].
   Open-loop replay of the manager's own decision transcript (E1 `scripted`
   arm: no digest reads, no watch gating, tick-indexed knob ladder) matches or
   exceeds the closed-loop manager in both seeds:
   final `eval_progress_rate` seed42 scripted 0.0947 vs manager 0.0951
   (per-motion paired W/L/T = 19/21/24, a coin flip); seed1337 scripted
   0.1092 vs manager 0.1056 (W/L/T = 23/7/34, scripted slightly ahead).
   Per the pre-registered E1 kill-criteria, **C-adaptive is dead** at this
   scale.
2. **The knob ladder itself shows no consistent value over control**
   [measured]. Final progress_rate, ladder-bearing arms vs control, flips
   sign across seeds: seed42 control finishes highest (0.0988 vs manager
   0.0951 / scripted 0.0947); seed1337 scripted finishes highest (0.1092 vs
   manager 0.1056 vs control 0.0916). Cross-seed means (manager 0.1004,
   scripted 0.1019, control 0.0952; manager +5.5% rel., scripted +7.1% rel.)
   do not survive the sign flip or the per-motion decomposition below; at
   n=2 no claim beyond noise is made.
3. **Arm-level differences are dominated by a single-motion stochastic
   breakthrough plus run noise that exists even at fixed seed** [measured].
   One motion of 64, `postmortem_convulsions_stomach_loop_R_001__A471_M`,
   exhibits a bistable breakthrough that accounts for 51.2% (seed42) and
   89.3% (seed1337) of the manager's s6→s7 progress jump — and 96.5% of
   control-seed42's own s7→s8 jump, **with no manager at all**. Under the
   identical ladder and identical seed, the breakthrough appears in scripted
   seed1337 (final 1.000, = manager) but **not** in scripted seed42 (final
   0.054 vs manager's 0.426): a same-schedule, same-seed disappearance, i.e.
   training stochasticity, not decisions. Excluding this one motion, the
   manager's final progress is ≤ control in **both** seeds (s42 0.0899 vs
   0.0996; s1337 0.0914 vs 0.0922), median paired per-motion final delta is
   exactly 0.0000 in both seeds, and W/L/T = 22/22/20 (s42), 26/18/20
   (s1337).

**Project implication** (carried from E1, endorsed here): the "adaptive
curriculum manager" framing does not survive phase 2. Residual value, if any,
lives in (a) *schedule discovery* — search over relaxation ladders rather
than reactive adjustment, and (b) the guardrail/journal/verification scaffold
itself, which is fully validated infrastructure (§4).

---

## 2. Evidence chain: v3 → v4 (+amendment) → E1

### 2.1 v3 — eval-scored mechanism at anecdote scale (2026-07-02)

1 seed (42), 6 segments × 50 iters, control vs manager from the shared
`model_step_002000.pt` baseline. Contribution: retired the v2 "self-inflated
tripwire" caveat by adding a **per-segment eval pass at fixed relaxed
thresholds** and moving the tripwire to `eval/progress_rate`.

- One decision (`foot_pos_xyz 0.20→0.25` after s2), journaled with digest
  hash and applied-iter provenance; survived its watch [verified].
- Final progress_rate manager 0.00400 vs control 0.00375 [measured,
  recomputed from `{control,manager}_journal_v3.json`] — exactly **one
  quantization step** (quantum 1/(2·2002) ≈ 0.00025 at 2 motions); explicitly
  not a value claim.
- Adversarial review finding **M1**: eval inherited the manager's own
  `foot_pos_xyz` override through the checkpoint-sibling config — the
  scoreboard boundary was not pinned. Fixed with an explicit eval re-pin +
  structural unit test; affected segments re-evaluated byte-identical (luck
  of the dynamics, not a design property) [verified].

### 2.2 v4 — multi-seed campaign (2026-07-07)

2 seeds × {control, manager} × 10 segments × 50 iters (40 segments total),
plus a held-out protected metric on a disjoint 64-motion split, plus an
explicit eval `motion_file` pin (closing a second M1-class leak pre-run).

Per-segment `eval_progress_rate`, all recomputed [measured]:

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

- Run integrity [verified]: 40/40 segments; all eval numbers from real
  per-segment eval passes (`eval_sources = _eval`); 0 rollbacks, 0 validator
  rejections, 0 eval errors; prefix identity within each seed holds — segment
  1 bit-exact across arms, divergence begins at s2 exactly when the first
  decision stream diverges (first change tick 2) [measured/verified via
  `multiseed_v4_report.json` `prefix_identity_all_seeds: true` + recomputed
  s1 float equality].
- Both seeds independently walked the same knob ladder [verified, journaled]:
  foot_pos_xyz 0.20→0.25 (t2), ee_body_pos 0.15→0.20 (t4), foot_pos_xyz
  →0.30 (t6); then s42: ee_body_pos →0.25 (t8), foot_pos_xyz →0.35 (t10,
  pending); s1337: ee_body_pos →0.25 at **t10** (pending). Policy determinism
  given similar digests — consistency, not seed-agreement evidence.
- Headline at the time: the s7 jump (manager +0.01222 s42, +0.01700 s1337
  [measured]) reproduced across seeds right after the t6 `foot_pos_xyz →0.30`
  decision; cross-seed mean final favored the manager (+5.5% rel. [measured];
  the v4 doc prints +5.4%, a rounding difference).
- Infra failure (not method): seed1337/manager first attempt died at s9 on
  disk exhaustion (~3.6 GB checkpoints/segment × 4 runs); caught by
  **per-segment artifact verification** (journal `segment_failed` + missing
  s10 snapshot), *not* by exit codes — the initial "all 4 runs rc=0" read
  was wrong. Rerun clean after manual checkpoint purge; failed artifacts in
  `archive_v4_failed/` [verified].

### 2.3 v4 post-review amendment — per-motion decomposition kills the signal

The adversarial review decomposed `per_motion_progress`; every number was
independently reproduced then, and re-reproduced for this report [measured]:

- The s7 jump is **one motion**, `postmortem_convulsions_stomach_loop_R_001__A471_M`.
  Full final-report-recomputed trajectories (per-segment motion progress):
  - control s42: 0.021 … 0.054, **1.0, 1.0**, 0.05 (breaks through unaided at
    s8, regresses at s10)
  - manager s42: 0.021 … 0.025, **0.426, 0.570, 0.566, 0.426**
  - control s1337: 0.017 … 0.054 (never breaks through)
  - manager s1337: 0.017 … 0.029, **1.0, 1.0, 1.0, 1.0**
  Attribution shares: 51.2% of manager-s42's s6→s7 delta, 89.3% of
  manager-s1337's, 96.5% of control-s42's own s7→s8 jump.
- Excluding this motion: manager ≤ control in both seeds (0.0899 vs 0.0996;
  0.0914 vs 0.0922); median paired delta 0.0000/0.0000; W/L/T 22/22/20 and
  26/18/20.
- Revised v4 verdict: mechanism validated at campaign scale; **no
  directionally positive value signal**; per-motion decomposition becomes a
  standing scorer requirement.

### 2.4 E1 (v5) — scripted-replay ablation (2026-07-08)

Third arm `scripted`: open-loop replay of the manager ladder (foot_pos_xyz
→0.25 landing at s3, ee_body_pos →0.20 at s5, foot_pos_xyz →0.30 at s7,
ee_body_pos →0.25 at s9; one-segment application lag identical to manager
semantics), same seeds/checkpoint/segments. 20/20 segments, 0 failures; E0
checkpoint purge in production (44 GB freed, incl. docker-exec fallback for
root-owned files) [verified].

Ladder fidelity [verified from `knobs_in`, recomputed]: seed42 is an exact
replay. **Timing caveat (per the corrected E1 doc)**: manager seed1337 issued
its second ee_body_pos rung at tick 10, which with the application lag would
land at s11 (never runs); the scripted ladder applies it at s9 — i.e. for
seed1337 the scripted second ee rung is two segments *earlier* than that
seed's manager transcript, and scripted is the only arm that actually
experienced it. The match-or-beat conclusion is robust to this (scripted ≥
manager there).

Recomputed results [measured]:

| seed | scripted (open-loop) | manager (closed-loop) | control |
|---|---|---|---|
| 42 | 0.0947 | 0.0951 | **0.0988** |
| 1337 | **0.1092** | 0.1056 | 0.0916 |

Per-motion paired at final segment: scripted-vs-manager W/L/T = 19/21/24
(s42), 23/7/34 (s1337); scripted-vs-control = 20/21/23 (s42), 32/17/15
(s1337) — the latter's apparent seed1337 edge carries the same single-motion
breakthrough confound.

**Prefix determinism, then micro-divergence** [measured, exact float
comparison]: with identical seed and identical knobs, scripted equals manager
**bit-exactly** for seed42 s1–s3 and seed1337 s1–s8; divergence begins
exactly where configs still match (seed42 s4: 0.086738 vs 0.085170 under
identical knobs) — i.e. GPU-level nondeterminism produces real, nonzero
run-to-run noise **at fixed seed**. This is the noise term that arm-level
comparisons at this scale sit inside.

**Breakthrough is stochastic** [measured]: postmortem motion final — scripted
s42 0.054 vs manager 0.426 (same ladder, same seed: did NOT reproduce);
scripted s1337 1.000 = manager 1.000 (did reproduce).

E1 verdict (pre-registered kill-criteria): **adaptivity dead as an
explanation of v4 differences**; E3(b) (survival-matched mpjpe re-eval)
demoted to low priority — it would refine a comparison whose headline is
already null.

### 2.5 Secondary metrics (context only)

- Final `mpjpe_l` (curriculum eval, mm) [measured]: control 43.55/43.27,
  manager 42.52/45.50, scripted 42.61/44.80 (s42/s1337). Manager/scripted
  mpjpe worsening co-moves with progress jumps — a **survivor-composition
  effect** (surviving deeper into 2002-frame clips adds harder frames);
  not established as tracking degradation (E3(a) showed persisted artifacts
  cannot support survival-matched per-frame analysis at 0 GPU-h).
- Held-out protected split [measured]: `heldout_success_rate` = 0.0 for all
  arms/seeds/segments (zero resolution by construction);
  `heldout_mpjpe_g` s1→s10 flat and arm-indistinguishable (control
  51.5→50.8 / 52.1→52.9; manager 51.5→50.8 / 52.1→52.6; scripted 51.5→49.6 /
  52.1→52.7). Reading: **no leakage, no measurable generalization transfer**;
  the harness works, its metrics lack resolution.

---

## 3. Methodology lessons (standing requirements going forward)

1. **Per-motion decomposition before any arm-level claim** [verified by
   consequence]. An aggregate over 64 motions hid a single bistable motion
   that manufactured a "+5.4%, cross-seed-reproducing" story twice (v4
   headline, then nearly again in E1's scripted-vs-control s1337 numbers).
   Decomposition is now a scorer requirement, not an optional audit.
2. **The scoreboard boundary must be pinned, not assumed** [verified]. Two
   instances of the same bug class: v3 M1 (eval inherited the manager's own
   termination override via checkpoint-sibling config) and the v4 pre-run
   eval `motion_file` inheritance leak. Fix pattern both times: explicit
   pin in `build_eval_command` + a structural unit test that fails if any
   action-space knob (or the motion set) lacks an eval pin.
3. **Per-segment artifact verification, not exit codes** [verified]. The v4
   disk-exhaustion failure produced a plausible summary JSON and an initial
   (wrong) "all 4 runs rc=0" conclusion; only journal-event + snapshot-
   presence checks exposed it. Run success is decided by artifacts.
4. **Fixed seed ≠ deterministic run** [measured]. E1's bit-exact prefixes
   followed by divergence under identical configs prove a nonzero fixed-seed
   noise floor; any effect size claimed at this scale must be benchmarked
   against replicate-run noise (→ E5, §5).
5. **Cheapest kill-experiment first** [design, validated by outcome]. The v5
   plan's tier-1 ordering (E0 → E1 before any seed scaling) resolved the
   project's framing question for ≈2.2 GPU-h instead of an n=5 value
   campaign (~15 GPU-h) on a doomed comparison.
6. **Adversarial review before commit of any results doc** [verified by
   consequence]. Both major reversals of this phase (M1, the per-motion
   amendment) came from review, not from the primary analysis.
7. **Disk hygiene is a correctness concern** [verified]. ~3.6 GB of
   checkpoints per segment silently killed a run at segment granularity;
   purge + free-space gate are now driver features, not ops chores.
8. **Quantization awareness** [measured]: progress_rate moves in quanta of
   1/(2·2002) ≈ 0.00025; v3's "one quantum ahead" and all sub-quantum
   readings were correctly refused as signal.

---

## 4. Validated infrastructure (the durable asset)

All items below survived a 6-run, 60-segment cumulative campaign (v4 40 +
E1 20) with zero mechanism failures [verified]:

- **`smoke_driver.py` closed loop**: segment training → console-log parse →
  run digest → policy propose → knob-registry validation (whitelist,
  one-notch, cooldown, one-pending gate) → apply-as-overrides → per-segment
  eval at fixed pinned thresholds → eval-side tripwire watch → outcome
  scoring → journal with provenance (digest hash, applied-at-iter).
- **Journal / rollback machinery**: every decision journaled with rationale,
  expected-effect check, and tripwire spec; failure events
  (`segment_failed`, `disk_gate_failed`) journaled. Caveat: **0 rollbacks
  occurred across all of phase 2** — tripwire guard behavior remains
  unit-test-only evidence [verified].
- **E0 disk hygiene in production**: post-segment purge of
  `last.pt`/`model_step_*.pt` (keeping eval-required `snapshot_*.pt`),
  free-space gate before each segment, docker-exec fallback for root-owned
  files; freed 44 GB during E1 with 0 failures [verified].
- **Held-out watcher**: disjoint 64-motion protected split, second eval pass
  per segment, metrics reachable by tripwires but invisible to the manager's
  knobs; demonstrated leak-free end-to-end [verified] (resolution of its
  current metrics is a separate, open problem).
- **Multi-seed harness + scorer** (`run_comparison_multiseed.sh`,
  `score_comparison_multiseed.py`): env-selectable seeds/arms/segments,
  prefix-identity checking, eval-source accounting, unit-tested.
- **Verification protocol**: per-segment journal-event + snapshot-presence
  checks as the run-success criterion (proved necessary in v4).
- **Scripted-arm ablation capability** (`ScriptedPolicy`): open-loop replay
  as a first-class arm — reusable for any future schedule-discovery work.

---

## 5. Open questions

1. **Fixed-seed noise floor (E5 replicate runs)** [design — pending]. E1
   observation 2 established the noise is nonzero but not its magnitude
   distribution. A replicate campaign (same arm, same seed, same config,
   multiple reps — e.g. control seed42 rep2+) would price the noise floor
   that any future effect must exceed. A launch log stub for this exists
   (`e5_noise_floor_v5.log`, `control_summary_v4_seed42_rep2.json` — both
   effectively empty, 2026-07-08): the run has **not** produced data yet; no
   numbers from it appear in this report.
2. **Motion-set curriculum pilot** [design]. All of phase 2 ran on 2
   curriculum motions from one shared checkpoint. The surviving hypothesis —
   value from schedule *discovery* rather than reactive adaptivity — needs a
   regime where fixed schedules can actually fail: library-scale motion
   sets, motion-set composition as the knob (not just termination
   thresholds), and/or distribution shift mid-run.
3. **Held-out metrics with resolution** [design]: partial-progress /
   per-frame success on the protected split instead of all-or-nothing
   success_rate (currently 0.0 everywhere).
4. **Survivor confound** [open, deprioritized]: survival-matched tracking
   metrics require ~1.3 GPU-h of re-eval (E3(a): persisted artifacts are
   per-motion only); only worth doing if a future comparison has a live
   headline.
5. **Tripwire under fire** [open]: rollback has never triggered on a real
   regression; a deliberate harmful-knob probe would convert it from
   unit-test evidence to campaign evidence.

---

## Appendix — data provenance

| Claim domain | Raw source (this repo, `experiments/curriculum-manager-phase2/`) |
|---|---|
| v3 numbers | `{control,manager}_journal_v3.json` |
| v4 numbers | `{control,manager}_journal_v4_seed{42,1337}.json` (10 segment entries each; `eval.progress_rate`, `eval.per_motion_progress` n=64, `heldout.*`, `knobs_in`, `decision`) |
| E1 numbers | `scripted_journal_v5_seed{42,1337}.json` (same schema) |
| Aggregates cross-check | `multiseed_v4_report.json` (prefix identity, eval sources) |
| Failed-run forensics | `archive_v4_failed/` |
| Narrative sources | `COMPARISON_V3_RESULTS.md`, `COMPARISON_V4_RESULTS.md` (+post-review amendment), `E1_SCRIPTED_ABLATION_RESULTS.md`, `E3A_FINDINGS.md`, `plan_v5_draft.md`, `review_v4_{data_audit,methodology}.md` |

Recomputation performed 2026-07-08 with python3 directly over the journal
JSONs; `progress_rate` verified to equal the mean of `per_motion_progress`
(64 motions) to full float precision.
