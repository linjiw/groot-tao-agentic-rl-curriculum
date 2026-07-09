# Adversarial methodology review — COMPARISON_V4_RESULTS.md (v4 multi-seed ON-vs-OFF)

Reviewer: methodology-review subagent, 2026-07-07. Scope: design validity, not
arithmetic. Sources read this pass: COMPARISON_V4_RESULTS.md,
COMPARISON_V3_RESULTS.md, SMOKE_RESULTS.md, run_comparison_multiseed.sh,
smoke_driver.py, score_comparison_multiseed.py, docs/design/08-curriculum-manager-agent.md,
skills/agentic/sonic-heldout-watcher/holdout.py, all four v4 journals
(`{arm}_journal_v4_seed{42,1337}.json`, incl. per-motion eval payloads),
`archive_v4_failed/`, `multiseed_v4_report.json`, resolved run configs under
`/workspace/wbc-training-logs/cmp_*`, and the motion-set directories.
Labels: [verified] = read in source/artifacts this pass; [design-doc] = from
doc 08 / results docs; [speculative] = reviewer inference.

Severity: **P0** invalidates a stated conclusion · **P1** materially weakens
one · **P2** improvement idea.

---

## Summary verdict

The *mechanism* claims (loop integrity, pinned scoreboard, prefix identity,
no-leakage plumbing) are well-supported and the doc's refusal to make a value
claim at n=2 is correct. But the two headline *evidence* claims — "the s7 jump
reproduces across both seeds" as the first cross-seed-replicated effect, and
the +5.4% cross-seed mean manager advantage — do not survive per-motion
decomposition of data already sitting in the journals: both are dominated by a
single bistable motion whose breakthrough **also occurs in a control arm with
no manager decision at all**. The design (shared baseline checkpoint,
deterministic policy, identical decision streams) further means the two
"seeds" are far weaker replicates than the doc's language implies.

---

## P0 findings

### P0-1 — The "s7 jump reproduces across both seeds" claim is not licensed as a decision effect

The doc calls this "the strongest signal in the run: a specific, journaled
decision followed by the same qualitative effect in two independent seeds."
Decomposing the per-motion progress in the journals [verified]:

- **mgr seed42, s6→s7** (jump +0.0122): `postmortem_convulsions_stomach_loop`
  goes 0.025→0.426, contributing **+0.0063 (~52%)** of the jump; second-largest
  contributor is `medium_heavy_one_hand...` (+0.241/64 ≈ +0.0038).
- **mgr seed1337, s6→s7** (jump +0.0170): the same motion goes 0.029→**1.000**
  (full clip), contributing **+0.0152 (~89%)** of the jump.
- **ctrl seed42, s7→s8** (jump +0.0153, *no manager, no decision, ever*): the
  same motion goes 0.054→**1.000**, contributing **+0.0148 (~97%)** of the jump.

So the "reproducible effect" is overwhelmingly a *single-motion breakthrough*
on one clip — and the identical breakthrough happens in a control arm one
segment later with zero intervention. The nonzero `success_rate = 0.015625 =
1/64` entries in the table are literally this one motion [verified]. The
design cannot distinguish (i) "the foot_pos_xyz→0.30 decision caused the
breakthrough" from (ii) "the shared 2k baseline checkpoint was near a
breakthrough on this clip and any/no perturbation reaches it within a segment
or two of noise jitter." Given (ii) demonstrably occurs in control, the claim
"a specific journaled decision followed by the same qualitative effect in two
independent seeds" (DOES-show #2) is **not licensed by this design**. What is
licensed: "manager arms broke through 1–2 segments earlier than control did in
one of two seeds" — a timing observation on one clip, at best.

Aggravating factor: the motion is bistable — ctrl seed42 reaches 1.000 at s8
and falls back to 0.050 by s10; mgr seed42 reaches only 0.426 and drifts
[verified]. Treating the mean over 64 motions as smooth when one term swings
by ~1/64 of the whole metric between adjacent checkpoints makes segment-level
jump attribution unreliable.

### P0-2 — The cross-seed mean advantage (+5.4%, DOES-show #4) is attributable to the same single motion

Final-segment paired per-motion comparison, computed from the journals
[verified]:

| seed | mgr wins / losses / ties (of 64) | median per-motion Δ | mean excl. postmortem: mgr vs ctrl |
|---|---|---|---|
| 42 | 22 / 22 / 20 | +0.0000 | **0.0899 vs 0.0996** (mgr worse) |
| 1337 | 26 / 18 / 20 | +0.0000 | **0.0914 vs 0.0922** (mgr slightly worse) |

Excluding the single postmortem clip, the manager's final progress_rate is
≤ control in **both** seeds; the median paired delta is exactly zero; win/loss
counts are near-even. The doc correctly refuses a *significance* claim, but
its verdict line still asserts "a directionally positive … curriculum-side
signal" and DOES-show #4 states the mean comparison as a shown fact. The
direction itself is a one-motion artifact interacting with endpoint choice
(control held postmortem=1.0 at s8–s9 and happened to lose it by the s10
snapshot; the manager s1337 arm happened to hold it). **"Directionally
positive" is not supported once the metric is decomposed** — the honest
summary is "indistinguishable, with one clip's bistable breakthrough deciding
the sign of the mean at the arbitrary final segment."

Note this is not arithmetic pedantry: the analysis above uses only data the
run already produced. The scorer aggregates `progress_rate` and never looks at
`per_motion_progress` [verified: score_comparison_multiseed.py], which is how
both P0s slipped past the doc's own (otherwise strong) caveat discipline.

---

## P1 findings

### P1-1 — Shared baseline checkpoint: the two seeds are not independent replicates (task angle a)

All four runs resume from the same `model_step_002000.pt`, itself trained with
seed 42 [verified: script + v3 doc]. Consequently any property of *that
checkpoint's* loss landscape — e.g., being on the verge of the postmortem
breakthrough (P0-1), or the timing of control's own late surge — will
"reproduce across seeds" without being a property of the manager, the method,
or even of training in general. Cross-seed agreement here bounds only the
*resumption-noise* variance, not run-to-run variance. Every phrase of the form
"in two independent seeds" overstates independence. Fix for v5: cross seeds
with ≥2–3 distinct baseline checkpoints (different baseline seeds, or at least
different steps: 1500/2000/2500). Two checkpoints × two seeds beats four seeds
on one checkpoint.

### P1-2 — Deterministic policy + identical decision streams: n=2 tests ONE schedule, not the manager (task angle b)

`TrainSideBandPolicy` is deterministic [verified: smoke_driver.py], both seeds
walked the same knob ladder at the same ticks [verified: journals], and the
archived failed seed1337 run replays segment-for-segment identically to the
re-run (len_mean and eval progress match to all digits for s1–s9) [verified:
archive_v4_failed vs final journal]. So "replication" here means: one fixed
intervention schedule, evaluated under two draws of training noise, from one
initialization. That is a legitimate A/B of *a schedule*, but it is not
evidence about the *adaptive controller*: the band trigger never faced a
digest where the two seeds should have decided differently, and the
closed-loop machinery (tripwire, rollback, axis selection under divergent
conditions) was never differentially exercised. The doc's own note ("this is
policy-consistency, not seed-agreement evidence") is honest but understates
the consequence: at this scale the manager is observationally equivalent to a
hard-coded schedule, which is exactly what the ablation must test (P1-3).

### P1-3 — The proposed ablation is the wrong (or at least incomplete) ablation (task angle c)

Next-steps #4 proposes: control + hard-coded `foot_pos_xyz→0.30 at iter 300`,
no manager. Problems:

1. **It replays the wrong counterfactual.** By iter 300 the manager arms had
   already applied foot→0.25 (iter 100) and ee_body_pos→0.20 (iter 200)
   [verified: journals]. A lone →0.30 at 300 on top of stock 0.2/0.15 is a
   state no arm ever visited; a null result would be uninterpretable. The
   correct primary ablation is **full decision-schedule replay** (open-loop
   application of the exact journaled ladder). Given P1-2, schedule-replay ≈
   manager is the most likely outcome, and finding that out is the point.
2. **It cannot answer "adaptive choice vs any relaxation"** without companion
   arms: (i) a **random-decision arm** (same cadence, same validator bounds,
   random knob/direction among legal moves) to test whether *any* bounded
   perturbation performs as well; (ii) a **timing-jitter arm** (same ladder,
   ticks shifted ±2 segments) to test whether the band trigger's timing
   carries information. Only if manager ≈ replay > random/jitter > control
   does "adaptive" earn anything.
3. **Given P0-1, all ablation arms must be scored per-motion**, or the
   postmortem clip's bistability will decide every pairwise comparison again.
   Pre-register the endpoint (see P2-2) before running.

### P1-4 — Eval-threshold asymmetry: the fixed relaxed scoreboard is closer to the manager's training condition (task angle f)

The scoreboard is honest in the sense that matters most: thresholds are fixed,
outside the action space, pinned per-knob with structural tests (M1 fix + v4
motion_file pin) [verified], and they are SONIC's stock eval.yaml rather than
anything chosen post hoc. So this is **not** a rigged-eval finding. But there
is a structural asymmetry: the manager's only available actions loosen
training thresholds *toward* the relaxed eval condition (foot 0.2→0.30 vs eval
pin 0.2 excepted — but ee 0.15→0.25 moves toward eval's relaxed ee value),
while control trains strictly and is evaluated leniently. Part of any manager
gain could therefore be train/eval condition alignment (policy experiences
late-clip states that eval credits) rather than better tracking competence.
That is arguably the intended mechanism, but the design can't separate
"learned more" from "distribution matched the test." Cheap disambiguation:
score every persisted snapshot on a **second fixed scoreboard at stock strict
thresholds** (eval-only re-runs from existing snapshots; no retraining). If
manager gains appear only under relaxed eval, the effect is condition
matching. [speculative as to outcome; the asymmetry itself is verified from
the knob directions and eval config handling.]

### P1-5 — The held-out "no leakage / no generalization" reading rests on instruments with ~zero sensitivity (task angle e)

`holdout.py` records only `heldout_success_rate` (full-clip all-or-nothing —
provably 0.0 for every smoke-scale policy, per the project's own
baseline-eval-diagnosis hierarchy) and `heldout_mpjpe_g` (the one metric the
project itself deprecates as survivor-biased) [verified: holdout.py:144–176,
journals]. `progress_rate` — the project's PRIMARY metric — is *not* extracted
from the held-out pass, though the same `metrics_eval.json` machinery
evidently produces it on the curriculum side. Consequences:

- "No measurable generalization transfer" is an **instrument null**, not an
  experimental null; the doc says "lack resolution" but still lists the
  held-out result as a finding.
- "No leakage" (DOES-show #3) is also weak: a leak that moved held-out
  *progress* without flipping full-clip success or moving executed-frame
  mpjpe_g would be invisible to these two metrics. The strong no-leakage
  evidence in this project has always been config-side (pins + structural
  tests), and that part is sound — the metric-side corroboration is close to
  vacuous.
- Fix is nearly free: the held-out `metrics_eval.json` files persist under
  `cmp_*_heldout_eval/`; extract `progress_rate`/`mpjpe_l` retroactively and
  re-issue the held-out table **without re-running anything**. This should be
  done before v5, since it may convert the null into a real (positive or
  negative) transfer measurement for the existing runs.

### P1-6 — Stale caveats misdescribe the experiment's actual scale [verified]

Two statements copied forward from v3 are factually wrong for v4:

1. "Both arms remain on **2 curriculum motions**" (NOT-show list). The
   resolved training configs pin `motion_file: …/robot_curriculum` — a
   **116,924-motion** directory [verified: cmp_control_seed42 s1/s10
   config.yaml; directory listing]; eval is a 64-motion subset.
2. The quantization caveat "progress_rate quantum is 1/(2·2002) ≈ 0.00025"
   (and the report JSON's identical caveat, and the "≈49–68 quanta" arithmetic
   for the s7 jump) assumes a 2-motion eval. The v4 scoreboard averages 64
   per-motion progress values, so the effective quantum is ~32× finer and the
   quanta arithmetic is wrong (in the conservative direction, but wrong).

Neither flips a conclusion, but both show the results doc's self-description
was not re-derived for the new protocol — the same failure mode (stale
assumption surviving a protocol change) that M1 and the registry-default bug
exemplified on the config side.

### P1-7 — Endpoint sensitivity: final-segment snapshot comparisons are fragile

All arm-level comparisons are read at the s10 snapshot, but the doc itself
notes control's late surge (s8–s9, seed 42) and trajectory crossing, and P0-2
shows the sign of the final mean flips on one clip's bistable state at the
arbitrary cutoff. With crossing trajectories, "final value at segment N" is a
lottery on N. Pre-specify a horizon-robust endpoint: mean over the last k
segments, or AUC over segments 6–10, computed per motion then paired.

---

## P2 improvements

### P2-1 — Survival-matched mpjpe proposal (task angle d): sound, with two caveats

The proposed fix (mpjpe on frames both arms survive, or per-frame-index
curves) correctly removes the composition confound and is the right next step.
Prefer the **per-frame-index curve** form: the "frames both survive"
intersection version conditions on a joint-survival event that itself depends
on both policies, so report it as descriptive, not as an unbiased estimator;
and report it per motion (pooled curves reintroduce composition across
motions). Since eval is deterministic from snapshots [design-doc, v3
measured], this is computable retroactively from persisted snapshots if
per-frame errors were logged, else needs one eval-code change for v5.

### P2-2 — Use the paired per-motion structure; it dominates adding seeds

The design already yields 64 paired observations per seed per segment — far
more statistical power than n-of-seeds will ever provide at this cost. For
v5 (and retroactively for v4): pre-register PRIMARY = paired per-motion
progress at the P1-7 endpoint, analyzed with a Wilcoxon signed-rank or a
motion-level bootstrap; report median, trimmed mean, win/loss counts, and a
separate **breakthrough count** (# motions with Δprogress > some prespecified
threshold) so heavy-tail events are a tracked quantity instead of a mean
contaminant. `progress_rate` as a *concept* is the right primary (task angle
g) — the flaw is the unweighted mean as the sole aggregate, plus success_rate
retaining zero resolution.

### P2-3 — Build an empirical noise floor from control

Control's own segment-to-segment deltas (18 per seed) give a null
distribution for "jump size without intervention." Any future "decision X was
followed by jump Y" claim should be reported against that permutation-style
null. This would have flagged P0-1 immediately (control's s8 jump, +0.0153,
exceeds both manager s7 jumps in seed 42).

### P2-4 — Exercise the tripwire on purpose

"0 rollbacks" across v3+v4 means the guard/rollback path has live-run coverage
of zero (doc admits this). Add one deliberate harm-injection run (e.g., force
an absurd loosen, or an lr spike) to demonstrate the tripwire fires, rolls
back, and journals correctly under real training — otherwise the safety story
rests entirely on fake-adapter unit tests.

### P2-5 — Minor accuracy fixes in the doc

- "segments 2–6 manager ≤ control **in both seeds**": false at s6/seed1337
  (0.0858 > 0.0841) [verified: doc's own table]. 
- State explicitly that every nonzero success_rate in the study equals 1/64 =
  the postmortem clip.
- DOES-show #4 gives the manager mean as 0.1004; the table above it says
  0.1003 — trivial, but the doc's standard is byte-level consistency.
- Disk-hygiene fix (next-steps #5) is right; also assert free space *before*
  the campaign, not only per segment, since the failure hit run 4 of 4.

### P2-6 — v5 arm matrix (what would actually be decisive)

Minimum decisive design, all scored per-motion at a pre-registered endpoint,
on ≥2 baseline checkpoints × ≥2 seeds: {control · manager · schedule-replay ·
random-decision · timing-jitter}, plus the strict-threshold second scoreboard
(P1-4) and a resolution-bearing held-out metric (P1-5). If manager ≻ replay is
not shown, the honest conclusion is "a fixed relaxation schedule suffices at
this scale" — which would still be a useful result, and is what the current
data mildly suggest [speculative].

---

## What is sound (credit where due)

- **Scoreboard boundary discipline** is genuinely strong: per-knob eval pins
  with structural tests (v3 M1), the v4 motion_file pin caught *before* the
  run, deterministic eval verified byte-identical, prefix-identity check
  automated in the scorer [verified].
- **Provenance**: every decision journaled with tick/knobs/eval payloads; the
  failed run archived rather than deleted, enabling the determinism check
  this review used [verified].
- **Claim hygiene**: the doc's NOT-show section, the refusal of a value claim
  at n=2, the survivor-bias warnings, and the explicit ablation ask are all
  correct instincts — the failures found here are of *decomposition* (never
  looking under the mean), not of overclaiming culture.
- **Held-out plumbing** (disjoint salt-hashed split, manifest integrity hash,
  process-level separation from the manager's knobs) is the right skeleton;
  it just needs metrics with resolution (P1-5).
- The infra failure handling (journaled `segment_failed`, artifact-presence
  verification exposing the wrong "all rc=0" conclusion) is a model of the
  method catching itself [design-doc].

## Bottom line

Mechanism claims: stand. Evidence claims: DOES-show #2 and #4 are invalidated
as stated (P0-1, P0-2) — the run's honest summary is "loop works end-to-end at
campaign scale; arm-level performance indistinguishable once a single bistable
clip is accounted for; adaptivity untested against a fixed schedule." The v4
data already contain most of what's needed to say something stronger: rescore
per-motion, extract held-out progress_rate from persisted artifacts, and run
the schedule-replay + random arms before buying more seeds.
