# v5 research program — from "promising n=2 signal" to a defensible curriculum-manager method

Status: **DRAFT plan (2026-07-07), not yet reviewed/committed.** Input: v4 results
(`COMPARISON_V4_RESULTS.md`), v3 (`COMPARISON_V3_RESULTS.md`), design doc 08 §11,
`NEXT.md`. Every GPU cost below derives from the single measured throughput figure:
**6.7 min/segment** at 256 train envs / 50 iters / 64 eval envs including both
per-segment eval passes (v3 measured 80 min for 12 segments; v4 confirmed at scale —
seed1337/manager re-run: 10 segments in 1h58m ≈ 11.8 min/seg *with* held-out pass +
degraded disk; we budget with the 6.7 figure and a 1.5× contingency where noted)
[verified]. Claim labels: [verified] = measured in this repo; [design-doc] = from
doc 08 / results docs; [speculative] = this plan's judgment.

## 0. What v5 must convert

v4's asset is exactly one cross-seed-replicated **effect** (s7 progress_rate jump
after `foot_pos_xyz 0.25→0.30` @ iter 300, both seeds) plus a +5.4% relative
cross-seed-mean final progress_rate at n=2 with overlapping ranges and one flipped
seed [verified]. The three claims the project ultimately wants, in order of
increasing cost to test, are:

- **C-adaptive**: the gain comes from the manager *choosing* decisions from run
  state, not from any fixed relaxation schedule.
- **C-value**: the manager beats control on the pre-registered endpoint at n≥5
  seeds with a paired test.
- **C-transfer/no-harm**: gains don't cost tracking quality (survivor confound
  resolved) and eventually show on a held-out split with a resolving metric.

v5 is ordered so the cheapest experiment that can *kill* each claim runs before
anything that merely *supports* it (§4).

Standing method rule carried forward [design-doc, doc 08 §11.5]: every results doc
goes through adversarial review before commit; per-segment artifact verification
(journal events + snapshot presence), not rc codes, decides run success.

---

## 1. Ranked experiment list

Cost arithmetic convention: `GPU-h = segments × 6.7 min ÷ 60`. Disk convention:
each segment leaves ~3.6 GB of checkpoints unless purged [verified, v4 bug #2];
/workspace has ~41 GB free — so **E0 is a hard prerequisite for every multi-run
experiment below** (e.g. E1's 20 segments would otherwise write ~72 GB > 41 GB).

### E0 — Driver disk hygiene + free-space gate  *[CPU only, prerequisite]*

- **Hypothesis**: none — infrastructure. v4 lost one full run (2 h) to `ENOSPC`
  at segment 9 [verified].
- **Protocol**: in `smoke_driver.py`, at segment end delete `last.pt` and
  `model_step_*.pt`, keep `snapshot_*.pt` and eval dirs (the exact manual recovery
  that worked in v4); assert ≥ 8 GB free before launching each segment, else fail
  fast with a journaled `disk_gate_failed` event. Unit tests: purge keeps
  snapshots; gate triggers on a mocked statvfs. Also purge stale v3/v4
  intermediates now (manual, once).
- **Cost**: 0 GPU-h. ~1–2 h dev.
- **Decision settled**: none; unblocks everything. **Kill-criteria**: n/a.

### E1 — Scripted-decision ablation (adaptive vs. fixed schedule)  *[cheapest project-killing question]*

- **Hypothesis (H1)**: the v4 gain is NOT reproduced by open-loop replay of the
  same decision schedule — i.e. adaptivity matters. (Honest prior: [speculative]
  it probably IS largely reproduced, because both seeds walked the same knob
  ladder deterministically [verified], meaning v4's manager already behaved
  nearly open-loop. Either answer is valuable; a kill here is a *pivot*, not a
  death — see decision.)
- **Protocol**: add a third arm `scripted` to `smoke_driver.py`: a
  `ScriptedPolicy` that replays the seed-42/1337 shared ladder verbatim
  (foot_pos_xyz →0.25 @ t2, ee_body_pos →0.20 @ t4, foot_pos_xyz →0.30 @ t6,
  ee_body_pos →0.25 @ t8, foot_pos_xyz →0.35 @ t10) with NO digest reads, NO
  tripwire watch gating (apply unconditionally at those ticks; keep the eval-side
  tripwire armed for safety but decisions are fixed). Same seeds 42 + 1337, same
  checkpoint, same 10 segments — directly comparable against the four existing v4
  runs (reuse them; do not re-run control/manager). Optionally a fourth
  minimal-scripted arm `single`: only foot_pos_xyz →0.30 @ t6 (isolates the s7
  decision) — run on 1 seed first.
- **Cost**: scripted 2 seeds × 10 seg = 20 seg × 6.7 min = **134 min ≈ 2.2 GPU-h**.
  Optional `single` 1 seed × 10 seg = **67 min ≈ 1.1 GPU-h**. Total ≤ 3.3 GPU-h.
- **Decision settled**: is "adaptive manager" the right framing, or is the real
  finding "a good relaxation schedule for this regime"?
  **Kill-criteria**: if scripted final (and s7 jump) matches manager within the
  observed cross-seed range in both seeds → C-adaptive is dead *at 2 motions /
  this checkpoint*; re-frame the project as "the manager is a schedule-discovery
  tool; adaptivity must be demonstrated in a regime where fixed schedules fail"
  (i.e. library scale / distribution shift, E8) and de-prioritize E5's seed
  expansion of the current setup.

### E2 — Random-valid-decision arm (selection skill control)

- **Hypothesis (H2)**: manager beats a policy that emits *random
  validator-passing* decisions at the same cadence — i.e. the band policy's
  binding-axis selection and timing carry information beyond "loosen something
  occasionally."
- **Protocol**: `RandomPolicy(seed)`: at each consultable tick, with p=0.5 emit
  `none`, else pick a uniformly random whitelisted knob + one-notch move in a
  random direction (validator, cooldown, one-pending gate all still enforced —
  this tests *selection*, not the guardrails). 2 policy seeds × 2 env seeds is
  ideal but expensive; start with 2 runs (env seed 42/1337, one policy seed each).
  Compare against existing v4 arms.
- **Cost**: 2 × 10 seg = 20 seg × 6.7 min = **134 min ≈ 2.2 GPU-h**.
- **Decision settled**: does skilled selection exist at all in this regime?
  **Kill-criteria**: random arm final within manager's range on both seeds →
  the *policy* adds nothing over the guardrail scaffold + generic loosening;
  the defensible artifact becomes the safety scaffold (validator/tripwire/
  journal), and policy work (E9 LLM arm) is frozen until a harder regime exists.
  Note E1∧E2 jointly can also show "only the specific schedule matters" (scripted
  wins, random loses) — the most informative outcome.

### E3 — Survivor-confound resolution (survival-matched tracking metrics)  *[near-free]*

- **Hypothesis (H3)**: manager's mpjpe_l/g worsening in v4 is composition
  (surviving into harder late-clip frames), not per-frame degradation.
- **Protocol**: (a) check whether persisted `metrics_eval.json` /eval dirs carry
  per-frame or per-motion-truncated series [verified they persist per segment;
  per-frame availability unknown — check first, CPU]. (b) If absent, extend
  `parse_metrics_eval`/eval harness to dump per-frame-index mpjpe and per-frame
  survival counts, then **re-eval the 40 persisted v4 snapshots** eval-only.
  (c) Score: mpjpe on the frame-index prefix both arms survive
  (min-survival-matched), plus per-frame-index survival and error curves,
  seed-paired. New scorer functions + unit tests in
  `score_comparison_multiseed.py`.
- **Cost**: eval-only pass ≈ 1–2 min each [verified, run_comparison_multiseed.sh
  header] → 40 snapshots × 2 min = **80 min ≈ 1.3 GPU-h** worst case; **0 GPU-h**
  if per-frame data already persists.
- **Decision settled**: can we state "no tracking-quality harm"?
  **Kill-criteria**: if survival-matched mpjpe_l is worse for manager by more
  than the cross-seed range on matched frames in both seeds → the manager trades
  quality for survival; progress_rate alone can no longer headline, and the
  policy needs a quality term (feeds §2 metric upgrades). This can quietly kill
  the current "harmless loosening" story, which is why it runs in tier 1.

### E4 — Tripwire live-fire (adversarial injection)

- **Hypothesis (H4)**: the eval-side tripwire detects a genuinely harmful change
  and rollback restores the pre-change trajectory. Currently unit-test-only
  evidence [verified: 0 rollbacks in all runs].
- **Protocol**: 1 seed, 1 arm, ~6 segments: `HostileScriptedPolicy` applies a
  known-harmful change at t2 — candidates: foot_pos_xyz *tighten* to 0.10, or a
  deliberately out-of-band large loosen applied via a temporarily widened
  registry range (keep the validator honest: widen the test registry, don't
  bypass it). Expected sequence: eval drop ≥ tripwire threshold ×2 consecutive →
  auto-rollback → journal `rolled_back` → post-rollback segment eval returns to
  pre-change value within quantization. If the first candidate doesn't actually
  regress eval (plausible — thresholds may not bind at eval settings
  [speculative]), escalate once; if nothing in the action space can regress eval,
  that ITSELF is a finding: the tripwire is untestable in-domain and the safety
  claim must be softened.
- **Cost**: 1 × 6 seg (+ ≤ 6 more for one escalation) = 6–12 seg × 6.7 min =
  **40–80 min ≈ 0.7–1.3 GPU-h**.
- **Decision settled**: is the safety story (a pillar of the method's
  defensibility, doc 08 axiom 5–6) real?
  **Kill-criteria**: tripwire fails to fire on a real ≥30%-relative eval drop, or
  rollback leaves knob/checkpoint state inconsistent → block all scaling work
  (E8) until fixed; the "guardrailed manager" claim reverts to design-only.

### E5 — Seed expansion to n=5 (the value claim)  *[run only if E1/E2 leave C-adaptive or C-value alive]*

- **Hypothesis (H5)**: manager > control on the pre-registered endpoint with
  seed-paired analysis at n=5.
- **Protocol**: 3 new seeds (e.g. 7, 123, 2024) × {control, manager} × 10
  segments, same checkpoint/protocol (`SEEDS="7 123 2024" bash
  run_comparison_multiseed.sh`). **Pre-register before launch** (in this file,
  amended + dated): PRIMARY endpoint = mean of final-2-segment
  eval/progress_rate on curriculum_eval64 (final-2 mean, not final, to damp the
  s8–s10 oscillation seen in v4 [verified]); analysis = paired-by-seed
  differences, sign test + Wilcoxon signed-rank across n=5, report effect size +
  full per-seed table; SECONDARY = survival-matched mpjpe_l (from E3 tooling) and
  segment-AUC of progress_rate. No other endpoints may be promoted post hoc.
- **Cost**: 3 × 2 × 10 = 60 seg × 6.7 min = **402 min ≈ 6.7 GPU-h** (~1 overnight).
- **Decision settled**: C-value at this scale.
  **Kill-criteria**: paired mean difference ≤ 0, or ≥ 2/5 seeds flip against the
  manager on the primary endpoint → no value claim at 2 motions; the project's
  remaining case rests entirely on E8 (regime where curricula should matter),
  and if E1 also killed adaptivity, recommend writing up the scaffold + negative
  result honestly and pivoting effort to the TAO reverse-transfer track.

### E6 — Horizon extension to s20 (trajectory-crossing question)

- **Hypothesis (H6)**: the manager's advantage persists/grows past s10 (v4 hints
  arms cross repeatedly; control surged late in seed 42 [verified]).
- **Protocol**: resume the four existing v4 runs from their s10 snapshots for 10
  more segments each (driver already supports `--initial-checkpoint`; add
  `--start-tick` continuity for journals or start fresh journals labeled
  `v5_ext`). Manager continues deciding; control continues untouched. Score s1–s20
  curves jointly.
- **Cost**: 4 runs × 10 seg = 40 seg × 6.7 min = **268 min ≈ 4.5 GPU-h**.
- **Decision settled**: is 10 segments a horizon artifact?
  **Kill-criteria**: control catches/passes manager by s20 in ≥ 3/4 runs → the
  effect is "earlier, not better" (still publishable as sample-efficiency IF the
  time-to-threshold metric of §2 favors manager; otherwise the value framing
  narrows sharply).

### E7 — Baseline-checkpoint independence

- **Hypothesis (H7)**: the effect is not specific to
  `wbc_baseline_10k .../model_step_002000.pt` (all v4 runs share it [verified]).
- **Protocol**: train one fresh 2000-iter baseline at a new seed (measured
  ~3.3–3.7 s/iter at 256 envs [verified, infra notes] → 2000 × 3.5 s ≈ 117 min ≈
  **2.0 GPU-h**), then control + manager × 10 segments from it (1 seed).
- **Cost**: 2.0 + (2 × 10 × 6.7/60) = 2.0 + 2.2 = **≈ 4.2 GPU-h**.
- **Decision settled**: checkpoint-generality of the effect.
  **Kill-criteria**: neither the s7-style jump nor a manager advantage appears
  from the fresh baseline → the v4 effect is checkpoint-idiosyncratic (a
  specific loss-landscape moment); any claim must be scoped accordingly.

### E8 — Library scale + held-out metric with resolution  *[gated on bones-seed HF access — external]*

- **Hypothesis (H8)**: in the real-library / low-competence regime (where the
  literature says curricula matter, doc 08 axiom 4), the manager's advantage is
  larger AND becomes visible on a truly held-out split via a resolving metric.
- **Protocol**: prerequisite (CPU, do regardless of access): upgrade the held-out
  pass to **per-frame progress / partial-progress rate** on the held-out 64
  subset (success_rate has zero resolution [verified]) — same eval harness change
  as E3(b). Once `g1.tar.gz` lands: convert/filter per `installation_training.md`;
  new salted-hash curriculum/held-out split via `sonic-heldout-watcher`;
  re-baseline (2k iters ≈ 2.0 GPU-h, arithmetic as E7); then control vs manager,
  2 seeds × 10 segments. Budget segment time at 1.5× contingency (larger motion
  lib → heavier sampler/eval; unmeasured [speculative]): 40 seg × 6.7 × 1.5 =
  402 min ≈ 6.7 GPU-h; total with baseline **≈ 8.7 GPU-h**. Disk: check library
  footprint (23.5 GB tar) against /workspace BEFORE download; may require
  pruning old runs [verified constraint].
- **Decision settled**: C-transfer, and whether the whole method matters where it
  is supposed to matter.
  **Kill-criteria**: manager ≤ control on curriculum-side AND held-out per-frame
  progress at library scale (2 seeds, paired) → the regime-dependence bet
  (doc 08 §9 caveat 1) fails in the direction that matters; the honest conclusion
  is "guardrail scaffold: yes; curriculum value: not demonstrated," and the
  project should ship the scaffold as the deliverable (§3).

### E9 — LLMPolicy arm (only after E1/E2 pass)

- **Hypothesis (H9)**: an LLM policy (Phase-1 `LLMPolicy`, playbook-driven) with
  the eval stream in its digest matches or beats `TrainSideBandPolicy` while
  producing auditable rationales and ≠ ladder-identical decision streams.
- **Protocol**: same harness, `--arm llm` wiring `LLMPolicy` behind `propose()`;
  playbook gets a progress_rate/mpjpe_l interpretation row (doc 08 §11 amendment
  7); 2 seeds × 10 segments; compare against existing arms. Journal every prompt
  digest hash. LLM API latency is wall-clock overhead, not GPU [speculative:
  +≤10 min/run].
- **Cost**: 20 seg × 6.7 min = **≈ 2.2 GPU-h** (+API cost).
- **Decision settled**: is the LLM-in-the-loop framing (the novelty claim, doc 08
  §10) supported by behavior beyond the deterministic band core?
  **Kill-criteria**: LLM stream reduces to the same ladder with worse latency and
  no digest-conditional divergence → the novelty claim narrows to "LLM as policy
  author," and the deterministic policy remains the shipped core.

### Cost roll-up

| Tier | Experiments | GPU-h |
|---|---|---|
| 1 (kill-questions) | E0 + E1 + E2 + E3 + E4 | ≈ 6.4–9.0 |
| 2 (value claim) | E5 + E6 | ≈ 11.2 |
| 3 (generality) | E7 + E8 + E9 | ≈ 15.1 (E8 gated) |

Tier 1 fits in ~1 day of A10G time; tiers 1+2 in ~2.5 days. All sequential
(single 23 GB A10G — never parallel [verified constraint]).

---

## 2. Method / framework improvements

**Policy design beyond `TrainSideBandPolicy`** (build order tied to experiments):
1. `ScriptedPolicy` / `RandomPolicy` / `HostileScriptedPolicy` (E1/E2/E4) — these
   are *method* components, not throwaways: every future policy claim needs the
   scripted and random baselines as standing comparison arms [speculative but
   standard].
2. **Effect-scored policy**: consume the journal's outcome labels (v4 produced
   the first `survived_effect_confirmed` [verified]) — bias toward knob axes with
   confirmed effects, away from `survived`-but-flat ones. Requires finishing the
   queued `expected_effect` scoring (doc 08 §11 amendment 4): score each decision
   against a declared expected metric delta at watch-window close.
3. **Quality-aware objective**: if E3 shows a quality trade, the band policy's
   trigger must consult survival-matched mpjpe_l, not len_mean alone.
4. `LLMPolicy` (E9) last — only once deterministic baselines bound its value.
5. Structural debt from NEXT.md ③/④ [design-doc]: registry verifies believed
   knob values against the resolved config.yaml (kills default-drift bug class);
   registry-level pending-gate. Both CPU; fold into E0's dev window.

**Metric upgrades**:
- Per-frame-index survival + error curves and survival-matched mpjpe (E3) as
  first-class scorer outputs.
- Held-out **per-frame progress rate** replacing all-or-nothing success (E8
  prerequisite; CPU-buildable now).
- **Time-to-threshold** (segments to reach a fixed progress_rate level, e.g.
  0.095) as the sample-efficiency metric — robust to the crossing-trajectories
  problem E6 probes.
- Keep the quantization caveat machinery (quanta = 1/(2·2002) at 2 motions
  [verified]); recompute quanta automatically from motion count when the library
  scales.

**Statistical protocol** (bake into `score_comparison_multiseed.py`):
- Pre-registration block in the plan/results doc: primary endpoint, analysis, and
  seed list fixed BEFORE launch; anything else labeled exploratory.
- Seed-paired analysis only (all arms share seed + checkpoint prefix identity
  [verified]); at n=5: sign test + Wilcoxon signed-rank, report per-seed table
  first, effect size + min..max range, never SD/CI below n=5 (existing honesty
  rule [verified], now with actual tests at n=5).
- Final-2-segment mean as the endpoint (damps single-segment oscillation).
- Multiple-arm correction: with 4+ arms (control/manager/scripted/random) report
  all pairwise paired differences but only the pre-registered
  manager-vs-control and manager-vs-scripted contrasts count as confirmatory.

---

## 3. What to generalize into a reusable skill/framework

The generalizable artifact — defensible even if C-value dies — is the
**guardrailed run-supervision loop**: digest → propose → validate → apply →
fixed-scoreboard eval → tripwire watch → outcome-scored journal, with the four
hard-won invariants as *structural tests*, not prose:
1. pinned scoreboard boundary (every actionable knob re-pinned in eval; the M1
   class of bug) [verified];
2. one-change-pending gate (attribution discipline) [verified];
3. registry-vs-resolved-config verification (default-drift class) [design-doc,
   queued];
4. per-segment artifact verification, not rc codes (v4 bug #3) [verified].

Concrete packaging [speculative, proposed]:
- **`skills/agentic/run-manager-core`** — engine-agnostic: digest schema, knob
  registry format + validator, decision/journal schemas, tripwire/rollback state
  machine, the scorer's honesty rules (per-seed-first, range-not-CI,
  quantization caveats). Extracted from `smoke_driver.py` +
  `sonic-knob-registry` + `sonic-run-digest`; SONIC specifics stay in
  `sonic-job-adapter`. Acceptance: the phase-0 replay harness passes against the
  extracted core with a mock adapter.
- **TAO reverse-transfer connection** (the project's stated second goal): the
  core's adapter seam maps onto TAO jobs — digest from TAO job logs/eval
  artifacts, knobs = spec fields (nested-dict deltas, respecting TAO's
  no-flat-keys rule), apply = relaunch-from-checkpoint via the SDK, tripwire on a
  protected eval metric. First target: extend `tao-curriculum-rl` (already
  scaffolded [verified]) with a "managed curriculum-over-SFT" mode supervising
  `data.ds_weights_alpha` at checkpoint cadence — the exact same per-run-segment
  mutation model amendment 3 verified for SONIC. This positions the manager as a
  TAO **agentic skill** ("supervise a training job"), which is the reverse
  transfer doc 04 promises, and works regardless of E5/E8 outcomes.
- **A results-doc convention skill**: the "What this does NOT show" +
  kill-criteria + adversarial-review workflow is itself a portable method
  component; write it down once as `references/honesty-protocol.md` in the core
  skill.

---

## 4. Explicit ordering — cheapest project-killing question first

| # | Q (what can die) | Exp | GPU-h | Cumulative |
|---|---|---|---|---|
| 0 | (unblock: disk kills runs) | E0 | 0 | 0 |
| 1 | "Adaptivity matters" — dies if a fixed script matches | E1 | 2.2–3.3 | ≈ 3.3 |
| 2 | "No tracking harm" — dies on survival-matched mpjpe | E3 | 0–1.3 | ≈ 4.6 |
| 3 | "Selection is skilled" — dies if random matches | E2 | 2.2 | ≈ 6.8 |
| 4 | "The safety net is real" — dies if tripwire never fires/rolls back | E4 | 0.7–1.3 | ≈ 8.1 |
| 5 | "The effect is real" — dies at n=5 paired | E5 | 6.7 | ≈ 14.8 |
| 6 | "It's not a horizon artifact" | E6 | 4.5 | ≈ 19.3 |
| 7 | "It's not this checkpoint" | E7 | 4.2 | ≈ 23.5 |
| 8 | "It matters where curricula should matter" (+transfer) | E8 | 8.7 (gated) | ≈ 32.2 |
| 9 | "The LLM adds something" | E9 | 2.2 | ≈ 34.4 |

Rationale for the top of the order [speculative, argued]: E1 attacks the
project's *novelty* claim for ~2 GPU-h and reuses all four existing v4 runs as
comparison arms — no other experiment can invalidate as much per GPU-hour. E3 is
near-free and can silently poison every later result if unresolved. E2/E4 close
out the mechanism-level claims before the expensive value campaign (E5) is
allowed to spend an overnight. E8 stays gated on external data access and on at
least one of E1/E2/E5 surviving; §3's skill extraction proceeds in parallel on
CPU regardless.

**Decision gate after tier 1** (write outcome into a v5 results doc before
starting tier 2): if E1 AND E2 both kill (scripted matches, random matches) →
skip E5/E6/E9, do E0-hardening + §3 extraction + E8-when-unblocked, and draft the
honest negative-result writeup. Otherwise proceed to E5 with the pre-registered
protocol.
