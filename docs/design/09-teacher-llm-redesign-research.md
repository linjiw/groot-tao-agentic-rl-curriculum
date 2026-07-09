# 09 — Deep-research input for the teacher-LLM redesign (multi-surface: rewards, tasks, DR)

Status: research synthesis (2026-07-07), produced by an adversarially-verified deep-research
run (20 sources fetched, 100 claims extracted, 25 verified by 3-vote refutation panels:
20 confirmed / 2 refuted / 3 unverified due to infra errors). Input to the v5+ redesign of
doc [`08`](08-curriculum-manager-agent.md)'s policy/teacher, motivated by the v4 negative
result (band policy behaved open-loop; see `COMPARISON_V4_RESULTS.md` post-review amendment)
and by GACL (arXiv 2508.02988, our own prior work): stateful teacher MDP over
(task, performance) history, regret objective via antagonist, VAE task representation,
grounding via ε-alternating real/synthetic sampling — extended here to a teacher that
schedules **rewards, tasks, and domain randomization**, not just termination thresholds.

Claim labels: [verified N-M] = deep-research refutation-panel vote; [unverified] = claim
extracted from a primary source but its verification votes errored (treat as a lead, not
evidence); [design] = this doc's judgment.

---

## 1. Novelty check: the mid-run claim SURVIVES, but must be narrowed

No published 2024–2026 system has an LLM mutating curriculum knobs of a single ongoing
robot-RL run at checkpoint cadence [verified 2-0/3-0/3-0/3-0 across constituent claims]:

- **AURA** (arXiv 2506.02507, ICRA 2026): LLM emits typed, statically-validated YAML per
  curriculum *stage* (full reward functions + DR strategies + PPO config); intervention is
  strictly BETWEEN stages — "human input… does not mutate an ongoing run."
- **Eurekaverse** (arXiv 2411.01775, CoRL 2024): LLM regenerates terrain code between
  iterations of 2000 PPO steps, but spawns 8 parallel runs per iteration and carries the
  best forward — evolutionary selection, not supervision of one run.
- **CurricuLLM** (arXiv 2409.18382, ICRA 2025): evaluator LLM acts between *subtasks*; its
  authors explicitly name within-run adaptivity (ingesting learning curves) as unrealized
  future work [verified 2-0] — third-party confirmation the gap is real.
- **CRAFT** (arXiv 2509.14380): VLM/LLM reward-refinement loop retrains a new policy per
  round. **ULTRA** (arXiv 2505.20671): per-timestep action substitution — different layer.
- **LAPP** (arXiv 2504.15472, TMLR) is the closest prior art and the reason to narrow: its
  LLM IS queried during a live run (every M epochs across 5000-epoch PPO, Go2 + Shadow
  Hand, real deployment) — but it emits *trajectory preference labels* routed through a
  learned preference predictor injected as dense per-timestep reward. No knob deltas, no
  protected metric, no tripwire/rollback [verified 3-0, 3-0].
- **DrEureka** authors explicitly found intra-training adaptation unnecessary in their
  setups (one-shot pre-training reward+DR generation) [unverified — extracted, not voted].

**Restated novelty claim** [design]: *mid-run, checkpoint-cadence, schema-validated bounded
knob deltas over multiple curriculum surfaces of a single ongoing run, under a protected
metric with checkpoint rollback* — not "mid-run LLM supervision" generically. Doc 08 §10
should be amended to cite LAPP as nearest neighbor.

## 2. The axiom-changing finding: analytic inner controllers should own the fast knobs

Doc 08 axiom 1 (cadence–expressiveness law) put the LLM at eval cadence and SONIC's
existing controllers below it. The strongest new robotics-grade evidence says the
architecture wants a **third tier**, and that some surfaces we assigned to the manager
belong to cheap analytic controllers the manager merely *supervises*:

- **KungfuBot/PBHC** (arXiv 2506.12851, NeurIPS 2025, real Unitree G1): adapts the
  tracking-reward tolerance σ in r(x)=exp(−x/σ) *per environment step* via
  σ ← min(σ, EMA(tracking error)) — derived from bi-level optimization. Ablation over four
  fixed σ settings: NO fixed setting is optimal across motions; the adaptive rule is
  near-optimal on all motion types [verified 2-0, 3-0].
- **EGM** (arXiv 2512.19043, preprint, G1 sim): segments motions into **1-second bins**
  with per-bin EMA of composite tracking error → globally normalized sampling
  probabilities (+ uniform-mix ratio and power smoothing; bins act as start frames).
  Removing the module: test E_mpkpe 57.35 → 71.04 mm; removing only the annealing:
  → 62.18 [verified 3-0, 2-0; medium confidence — unreviewed, sim-only, pre-distillation].
- **GMT** (arXiv 2506.14770): per-motion completion level c_i (decay 0.99 on completion)
  drives BOTH sampling probability and a per-motion termination threshold
  E_i — task and termination curricula coupled per-motion inside the run
  [unverified — extracted, not voted].

**Design amendment A** [design]: three-tier control. Tier 0 (per-step, analytic): σ-EMA
reward tolerance, per-bin error-EMA sampling, per-motion completion levels — new inner
controllers we add to SONIC's existing ones. Tier 1 (per-segment, deterministic): band
logic, tripwire, validator — unchanged. Tier 2 (checkpoint/stage cadence, LLM): supervises
tier-0/1 **meta-parameters** (σ init/EMA rate/floor, bin size, uniform-mix ratio,
temperature, band thresholds) and makes **stage-transition decisions**. The LLM emitting
every delta itself was the v4 design's structural reason for behaving like a schedule.

Note SONIC already ships a failure-EMA bin sampler — EGM/GMT are direct upgrades to it
(sub-clip bins; completion-level coupling), implementable as Family-A inner controllers
whose meta-knobs enter the registry.

## 3. Multi-surface scheduling (rewards + tasks + DR): what's known

- **AURA** is the schema template: typed YAML per stage covering reward functions, DR
  strategies, and training config, statically validated "before a single GPU cycle is
  spent"; hardware evidence on three humanoid platforms [verified 2-0, 2-0]. It validates
  our knob-registry approach and extends it to reward/DR surfaces — what it lacks is
  exactly our mid-run loop and guardrails.
- **Ordering is a first-order effect**: EGM introduces DR, complex terrain, and strict
  penalties only in stage 2, after basic tracking is learned; applying all difficulty from
  the start nearly doubles error (110.42 vs 57.35 at equal budget) [verified 3-0; single
  datapoint, sim-only]. → The teacher's action space should include **when each surface
  activates** (stage gates), not just magnitudes within a surface.
- **TransCurriculum** (arXiv 2603.14156, preprint, real Go1): adding curriculum axes with
  the teacher held constant gives per-axis attribution — command-only 27% transfer loss,
  +DR 23%, +terrain 18% [verified 2-0]. Its two headline claims — history-conditioned
  transformer teacher; joint 3-axis scheduling in one run — are [unverified] (verifier
  errors); **if true it is the closest analogue to our stateful multi-surface teacher and
  must be re-verified first** (same lab lineage as GACL: Xiao group).
- **Cautionary**: Eurekaverse's teacher is itself a success-band policy (harder >80%,
  easier <20%) conditioned only on the last iteration — yet it did not go open-loop,
  plausibly because its outputs are *generative code* selected evolutionarily, not
  deterministic bounded deltas [verified 1-1 split — treat as hypothesis]. Together with
  v4, this suggests our failure mode is *deterministic low-entropy action spaces collapse
  to schedules*, pointing at richer/generative actions or history conditioning, not band
  retuning [design].
- **Cadence warning from PBT literature**: greedy short-horizon exploit/mutate decisions
  can underperform plain random search over long horizons (Brax result, arXiv 2506.03225;
  greediness critique, arXiv 2109.13800) [unverified — extracted]. Supports our
  final-2-segment endpoint, longer watch windows, and argues the journal's outcome scoring
  should use a horizon longer than one segment.

## 4. GACL transfer: what maps, what the literature now says

Coverage caveat: the GACL/UED verification angle produced no *confirmed* claims this round
(verifier budget lost to rate limiting), so these are extracted leads [unverified] plus
our own judgment [design]:

- **Stateful teacher over (task, performance) history** — GACL's ablation (−4.2% without
  performance monitoring, −2.0% without task tracking) is the argument for making our
  journal the teacher's *state*, not just audit metadata. Our digest already carries
  decision history; the redesign should condition the policy on the full
  (decision, per-motion outcome) sequence — GACL's s_t^T, instantiated as context for the
  LLM. TransCurriculum [unverified] may be published proof this works on legged robots.
- **Grounding (GACL's biggest ablation, −5.5%)** — the direct analogue in our setting:
  the curriculum motion distribution must keep ε mass on the *target/reference*
  distribution. SONIC's `uniform_sampling_rate` floor IS the grounding knob (uniform over
  the real library = reference distribution); EGM's uniform-mix ratio is the same
  mechanism. **DRED** (arXiv 2402.03479, ICML 2024) is the published successor: VAE-
  grounded level generation with a *scheduled* real/generated mix ratio [unverified] —
  matching GACL's own future-work note that fixed ε should become adaptive. → Make the
  mix/floor ratio a first-class teacher-scheduled knob with a hard lower bound in the
  registry (grounding is a guardrail, not just a knob) [design].
- **Regret/antagonist objective** — the picture sharpened since doc 08 axiom 2 demoted
  PLR: "No Regrets" (arXiv 2408.15099) shows PLR/ACCEL-style *regret approximations*
  (value-loss, MaxMC) do not correlate with true regret — they correlate with success
  rate, prioritizing already-mastered tasks [unverified]. LP-ACRL (arXiv 2601.17428):
  learning-progress sampling reaches 80% success in 1500 iters where PLR doesn't in 3000,
  on a 600-instance legged task space [unverified]. GACL's antagonist-based regret (a
  truer estimator than value-loss proxies) beat baselines on quadruped locomotion — so
  the honest reading [design]: regret-*approximations* are discredited; antagonist-regret
  works but costs a second policy; **per-bin learning progress (Δ of the bin error-EMA)
  is the cheapest signal with the best legged evidence** and harmonizes GACL's
  performance-monitoring lesson with axiom 2. Adopt LP-over-bins as the primary tier-0
  signal; antagonist-regret is a later ablation arm, not the default.
- **Latent task representation** — for motion tracking, the "task space" is the motion
  library; bins/clips are already discrete. A VAE is unnecessary until we do generative
  task synthesis (motion generation is gated externally anyway, doc 04 C3). ADD
  (arXiv 2410.19715, NeurIPS 2024): diffusion generator guided by regret gradient
  [unverified] is the successor lineage if/when that unlocks.

## 5. Attribution & safety: mostly open territory (good for us)

The per-decision-attribution angle (shadow branches, within-run A/B, OPE of curriculum
decisions) surfaced **no published system** — the search found only borrowable statistical
machinery [all unverified — extracted]:

- **AdaStop** (arXiv 2306.10882, TMLR 2024): group-sequential testing that adaptively
  decides how many runs/evals are needed to declare one configuration better —
  directly usable for our arm comparisons (E5) and for deciding when a knob-decision's
  effect is established.
- **HCPI** (Thomas et al., ICML 2015): return a policy change only with a user-specified
  statistical lower-bound guarantee — the pattern for a *confidence-gated apply*: a
  decision ships only if its expected effect clears a bound at confidence δ.
- **Shadow-branch counterfactual evaluation** (branch short segments from the snapshot
  with/without the change at decision ticks) appears to be **unclaimed territory** — it
  would strengthen the novelty claim and directly fixes v4's attribution failure
  (one bistable motion masquerading as a decision effect) [design].

## 6. Refuted / do-not-cite

- Athena-WBC (arXiv 2607.04837) beating a SONIC-recipe baseline: **refuted 0-3**.
- "CurricuLLM makes no LLM calls during training": refuted 1-2 (its evaluator runs
  between subtasks); the correctly-scoped claim (no within-run learning-curve adaptivity)
  is confirmed.
- Athena-WBC's claim that motion-resampling alone cannot close long-tail gaps is
  [unverified] but load-bearing if true — it would cap what the Family-A surface can
  deliver and strengthen the case for per-motion-cluster reward/DR levers. Re-verify.

## 7. Proposed design amendments (for doc 08 §12 / the v6 design)

1. **Three-tier control** (§2): add tier-0 analytic controllers (σ-EMA reward tolerance,
   EGM-style 1s-bin LP sampler, GMT-style per-motion completion levels); LLM supervises
   their meta-parameters + stage gates. Amends axiom 1.
2. **Action space grows to three surfaces** with AURA-style typed schema: reward
   (σ meta-params, penalty-weight ramps via existing `schedule_dict`), task/data
   (bin sampler meta-knobs, uniform-mix/grounding floor), DR (push/friction ranges +
   stage gates). One *surface-atomic* change per tick preserves attribution; stage-gate
   flips are journaled as first-class decisions.
3. **Stateful teacher, GACL-style**: policy context = full journaled
   (decision, expected_effect, per-motion outcome) history + per-bin LP trends. The
   per-motion/per-bin digest upgrade is the prerequisite (already a v5 standing
   requirement from the v4 amendment).
4. **Grounding as guardrail**: hard lower bound on uniform/reference sampling mass in the
   registry; the scheduled mix ratio above the floor is teacher-controlled (DRED/GACL-ε).
5. **Analyst/actuator split** (CRAFT pattern): a free-form LLM analyst reads the digest
   (optionally rendered plots — CRAFT shows VLM-on-training-curves works) and writes a
   diagnosis; a constrained actuator maps diagnosis → schema-validated bounded delta.
   Keeps guardrails while letting the LLM add information beyond the band core.
6. **Confidence-gated outcomes**: score decisions over ≥2-segment horizons (PBT
   greediness warning); adopt AdaStop-style sequential tests for arm-level claims and
   HCPI-style bounds for high-risk decisions; prototype shadow-branch evaluation at
   decision ticks (rare by design, so ~2× cost only at those ticks).
7. **Signal hierarchy**: per-bin learning progress primary; failure-EMA secondary
   (already SONIC-native); antagonist-regret only as a later ablation arm.

## 8. Follow-ups (cheap, ordered)

1. Re-verify the three [unverified] load-bearing claims: TransCurriculum's stateful
   teacher + joint 3-axis scheduling; Athena-WBC's resampling-insufficiency claim.
2. Read TransCurriculum in full (same-lab lineage; closest published analogue).
3. Fold amendments 1–7 into the v5 plan: tier-0 controllers (σ-EMA, bin-LP sampler) are
   CPU-designable now and testable at 2-motion scale; the multi-surface teacher needs
   E8's library scale to be meaningful — consistent with the existing E1/E2 gate.

Sources (primary, arXiv): 2506.02507 (AURA) · 2411.01775 (Eurekaverse) · 2409.18382
(CurricuLLM) · 2509.14380 (CRAFT) · 2504.15472 (LAPP) · 2505.20671 (ULTRA) · 2506.12851
(KungfuBot/PBHC) · 2512.19043 (EGM) · 2506.14770 (GMT) · 2603.14156 (TransCurriculum) ·
2607.04837 (Athena-WBC, partly refuted) · 2408.15099 (No Regrets) · 2601.17428 (LP-ACRL) ·
2402.03479 (DRED) · 2410.19715 (ADD) · 2306.10882 (AdaStop) · 2506.03225, 2109.13800
(PBT cadence/greediness) · Thomas ICML 2015 (HCPI) · 2508.02988 (GACL, ours).
