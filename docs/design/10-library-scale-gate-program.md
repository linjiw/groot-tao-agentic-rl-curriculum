# 10 — The Library-Scale Gate Program: research goal, design, implementation, experiments, validation

**Status: adopted plan (2026-07-09).** Successor to the NARROW-GO verdict (Hermes plan
2026-07-08) and to phase 2's closed-out adaptivity null (`PHASE2_FINAL_REPORT_DRAFT.md`).
Decisions fixed with the user 2026-07-09: primary goal = library-scale tier-0 gate →
teacher-LLM unfreeze only on pass; IsaacLab pilot finishes then pauses (generalization
platform later); target deliverable = **conference paper (ICRA/CoRL/NeurIPS bar)**.

Claim labels (house convention): **[verified]** structural fact checked against
artifacts/code in this repo · **[measured]** number recomputed from raw artifacts ·
**[design]** intent/judgment, not yet evidenced.

---

## 0. Two corrections to the experimental record (2026-07-09, recomputed)

These change what the next experiment must be, so they lead the document.

### 0.1 v4/E1 already trained at library scale — the "2 curriculum motions" caveat is stale

Direct from the run logs [measured]:

| Run | Evidence | Training motions |
|---|---|---|
| v3 (2026-07-02) | `wbc-training-logs/cmp_control_control_s1.log` | 2 |
| baseline_10k (warm-start source) | `baseline_10k.log` | 2 |
| **v4** (2026-07-07) | `cmp_control_seed42_control_s1.log`: "Loaded 116924 motions" | **116,924** (`robot_curriculum`) |
| **E1** (2026-07-08) | `cmp_scripted_seed1337_scripted_s1.log`: same | **116,924** |

Resolved `config.yaml` of every v4-era segment confirms
`motion_file: data/motion_lib_bones_seed/robot_curriculum` [verified]. The
"2 curriculum motions" caveat in `COMPARISON_V4_RESULTS.md` and
`PHASE2_FINAL_REPORT_DRAFT.md` §1/§5.2 described v3 and was carried forward unrevised
(amendment filed in the draft report).

**Reframed conclusion:** the phase-2 adaptivity null was obtained *at library-scale
training* — but with (a) a warm-start checkpoint that had only ever seen 2 motions,
(b) a 500-iter horizon, and (c) **termination thresholds as the lever**. Per-segment
the motion lib loads a 256-motion batch sampled from the 116,924 [verified, logs], so
the training distribution — the thing a curriculum is supposed to shape — was
essentially untouched by the phase-2 action space. The open question is therefore not
"does it work at scale?" but **"does a lever that actually grips the training
distribution (per-step tier-0 control) produce a real effect above the measured noise
floor?"** — exactly the NARROW-GO gate, now sharpened.

This convergence is independently corroborated: the IsaacLab pilot hit the identical
failure mode the same week (its GAPS.md A0: command-range lever ~0 effect on the
held-out metric while training duration dominates). **Lever–metric coupling is now a
named validity gate in this program (V5, §5).**

### 0.2 Fixed-seed replicates are bit-deterministic modulo kernel events — effective n < nominal n

The four seed-42 control replicates [measured, exact float comparison over journals]:
rep2 ≡ rep3 bit-exactly on all 9 shared segments (identical `rew_series` too); rep1 and
rep4 diverge from them only from s2 / at the documented kernel-event segments (E5c).
Distinct final `progress_rate` outcomes: {0.0988, 0.1063, 0.0969} — a **±5% relative
band across effectively-identical runs**. Consequences:

1. Replicate-count arithmetic must count *distinct trajectories*, not launches.
2. The eval-endpoint noise band (~9.4% relative range at this horizon/operating point)
   is consistent with the E5b chaos floor route (window-mean 1.31e-2, τ = 3.9e-2)
   [measured] — two independent measurements agreeing that **any arm effect below
   ~10% relative on final progress_rate is inside fixed-seed noise at this scale**.
3. Seed diversity is mandatory for statistical power; replicates at one seed mostly
   re-measure the same trajectory.

---

## 1. Research goal

> **Primary question (the gate):** Does per-step analytic curriculum control — a
> σ-EMA tracking-reward tolerance (PBHC-style) and/or a per-bin learning-progress
> sampler (EGM-style) — produce a real improvement over stock SONIC adaptive sampling
> when training at library scale (116,924-motion bones-seed curriculum split), above a
> noise floor measured at the same operating point?
>
> **Secondary question (unfrozen only on a pass):** Does an LLM supervising the
> tier-0 controllers' *meta-parameters* (via the registry-mediated meta-knob wiring)
> beat an open-loop replay of its own meta-schedule — i.e., does adaptivity exist at
> the meta level after it died at the knob level?

### Claims ladder for the paper (each with its kill experiment)

| # | Claim | Established by | Dies if |
|---|---|---|---|
| C1 | Tier-0 control beats stock SONIC at library scale, n≥5 paired seeds, above τ | G2→G4 | effect ≤ τ or sign-flips across seeds |
| C2 | LLM meta-supervision beats scripted meta-schedule replay | G5 (gated on C1) | scripted-meta matches manager-meta (the E1 test, one level up) |
| C3 | Gains transfer to a protected held-out split on a resolving per-frame metric | G4 held-out arm | held-out flat/negative while curriculum-side improves |
| M | Methodology: chaos-floor-calibrated equivalence gating + per-motion decomposition + scripted-replay ablation as the standard for run-level curriculum claims | already evidenced (phase 2 + E5b/E5c/E6) | — (survives any C outcome) |

**Honest positioning:** M plus a *negative* C1 is a workshop paper; M + C1 is a solid
conference submission; M + C1 + C2 is the full GACL-lineage story (teacher-LLM
scheduling rewards/tasks/DR — the long-standing goal). The program is sequenced so
money is only spent on C2 after C1 exists.

### Novelty (from doc 09's verified sweep)

- No published system does mid-run, checkpoint-cadence, schema-validated bounded knob
  deltas on a single ongoing robot-RL run under a protected metric with rollback
  (nearest neighbor LAPP: preference labels, no knobs/guardrails) [verified 3-0].
- PBHC validated σ-EMA and EGM validated bin-LP *each alone, on other stacks*; a
  noise-floor-honest head-to-head + combination on SONIC/GR00T-WBC at 116k-motion
  scale, plus the meta-supervision layer, is unclaimed territory [design].
- The negative space is also novel: phase 2 is, to our knowledge, the first
  adversarially-audited demonstration that checkpoint-cadence knob adaptivity
  collapses to a schedule (with the single-motion-breakthrough confound dissected).

---

## 2. Design plan

Architecture = doc 09 §7 three-tier control, instantiated narrowly:

```
tier 2 (checkpoint cadence, FROZEN until G2 passes):
    LLM analyst/actuator proposing META-knob deltas only
    (sigma_ema.ema_rate, sigma_ema.sigma_floor, bin_lp.power, bin_lp.uniform_mix)
    through the SAME static gate as any knob (whitelist/range/max-step/cooldown/pending)
         │  meta_knobs.py wiring [verified: built, 163/163 tests]
tier 1 (per-segment, deterministic; UNCHANGED, fully validated):
    RunManager loop: digest → propose → validate → apply → pinned eval →
    tripwire watch → journal   [verified: 60-segment campaign, 0 mechanism failures]
tier 0 (per-step, analytic; THE NEW EXPERIMENTAL SUBJECT):
    SigmaEMAController  — σ ← min(σ, EMA(tracking error)), monotone, floor-clamped
    BinLPSampler        — |Δ error-EMA| per bin → sampling probs, grounding floor
    [verified: built + unit-tested; NOT yet inserted into SONIC training]
```

### 2.1 What exists (durable assets, all [verified])

| Asset | Where | State |
|---|---|---|
| Engine-agnostic run-manager core (loop/registry/tripwire/digest/journal/equivalence) | `experiments/run-manager-core/` | 163/163 tests; SONIC assembly = `smoke_driver.py` (SmokeDriver on RunManager) |
| Tier-0 controllers + meta-knob registry wiring | `core/controllers.py`, `core/meta_knobs.py` | unit-tested; setters clamp-and-report |
| Config-drift guard (belief vs resolved config.yaml) | `sonic-knob-registry` + `sonic-job-adapter` | 38+20 tests; closes doc 08 §11 am. 8 |
| Measured noise machinery: chaos floor 1.31e-2, τ=3.9e-2, journal gate | `core/equivalence.py`, E5b/E5c/E6 docs | measured at 50-iter/256-env operating point — **must re-measure at G1's** |
| Multi-seed harness + scorer + per-motion decomposition | `run_comparison_multiseed.sh`, `score_comparison_multiseed.py` | validated on 60 segments |
| Held-out protection (salted split, leak-free end-to-end) | `sonic-heldout-watcher`, manifest v0.1.0 | works; its success_rate metric has zero resolution |
| Library data: 116,924 curriculum / 12,861 held-out motions converted | `/workspace/.../motion_lib_bones_seed/` | landed + already exercised by v4/E1 |

### 2.2 What must be built (gap list)

| Gap | Approach | Constraint |
|---|---|---|
| **Tier-0 insertion into live SONIC training** | Hydra-target swap, no submodule edits: reward term `func:` → our wrapper module (σ-EMA reads live error tensor, updates σ); `motion_lib_cfg._target_` → our MotionLibBase subclass overriding `update_adaptive_sampling_probabilities` with Bin-LP | pinned WBC submodule untouched; if impossible without edits → STOP, redesign with user (NARROW-GO risk #3) |
| **Library-native warm start** | fresh ~2,000-iter baseline trained ON `robot_curriculum` (v4's 2k ckpt saw only 2 motions — a distribution jolt at t=0 [measured, §0.1]) | ~2–4 GPU-h once, shared by every arm |
| **Held-out metric with resolution** | per-frame progress / partial-progress on the held-out 64 subset (eval-harness dump + scorer); replaces all-or-nothing success_rate (0.0 everywhere in v4 [measured]) | CPU + eval-only passes |
| **Noise floor at the new operating point** | G1 re-measurement (equivalence.py's own re-measure rule: engine/horizon/operating point changed) | ~1 night GPU |
| **Meta-schedule scripted arm** (for G5) | `ScriptedPolicy` replaying a manager-meta transcript — exists, needs meta-knob action space | CPU |
| **LLM meta-policy** (for G5) | Phase-1 `LLMPolicy` + playbook rows for tier-0 meta-knob interpretation | CPU + API |

### 2.3 Explicitly frozen (unchanged from NARROW-GO)

Antagonist-regret arm · VAE task representations · multi-surface (reward-weight/DR)
expansion beyond the two tier-0 controllers · generative task synthesis · any tier-2
work before G2 passes. IsaacLab track: pilot finishes, gets an honest writeup, pauses
(re-enters at paper time as the generalization platform if C1 holds).

---

## 3. Implementation plan

Phases I0–I4; single A10G (23 GB), so CPU work overlaps GPU windows. Estimates use the
measured 6.7–11.8 min/segment envelope with 1.5× contingency at library scale.

### I0 — Record hygiene + IsaacLab pilot closeout *(now; CPU)*
1. ~~File the §0.1 amendment in `PHASE2_FINAL_REPORT_DRAFT.md`~~ (done 2026-07-09) +
   propagate to `COMPARISON_V4_RESULTS.md` before any citation of either.
2. Let the running IsaacLab scripted arm finish; write its pilot findings doc
   (machinery-validation claims only, per its EVAL_FRAMEWORK §5); pause the track.
3. Commit this doc; update `NEXT.md` to point here.

### I1 — Tier-0 insertion spike *(2–3 days CPU + ~1 GPU-h smoke)* — **the risk item**
1. Verify Hydra-target swap feasibility for (a) a reward-term `func` pointing at a
   module under `experiments/` visible inside the container, (b) a
   `motion_lib_cfg._target_` subclass. Acceptance: a 10-iter smoke run with both
   swapped in, resolved `config.yaml` showing our targets, bit-identical to stock when
   the controllers are configured as no-ops (gate-0 check via `compare_journals`).
2. σ-EMA wrapper: per-step σ update from the live tracking-error tensor; σ journaled
   per segment (state_dict → parsed stream → digest).
3. Bin-LP subclass: replace failure-rate mixing with LP-over-bins; grounding floor
   pinned at construction (never a knob [verified: meta_knobs.py design]).
4. **Stop rule:** if either insertion requires editing the pinned submodule → halt,
   surface to user with the concrete blocker.

### I2 — Measurement upgrades *(parallel, CPU + eval-only GPU)*
1. Per-frame held-out progress metric: eval-harness dump + `score_comparison_multiseed`
   support + unit tests.
2. Scorer: effective-replicate detection (bit-identity collapse before any σ/CI is
   computed — the §0.2 lesson, made structural).
3. Library-native baseline training script + artifact verification.

### I3 — G-series execution *(GPU nights; §4)* — in strict order G0→G1→G2, gate-checked.

### I4 — Conditional: C2 build-out *(only after G2 passes)*
Meta-knob registry file for the four meta-knobs; LLM playbook rows; manager-meta +
scripted-meta arms; then G5.

---

## 4. Experiment plan (G-series)

Naming: "G" (gate) — avoids the E-numbering collision (plan-v5's E6 ≠ the shipped E6
journal-equivalence gate). Cost convention: `GPU-h = segments × 12 min ÷ 60 × 1.5
contingency` at library scale unless measured otherwise. Every experiment gets a
pre-registration block (endpoint, τ, seeds, decision rule) committed **before launch**
— `<NAME>_PREREG.md` beside the driver.

### G0 — Library-native warm start + insertion smoke *(prereq)*
- **Protocol:** train ~2,000 iters on `robot_curriculum` from scratch (seed 42);
  verify eval curve is in the live region on `curriculum_eval64`; run the I1
  no-op-controller bit-identity smoke against it.
- **Cost:** ~2–4 GPU-h. **Kill criterion:** none (infrastructure), but the no-op
  bit-identity check failing blocks everything (means insertion perturbs training).

### G1 — Noise floor + lever sensitivity at the new operating point *(the τ for everything)*
- **Protocol:** (a) 3 launches of the stock config from the G0 checkpoint, 10 segments,
  seed 42 + 1 ε-perturbed chaos run (E5b method: perturb a continuously-acting path,
  fp32-distinct) → distinct-trajectory count, window-mean floor, final-endpoint band →
  `τ_G = calibrate_tau(floor, safety_factor=3)`. (b) **Lever sensitivity (V5):** one
  run each with σ-EMA at an aggressive-but-sane setting and Bin-LP at power=2.0 —
  not to claim value, only to verify the lever moves the endpoint > τ_G *for someone*.
- **Cost:** ~6 runs × 10 seg ≈ 18 GPU-h (2 nights).
- **Kill criteria:** (i) τ_G so large it swallows a 10% effect → surface to user before
  spending G2 (NARROW-GO Phase-A checkpoint, inherited). (ii) *Neither* controller
  moves the endpoint > τ_G in the sensitivity probe → the honest finding is "tier-0
  levers can't move this metric at this budget"; program pivots to writeup + IsaacLab
  generalization of M (do NOT run G2 blind — phase 2's mistake).

### G2 — **The gate:** tier-0 vs stock, thinnest slice *(NARROW-GO centerpiece)*
- **Arms:** {stock} vs {σ-EMA} (single controller — the one with the strongest external
  evidence, PBHC [verified 2-0/3-0]); everything else frozen; identical G0 warm start.
- **Protocol:** 2 arms × 3 seeds × 10 segments; endpoint = final-2-segment mean of
  `eval/progress_rate` on `curriculum_eval64`; per-motion W/L/T decomposition
  mandatory; secondary: survival-matched mpjpe_l, per-frame held-out progress.
- **Cost:** 6 runs × 10 seg ≈ 18 GPU-h (2 nights).
- **Decision rule (pre-registered):** PASS = paired mean delta > τ_G with consistent
  sign across ≥2/3 seeds *after* excluding any single motion contributing >50% of the
  delta (the postmortem-motion rule). FAIL → **project-level NO-GO for the GACL line**:
  close out with M + the honest negative (doc 08:135 prior), pivot to IsaacLab breadth.
- **On pass:** G3 adds Bin-LP as a second arm (+combined) at the same protocol (~18 GPU-h).

### G4 — Confirmatory value campaign *(the paper's Table 1; only after G2/G3 pass)*
- **Protocol:** winning tier-0 config vs stock, **n=5 seeds**, 20 segments (horizon
  doubled to answer the trajectory-crossing question), held-out per-frame metric live,
  AdaStop-style sequential seed addition (stop early if significance resolves).
- **Cost:** ≈ 2 arms × 5 seeds × 20 seg ≈ 60 GPU-h (~5–6 nights; sequenced across a week).
- **Kill criteria:** paired delta ≤ τ_G, or ≥2/5 seeds flip sign, or the single-motion
  exclusion rule erases the effect → C1 dies at confirmatory scale; report both G2 and
  G4 honestly (the disagreement itself is a finding about small-n gates).

### G5 — Tier-2 meta-supervision *(C2; only after C1 exists)*
- **Arms:** {tier-0 fixed meta-params (=G4 winner)} vs {LLM manager-meta} vs
  {scripted-meta replay of the manager's own transcript} — the E1 ablation applied at
  the meta level, pre-registered as the load-bearing comparison.
- **Protocol:** 3 arms × 3 seeds × 20 segments; LLM = analyst/actuator split (doc 09
  am. 5), every prompt digest-hashed into the journal.
- **Cost:** ≈ 9 runs × 20 seg ≈ 54 GPU-h (+API) — spent only with C1 banked.
- **Kill criterion:** scripted-meta matches manager-meta → adaptivity is dead at the
  meta level too; the paper ships C1 + M with the LLM framed as policy-author, not
  supervisor (still a coherent, honest story).

### G6 — Tripwire live-fire *(safety pillar; cheap, schedulable any gap)*
- Plan-v5 E4 unchanged: hostile scripted policy, known-harmful change, expect
  detect→rollback→recovery journaled. ~1 GPU-h. Kill: tripwire fails to fire on a real
  ≥30% eval drop → safety claim reverts to design-only; block G5 (an LLM arm without a
  proven tripwire is not defensible).

### Budget roll-up

| Stage | GPU-h | Cumulative | Calendar (1 A10G, nights) |
|---|---|---|---|
| G0 | 3 | 3 | ½ night |
| G1 | 18 | 21 | 2 nights |
| G2 | 18 | 39 | 2 nights |
| G3 (on pass) | 18 | 57 | 2 nights |
| G4 (on pass) | 60 | 117 | ~6 nights |
| G5 (on pass) | 54 | 171 | ~5 nights |
| G6 | 1 | — | any gap |

Worst-case spend to a *decisive negative*: ~39 GPU-h (G0–G2). Full positive arc to the
paper: ~170 GPU-h ≈ 3 calendar weeks on this box — tight but feasible for a fall
deadline **[design; flag: if C1 passes, requesting a second GPU shortens G4/G5 by 2×].**

---

## 5. Validation plan

The bar: every headline number survives the adversarial review that killed the v4
headline. Named gates V1–V8; a results doc may not be committed unless all applicable
gates pass.

- **V1 — Pre-registration.** Endpoint, τ, seeds, exclusion rules, and the decision
  table are committed before launch (`*_PREREG.md`). Anything not pre-registered is
  labeled exploratory and cannot headline.
- **V2 — Measured noise floor.** No arm comparison without a τ measured at the same
  operating point (engine, horizon, env count, warm start). Fixed-seed replicates are
  collapsed to distinct trajectories before any spread statistic (§0.2, structural in
  the scorer per I2.2).
- **V3 — Per-motion decomposition.** Every arm-level claim is decomposed per motion;
  any single motion contributing >50% of a delta triggers the exclusion re-analysis;
  W/L/T and median paired delta reported alongside means. (This rule alone reversed
  the v4 headline [verified by consequence].)
- **V4 — Scripted-replay ablation.** Any adaptivity claim (G5) requires the open-loop
  replay arm; "beats control" is never evidence of adaptivity, only of the lever.
- **V5 — Lever sensitivity.** Before any ON-vs-OFF campaign, demonstrate the lever can
  move the endpoint > τ for *someone* (G1b). A null on an insensitive lever–metric
  pair is uninformative and may not be reported as "adaptivity doesn't help."
- **V6 — Pinned scoreboard + protected metric.** Eval re-pins every action-space knob
  and the motion set (M1-class structural unit test, existing); held-out split stays
  salted, integrity-checked, invisible to every policy; config-drift verification
  (`verify_against_config`) runs before every decision [verified: shipped].
- **V7 — Artifact-based run verification.** Segment success = journal events +
  snapshot presence + eval-source accounting (`_eval` provenance), never exit codes;
  disk gate + purge active (E0) [verified: shipped, caught the v4 ENOSPC failure].
- **V8 — Adversarial review before commit.** Two independent reviews (data audit +
  methodology) on every results doc; every quantitative claim recomputed from raw
  journals by the assistant before acceptance (house rule). Reviews live beside the
  results doc in-repo.

**Statistical protocol** (in `score_comparison_multiseed.py`, extended per I2): paired-
by-seed analysis only; n=3 → report per-seed table + range, no tests; n=5 → sign test +
Wilcoxon signed-rank + effect size; sequential seed addition AdaStop-style; final-2-
segment mean endpoint; quantization caveat auto-recomputed from motion count; multiple-
comparison discipline = only pre-registered contrasts are confirmatory.

**Reproducibility packaging (paper artifact):** journals + resolved configs + scorer +
prereg docs + this validation checklist ship as the artifact; the run-manager core and
skills are the reusable-code contribution (TAO skill-bank lineage, doc 04 reverse
transfer preserved).

**Paper-bar checklist (ICRA/CoRL/NeurIPS):** measured-τ methodology section (M) ·
n≥5 confirmatory table (G4) · per-motion + held-out decomposition figures · scripted-
replay ablation (G5 or the phase-2 E1 as the motivating negative) · tripwire live-fire
(G6) for the safety claim · limitations section written from the kill-criteria table,
not post-hoc.

---

## 6. Risks & open questions

1. **I1 insertion feasibility** — the program's single hard dependency; stop rule
   defined (§3-I1.4). Mitigation: both insertions are Hydra-target swaps with
   precedent (the live-verified `++...uniform_sampling_rate` override chain).
2. **τ_G may be large at library scale** (more chaotic sampling → wider floor);
   G1 kill-criterion (i) surfaces this before G2 money is spent.
3. **σ-EMA interacts with SONIC's existing failure-EMA sampler** — two adaptive
   mechanisms coupling; G2 keeps stock adaptive sampling ON in both arms (the
   comparison is +σ-EMA, not instead-of), isolating the addition [design].
4. **Horizon**: 10–20 segments from a 2k warm start is still early training; a G4
   pass earns a case for a longer-horizon replication (cluster ask), a fail at 20
   segments is scoped honestly to this budget.
5. **Single-box compute**: G4+G5 ≈ 11 nights serial; deadline pressure → second-GPU
   request is pre-authorized to raise at C1-pass time.
6. **IsaacLab pilot findings** must be written up before pausing or the track's three
   integration bugs + the A0 lever finding rot outside the repo (I0.2).
