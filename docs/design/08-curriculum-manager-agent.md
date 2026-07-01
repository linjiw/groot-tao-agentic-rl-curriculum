# 08 — The Curriculum-Manager Agent: Design Plan

**An LLM agent that supervises a live SONIC whole-body-tracking PPO run at checkpoint cadence, tuning the meta-parameters of SONIC's built-in adaptive mechanisms — built as a TAO-style skill bank.**

Status: design draft (2026-07-01). Claims about SONIC internals are **[verified]** against the pinned submodules (WBC `0e35637`); claims from external papers cite arXiv IDs; everything else is **[design]**.

---

## 1. The one-paragraph pitch

SONIC's public training stack has *no staged curriculum* — its released curriculum config is empty, and what it ships instead is three always-on adaptive controllers: per-bin failure-rate-weighted motion sampling (`motion_lib_base.py:2501–2580`), a KL-adaptive learning-rate controller (`ppo_trainer.py:2142–2166`), and content-adaptive termination thresholds. Meanwhile, the LLM-for-RL literature (Eureka → DrEureka → CurricuLLM → AURA, arXiv 2506.02507) stops at the *per-run* boundary: LLMs design rewards/curricula before training and revise between runs, but **no published system supervises a single ongoing RL run**. The Curriculum-Manager Agent fills that gap: a slow LLM outer loop that reads structured training digests every N checkpoints and emits *schema-validated, bounded deltas* to the meta-parameters of SONIC's fast inner controllers — never touching per-step decisions itself. It is buildable today because every observation and action surface it needs already exists in the pinned source (§4, §5), and it is packaged as TAO-convention skills so the same agent harness (Claude Code / any skill-consuming agent) that runs TAO's `autoresearch` can run this.

## 2. Design axioms (from the literature sweep + adversarial review)

These are the load-bearing conclusions from research; each shapes an architectural decision below.

1. **Cadence–expressiveness law.** Across PBT (arXiv 1711.09846), adaptive-KL controllers, TAO autoresearch, and every LLM-HPO system: the more expressive the controller, the coarser its cadence. Scalar controllers act per-update; LLMs act per-run/per-stage. → The LLM acts only at **eval/checkpoint cadence** (every ~500–2000 PPO iterations), and only on inner-controller *meta-parameters*.
2. **The validated curriculum signals for legged robots are success-rate thresholds and learning progress** (ADR arXiv 1910.07113; Rudin terrain promote/demote arXiv 2109.11978; LP-ACRL arXiv 2601.17428). Regret and value-loss signals (PLR/PAIRED/ACCEL) have zero legged-robot evidence and lost head-to-head in LP-ACRL. → Stage-1's PLR-regret sampler (design doc `02`) is **demoted**; the agent's inner signals are failure-rate EMAs and success-band targeting.
3. **Failure-weighted sampling needs a floor and a cap** (SONIC: uniform floor 0.1, cap ≈ β× mean; BeyondMimic arXiv 2508.08241: 10% floor, EMA 0.999/0.001; MaskedMimic: 3e-3 floor). The floor prevents catastrophic forgetting; the cap prevents gradient-variance blowup on broken mocap. → The agent tunes `uniform_sampling_rate`, `adp_samp_failure_rate_max_over_mean`, `bin_size` — never per-clip weights.
4. **Anneal thresholds/sigmas with competence, not wall-clock** (ASAP 1.5→0.3 m, arXiv 2502.01143; PBHC bidirectional level-up/down on episode-length bands, arXiv 2506.12851; GMT per-motion annealing, arXiv 2506.14770). At SONIC's 100M+-frame scale fixed thresholds sufficed — annealing matters most in low-data / hard-motion regimes. → Stage-2's termination curriculum becomes a *competence-gated* schedule the agent steps, replacing the fixed `num_steps` milestones that review flag F3 showed were degenerate at 4096 envs.
5. **Protect the manager's metric from the manager** (METR: o3 reward-hacked 10/10 runs; MLE-bench grading-script manipulation; prompt-level "don't cheat" demonstrably fails). → Curriculum health is judged on a **held-out motion set the agent cannot reweight, filter, or see the composition of**, evaluated at fixed thresholds outside the action space.
6. **Anti-thrash by design** (ADR dual thresholds t_L < t_H; PBHC level-down bands; AIDE one-atomic-change). → Hysteresis bands, sustained-trend triggers (N consecutive evals), per-knob cooldowns, "do nothing" as the explicit default action, and checkpoint-anchored auto-rollback tripwires.
7. **Schema-validated structured output before GPU spend** (AURA's statically-validated YAML workflow; Eureka's 16-candidate selection rules). → Every agent decision is a typed JSON/YAML delta validated against a knob registry (bounds + max step size) *before* it is applied.

## 3. Architecture: two-tier control

```
┌─────────────────────────────────────────────────────────────────┐
│  OUTER LOOP — Curriculum-Manager Agent (LLM, skill-driven)      │
│  cadence: every K evals (~500–2000 PPO iters)                   │
│                                                                 │
│  reads:  digest.json  (structured, trend-annotated)             │
│  emits:  decision.yaml (typed deltas, schema-validated)         │
│  memory: decision journal w/ outcomes (retrieval, AURA-style)   │
└───────────────┬─────────────────────────────▲───────────────────┘
                │ apply (bounded, cooldown,   │ digest builder
                │ atomic, from checkpoint)    │ (wandb + eval parse)
┌───────────────▼─────────────────────────────┴───────────────────┐
│  INNER CONTROLLERS — SONIC's existing adaptive mechanisms       │
│  cadence: every step / 200-step sync / episode reset            │
│                                                                 │
│  • failure-EMA bin sampler      (motion_lib_base.py:2501–2580)  │
│  • KL-adaptive LR               (ppo_trainer.py:2142–2166)      │
│  • schedule_dict interpolator   (scheduler.py:296–353)          │
│  • termination modify_term_cfg  (IsaacLab CurriculumManager)    │
└─────────────────────────────────────────────────────────────────┘
```

Three processes, matching SONIC's own layout **[verified]**:
- **Trainer** — `accelerate launch train_agent_trl.py` (untouched apart from the Stage-2-style config patches already validated).
- **Eval watcher** — the existing separate `eval_exp.py` / `im_eval_callback` process (~250-step cadence) producing `success_rate`, `failed_keys`, per-motion MPJPE (`im_eval_callback.py:741–763`).
- **Manager** — new. Wakes at decision cadence, builds the digest, invokes the LLM with the skill + journal, validates the emitted delta, applies it, arms the rollback tripwire.

### The decision loop (one tick)

1. **Digest** — assemble `digest.json`: eval `success_rate` (+ trend over last M evals), held-out success rate, `failed_keys` diff vs. last tick, failure-rate-vector summary stats (entropy, top-k bins, cap-saturation fraction), episode-length distribution, KL / entropy / grad-norm, per-term reward means, current knob values + time-since-last-change, and the outcomes of the agent's last decisions.
2. **Decide** — LLM reads digest + decision journal, reasons under the skill's playbook, emits `decision.yaml`: either `action: none` (default) or 1 atomic change: `{knob, new_value, rationale, expected_effect, tripwire}`.
3. **Validate** — static check against the knob registry: knob whitelisted, value in hard range, |delta| ≤ max step, cooldown elapsed, ≤1 change per tick. Reject → journal the rejection, do nothing.
4. **Apply** — from the latest checkpoint, via the knob's mechanism (§5). Journal `{tick, digest_hash, decision, applied_at_iter}`.
5. **Watch** — tripwire: if held-out success drops > X% for N consecutive evals post-change → auto-revert to pre-change checkpoint + knob value, mark decision `failed` in the journal (which the LLM sees next tick).

## 4. Observation space (read surfaces — all verified)

| Signal | Source | Cadence | Use |
|---|---|---|---|
| `success_rate` (scalar) | `im_eval_callback.py:747` | eval pass | primary trend; stage gating |
| `failed_keys` (motion keys) | `im_eval_callback.py:741–743` | eval pass | which families are hard; diff across ticks |
| per-motion MPJPE (g/l/pa) + tracking metrics | `im_eval_callback.py:476, 750–763` | eval pass | mastery detection (Stage-4 signal, solved at *manager* cadence — no train-loop plumbing needed) |
| `adp_samp_failure_rate` (per-bin tensor) | `motion_lib_base.py:2531–2552` | 200-step sync | sampler health: entropy, cap-saturation, concentration |
| per-term reward means, KL, entropy, grad-norm, episode length | trainer wandb / stdout | per iteration | Eureka-style component feedback; thrash detection |
| **held-out success rate** | new: eval watcher on a frozen motion subset, fixed relaxed thresholds | eval pass | **the protected metric** (axiom 5) |

Note the Stage-4 plumbing problem from design doc `02` ("eval-side MPJPE can't reach the training-side sampler at training cadence") **dissolves at manager cadence**: the manager reads eval output between ticks and acts through config, so no cross-process tensor plumbing is needed.

## 5. Action space (write surfaces — typed knob registry)

Three disjoint knob families (review caveat `07:114`: data curriculum ≠ hyperparameter schedule — keep separate), each with mechanism, hard range, max step, cooldown:

**Family A — data curriculum** (live-read by sampler at 200-step sync **[verified]**):
| Knob | Mechanism | Range | Max step |
|---|---|---|---|
| `uniform_sampling_rate` | `motion.yaml adaptive_sampling.*` | [0.05, 0.5] | ×1.5 / ×0.67 |
| `adp_samp_failure_rate_max_over_mean` (cap β) | same | [2, 500] | ×2 / ×0.5 |
| `bin_size` | same | {25, 50, 100} | one notch |
| retire/replay mix (Stage-4: `retire_factor`, replay share r) | sampler subclass (design doc `02` Stage 4) | r ∈ [0, 0.1] | ±0.02 |

**Family B — competence-gated schedules** (via `modify_term_cfg` at reset boundaries + `schedule_dict` **[verified]**):
| Knob | Mechanism | Range | Max step |
|---|---|---|---|
| termination thresholds (`anchor_pos`, `ee_body_pos`, `foot_pos_xyz`) | Stage-2 patch's `step_curriculum` slots — but the *agent* steps them on success-band evidence, not `num_steps` (fixes flag F3 structurally) | [0.15, 0.5] m | one 0.02–0.05 m notch, tighten only if held-out success ≥ t_H, loosen if ≤ t_L (ADR dual threshold) |
| DR push scale (Stage-3) | `force_push_linear_curriculum` slot / `modify_env_param` on `push_robot.params.velocity_range` | [0.3, 1.0] | ±0.1 |
| penalty-weight ramps (`feet_acc`, `action_rate_l2`) | `schedule_dict` (`scheduler.py:296–353`, consumed `ppo_trainer.py:1703–1706`) | per-term | segment endpoint edits only |

**Family C — optimizer meta-parameters** (rarely touched; the KL controller is the fast loop):
| Knob | Mechanism | Range |
|---|---|---|
| `desired_kl` | trainer config | [0.005, 0.02] |
| LR clamp bounds | trainer config | within [1e-5, 2e-4] |
| `entropy_coef` | trainer config | ×2 / ×0.5 |

**Explicitly outside the action space:** per-clip sampling weights, eval thresholds, held-out set composition, reward *function code* (reward-code generation is Eureka's per-run regime — a possible later skill, not this agent), anything in the eval watcher.

## 6. Core algorithms

1. **Success-band curriculum stepping (ADR-style dual threshold).** For each schedulable axis: tighten one notch when held-out success ≥ t_H (e.g. 0.85) for N consecutive evals; loosen one notch when ≤ t_L (e.g. 0.5). Hysteresis gap prevents thrash; contraction rule prevents the stuck-at-infeasible-frontier failure ADR is known for. The LLM sets/retunes (t_L, t_H, N, notch size) per axis and decides *which* axis deserves the next move — the band logic itself can run deterministic between ticks.
2. **Sampler-health regulation.** Compute failure-rate-vector entropy + cap-saturation each tick. Concentration too high (few bins dominate, cap saturated) → raise floor or lower β; distribution near-uniform with stagnant success → lower floor / raise β to sharpen. This is the knob-level version of what SONIC's comments say the cap/floor exist for **[verified: code comments cite forgetting, gradient variance, broken-mocap overfit]**.
3. **Mastery retirement with anti-forgetting replay (Stage-4, manager-side).** Per-motion MPJPE + failure EMA below mastery bands for M ticks → retire (×`retire_factor`), keep replay share r; regression on a retired family in held-out evals → un-retire. Reversible by construction.
4. **KL-LR supervision, not replacement.** The ported `kl_adaptive_lr.py` (19/19 tests) is the inner loop; the agent only moves `desired_kl`/clamps when digest shows persistent pathology (KL pinned at a clamp bound for many iters).
5. **Decision journaling + outcome attribution (AURA-style memory).** Every decision carries `expected_effect`; at the next tick the digest builder scores it (met / not met / regressed). The journal is retrieval context — the agent learns which knobs help *in this run*, and one-atomic-change keeps attribution clean.

## 7. Skill decomposition (the TAO-style deliverables)

Following the `tao-curriculum-rl` scaffold conventions (frontmatter, feasibility-tiered sections, [verified]-tagged claims, `references/seams.md`):

| Skill | Contents | Depends on |
|---|---|---|
| `skills/agentic/sonic-run-digest` | digest builder: wandb/stdout/eval parsing → `digest.json` + trend annotation; **pure-CPU, testable now on recorded logs** | none |
| `skills/agentic/sonic-knob-registry` | the typed knob registry (§5) as data + validator; static delta-checking; **pure-CPU, testable now** | none |
| `skills/agentic/sonic-curriculum-manager` | the playbook skill the LLM runs: decision-loop procedure, per-knob heuristics (§6), digest interpretation guide, journal format, tripwire policy | the two above |
| `skills/agentic/sonic-job-adapter` | the SONIC-as-job shim TAO's loop needs (review-verified: `AutoMLRunner` can't drive `accelerate launch` + multi-venv + IsaacSim FileLock as-is, `03:30–31`) — launch/checkpoint/rollback/eval-watcher lifecycle | cluster |
| extend `tao-curriculum-rl` | cross-link: the manager's Family-B schedule logic reuses the already-ported `curriculum_schedule.py` + `kl_adaptive_lr.py` verbatim | done |

## 8. Phased build (hardware-honest)

**Phase 0 — no GPU needed (now):** `sonic-run-digest` + `sonic-knob-registry` skills; a **replay harness** that feeds recorded/synthetic training logs to the manager and checks its decisions against the registry + playbook (unit tests: does it hold in thrash scenarios? does it tighten on sustained success? does the tripwire fire?). This is the same prove-the-mechanism-cheaply pattern the repo already used for Stage-2 static validation and the KL-LR port.

**Phase 1 — A10G-scale closed loop:** run the manager against a *toy* RL run that fits the box — the existing `rlvr_demo.py` REINFORCE loop, or a small non-IsaacLab tracking task — to validate the full tick loop (digest → decide → validate → apply → tripwire) end-to-end with a live optimizer. Measures the manager's *behavior*, not SONIC performance.

**Phase 2 — reduced-scale SONIC (when IsaacLab box exists):** `num_envs=256` smoke, manager ON vs. manager OFF (fixed defaults) vs. hand-designed Stage-2 schedule. Success = manager matches or beats the hand schedule on time-to-success-plateau *without tripping rollbacks*, and every decision journal entry is human-auditable.

**Phase 3 — full-scale ablation (64+ GPU cluster):** the flagship comparison, absorbing the old Stage-2 flagship as one arm: {no curriculum (SONIC default), hand-staged (Stage 2+3), manager-driven}. Also the honest test of axiom 4's prediction — that the manager's edge concentrates in low-data / hard-motion regimes and shrinks at full SONIC scale.

## 9. Risks & honest caveats

- **Value proposition is unquantified** (review `07:111`): SONIC at scale worked with *no* staged curriculum. The manager's bet is regime-dependent (small data, new embodiments, hard-motion subsets). Phase 2/3 are designed to falsify this cheaply before large spend.
- **Eval/train threshold decoupling** (`07:112`): the agent tunes strict training thresholds while success is measured at relaxed eval thresholds. Mitigation: the digest carries *both*; the playbook forbids tightening on eval-side evidence alone.
- **LLM thrash / metric gaming**: mitigated structurally (axioms 5–6), not by prompt. The held-out set and eval watcher are outside the action space *by process boundary*, not by instruction.
- **Cadence starvation**: at 250-step eval cadence and N-consecutive-eval triggers, decisions are rare by design; a short run may see few interventions. That is acceptable — "do nothing" is the correct default for a healthy run.
- **The old Stage-1 PLR track is demoted** on external evidence (axiom 2), not deleted: if LP-ACRL-style learning-progress scoring is wanted later, it slots into the same sampler subclass seam (`motion_lib_base.py:2558–2580`) as one more Family-A knob.

## 10. Novelty statement

Per the literature sweep (mid-2026): LLMs design rewards/curricula *before* runs (Eureka, Text2Reward, CurricuLLM, AURA) and mutate hyperparameters *between* trials (PBT/TAO autoresearch/LLM-HPO); adaptive sampling and threshold annealing run *inside* runs without an LLM (SONIC, BeyondMimic, PBHC, GMT). **No published system places an LLM in supervisory control of a single ongoing RL run.** This design occupies exactly that slot, on a now-public training stack (SONIC, arXiv 2511.07820), with every required read/write surface verified in pinned source, and with the guardrail stack (protected metric, hysteresis, atomic bounded deltas, checkpoint rollback, decision journaling) assembled from the documented failure modes of 2024–26 agentic-ML systems.

## 11. Design amendments from execution (2026-07-01 — Phases 0–2 built and run)

All phases through the Phase-2 mechanism smoke were executed the same day this
design landed. The architecture survived contact intact — `propose(digest,
state, registry)` carried unchanged from the replay harness to the toy LLM
loop to real SONIC training — but execution amended the design in five places.
Evidence: `experiments/curriculum-manager-phase{0,1,2}/`, two adversarial
review rounds (project-aware reviewer agent), and live runs on this box's
A10G (`docs/infra-guide.md`).

1. **Target the *binding* termination axis, not a chosen favorite.**
   [measured] In real 64-env runs, `anchor_pos` — §6.1's illustrative axis
   and the Stage-2 patch's lead term — terminated **~0 episodes** (its
   `threshold_adaptive` path appears to make the strict value non-binding;
   hypothesis, not source-verified) while `foot_pos_xyz`/`ee_body_pos` did
   all the terminating. Loosening a non-binding term is a measured no-op
   (v1 smoke: byte-identical to control). Amendment: the digest carries
   per-term termination fractions (`termination_terms_mean_recent`,
   windowed — single-iteration fractions are too noisy at small env
   counts), and §6.1's band stepping selects the axis by binding fraction.
   Playbook rows 1–2 updated accordingly.

2. **One-change-*pending*, not just one-change-per-tick — and the harness
   enforces it.** [measured] §3's "one atomic change per tick" plus per-knob
   cooldowns was insufficient: a policy can hop to a *different* knob each
   tick (the cooldown itself steers it there), orphaning the previous
   change's tripwire and poisoning its rollback point (v1 smoke, caught by
   review). Amendment: while any change is under tripwire watch, the policy
   is not consulted at all (driver-level gate; registry-level enforcement
   is queued). This subsumes §6.5's attribution discipline.

3. **Knob mutation model is per-run-segment, not within-process.** The
   verified mechanism is: stop → snapshot `last.pt` (it overwrites in
   place) → relaunch from checkpoint with new Hydra overrides
   (`sonic-job-adapter`). §5's Family-A "live-read at 200-step sync"
   remains true of the mechanism but is not how the manager currently
   drives it. Within-process mutation stays future work.

4. **Outcome scoring is currently `survived`, not `met`.** §6.5 specifies
   scoring against `expected_effect`; what's implemented is tripwire
   survival over the watch window. The journal label was renamed to say
   what it is. True effect-scoring (and `digest_hash`/`applied_at_iter`
   journal fields from §3) are queued.

5. **Standing adversarial review is part of the method.** Each experiment's
   results doc goes through a project-goal-aware reviewer agent before
   commit (two rounds on the Phase-2 smoke; both rounds caught real
   defects the tests missed). This operationalizes the repo's honesty norm
   for a project whose central risk is self-deception about its own value
   (§9 caveat 1).

**Phase status (2026-07-01):** Phase 0 ✅ (replay harness, 50 tests) ·
Phase 1 ✅ (real `claude -p` policy on a knob-responsive toy loop; caught a
playbook contraction-rule bug) · Phase 2 infra ✅ + **mechanism smoke ✅**
(manager-ON vs OFF on real SONIC training, 6 segments/arm, pinned seed,
first decision cleanly attributable) — but **value comparison not yet run**:
the box has 2 motions (bones-seed HF-gated), so the protected metric is
unexercised and len/rew gains under loosening are partly definitional.
Phase 2's success criterion (§8) remains open until the real motion library
+ per-segment `im_eval` land. Phase 3 unchanged (cluster).
