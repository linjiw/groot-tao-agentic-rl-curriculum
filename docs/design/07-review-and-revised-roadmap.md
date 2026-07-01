# 07 — Comprehensive Review & Revised Roadmap

*Multi-agent adversarial review (2026-07-01). Three independent agents audited the repo/infra, the TAO→GR00T direction (incl. Cosmos leverage), and the GR00T→TAO reverse transfer (RL/curriculum into TAO). Every claim is tagged **[verified]** (read in pinned source this pass), **[design-doc]** (asserted in `docs/design/*`), or **[speculative]** (agent inference). Submodules were populated and read directly: WBC `0e35637`, Isaac-GR00T `ab88b50`.*

---

## 0. TL;DR

1. **Repo is healthy and honest.** Submodules pinned + populated; the Stage-2 patch **applies clean** and its static-validation claims (AST compile, YAML parse, 6/6 milestone unit checks) **reproduce on a live run**. The "coded + statically validated, never trained" framing is accurate, not over-claimed.
2. **The next-step goal is wrong-ordered.** Making Stage-2 the *first* experiment optimizes for a 64-GPU cluster we don't have, over the two cheapest **project-killing** questions — both of which run on the single A10G we *do* have. Reorder.
3. **Biggest new finding (TAO→GR00T):** `tao-finetune-cosmos-reason` **already ships a first-class LoRA path** (`spec_template_train.yaml:69–80`, `enable_lora` across actions). C1 is now "adapt a working stack," not "port an idea." *But* its verified base is `Cosmos3-Nano` and its conversion helper defaults to `Qwen/Qwen3-VL-8B-Instruct` — Cosmos-Reason2-2B compatibility is an **untested spike**.
4. **Biggest new finding (GR00T→TAO):** SONIC hands TAO a complete, verified RL **chassis** (GAE + clipped PPO + KL-adaptive LR + anti-concentration sampler + `schedule_dict` scheduler) — but **not the engine (the reward)**. TAO already has two clean curriculum seams (`ds_weights_alpha` [verified], `Gr00tTrainer.compute_loss` override [verified]) and a capable agentic brain (`autoresearch`/`pbt`/`eval_fn`).
5. **Load-bearing risk unchanged:** GR00T emits **continuous** flow-matching tokens; SONIC's decoder expects **FSQ-quantized** lattice tokens. Every "drive SONIC from GR00T tokens" path inherits this. It is the repo's own #1 risk and is **measurable on the A10G today** — so measure it first.

---

## 1. Infra / Repo Health — Verdict: healthy, with 3 fixable flags

**Verified state.** HEAD `13dacc7`; submodules pinned exactly (`0e35637` / `ab88b50`) and fully populated. Anchor facts independently re-confirmed in source:
- GR00T backbone `= nvidia/Cosmos-Reason2-2B` — `qwen3_backbone.py:107` (default arg) + `:151` (`Qwen3VLForConditionalGeneration.from_pretrained`); `set_trainable_parameters:174` uses **boolean `requires_grad_` gates only — no `peft`/`LoraConfig` anywhere**. [verified]
- SONIC `= class TRLPPOTrainer(PPOTrainer)` — `ppo_trainer.py:321`, `trl==0.28.0` pinned `pyproject.toml:98`. [verified]

**Flagship artifact audit.** `git apply --check` on `patch/stage2_termination_curriculum.patch` → **applies clean**. `apply_and_validate.sh` run against a throwaway submodule copy → all checks pass (AST-compile ✓, YAML "3 well-formed terms" ✓, 6/6 milestone assertions ✓). `apply_stage2.py` is genuinely idempotent (anchor-string based); anchors verified verbatim in source. Termination term names (`anchor_pos`, `ee_body_pos`, `foot_pos_xyz`) are real and the curriculum endpoints (0.15/0.15/0.20 in `threshold_tighten.yaml:30,40,50`) match `base_adaptive_strict_ori_foot_xyz.yaml`. **This is the strongest, most honest part of the repo.**

**Flags to fix:**
| # | Flag | Evidence | Fix |
|---|---|---|---|
| F1 | **IsaacLab is un-pinned yet load-bearing.** Step-0 proof cites `external/IsaacLab/.../termination_manager.py:167`, but `external/IsaacLab` **does not exist** — real copy is `/workspace/IsaacLab`, captured nowhere. The mechanism is correct (verified `termination_manager.py:168` `value = term_cfg.func(...)`, `set/get_term_cfg` at :216/:231), but the citation path is misleading and the proof silently rots if IsaacLab drifts. | `00-env-and-step0-verification.md:30`, `05:13`; `.gitmodules` lists only 2 repos | Pin IsaacLab (submodule or recorded commit) + fix the cited path. |
| F2 | **`select_layer=12` presented as intrinsic.** Source *default* is `select_layer=-1` (`qwen3_backbone.py:110`); 12 is a config-time value (`gr00t_n1d7.py:47`). | `04:12`, `03:53`, `06:37` | Reword docs to "config sets 12," not "code keeps 12." |
| F3 | **Shipped curriculum config degenerate at scale.** `num_steps=[0,30000,80000]` counts env-steps-across-all-envs; at 4096 envs milestones fire almost instantly. Authors already flag this. | `RUN.md:54–57`, `NEXT.md:101` | Re-scale before any launch; needs `num_learning_iterations`. Config is launch-ready *after* re-scale, not as-is. |

Plus housekeeping: stale "HEAD @ 918719d" refs (`NEXT.md:14`, `2026-06-30.md:68`); `ABSTRACT.md` orphaned from the doc map in `00-overview.md`.

---

## 2. Is Stage-2-first the right goal? — No. Reorder.

The docs justify Stage-2-first as "cheapest high-signal, no new infra, no representation-mismatch risk" (`05:3`). **That is true only conditional on already owning a 64-GPU IsaacLab cluster.** Reality (verified `nvidia-smi`): **1× A10G, 23 GB, no IsaacLab installed** (training path "hard-fails without Isaac Lab"). So Track A's day-1 definition-of-done is gated on procuring hardware the project does not have.

Worse, by the repo's *own* risk register Stage-2 is low-information:
- Its main open question (does a live threshold change propagate?) was **already resolved** by Step-0. What remains is schedule tuning whose headline success is "curriculum **matches** baseline final success_rate" (`05:95`) — a refinement, not a de-risking.
- The two genuinely dangerous unknowns are **cheaper and cluster-free**: Risk #1 (continuous-vs-FSQ mismatch, `06:7`) and Open-Q #4 (off-lattice decode-error metric, `06:38`); plus the Cosmos-Reason2-2B→GR00T strict-loader compatibility spike (`06:36`, Q2/Q5). A "no" on either **invalidates whole tracks** — exactly what you want to learn first, and both plausibly run on 23 GB.

---

## 3. Revised Experiment Ordering

**#1 — Off-lattice FSQ decode-error measurement.** *[A10G, ~hours]*
Load the SONIC FSQ decoder (flat-64 continuous ONNX deploy path, `03:14`); feed (a) teacher FSQ-quantized tokens and (b) perturbed/continuous off-lattice tokens; measure tracking/action degradation vs a snap-to-lattice projection. Answers Open-Q #4 and quantifies the repo's #1 risk with **zero cluster dependency**. If degradation is catastrophic and snap-to-lattice doesn't recover it, the whole VLA-as-agent track needs quantization-aware BC *first* — a roadmap-changing result. Files: `run_vla_inference.py` (1.25 guard), `universal_token_modules.py`.

**#2 — Cosmos-Reason2-2B ↔ GR00T strict-loader + LoRA-attach spike.** *[A10G]*
(a) Run `prepare_cosmos3_vlm_checkpoint.py` **overriding** its default `--vlm-model-name Qwen/Qwen3-VL-8B-Instruct` to the 2B target; check output `config.model_type==qwen3_vl` + shard/tokenizer completeness (script :36–79). (b) Load converted (+`merge_and_unload`ed) dir through GR00T `setup.py`; confirm zero missing/unexpected/mismatched keys (loader raises otherwise, `:106–120`; only `action_head.mask_token` whitelisted). (c) Attach `LoraConfig(r=8, target_modules=['q_proj','v_proj'])` on the kept 12 layers *after* the truncation pop (`:159`); test merge→re-validate. 2B bf16 (~4–5 GB) fits in 23 GB for load/validate. This is the prerequisite spike for the **entire** TAO→GR00T bridge; a "no" is load-bearing and cheap. Settles Open-Q #2/#5.

**#3 — Stage-2 curriculum vs baseline.** *[ONLY once a 64+-GPU IsaacLab cluster exists]*
Keep it as the flagship, demoted to "when cluster exists." **Before** burning GPU: (a) fix the `threshold_tighten.yaml` milestone re-scale (F3); (b) run the cheap live-threshold mechanism check (`RUN.md:23–28`: log 0.30→0.22→0.15) before full training; (c) pin IsaacLab (F1).

**Housekeeping to add now:** wire `apply_and_validate.sh` into a GitHub Action against the pinned submodule (`NEXT.md:84`) — the patch's only real rot risk is upstream WBC drift, and CI catches it.

---

## 4. TAO → GR00T: prioritized incorporations (Cosmos leverage)

Ordered by benefit ÷ risk. Feasibility: **now** / **spike** / **infeasible-today**.

| P | Item | Mechanism · files | Feasibility | WBC benefit |
|---|---|---|---|---|
| **P1** | **DEFT-style agentic failure loop over SONIC's existing sampler** | Wrap the already-working `failed_keys → update_soft_sampling_weight` loop (`motion_lib_base.py:491,517`; `im_eval_callback.py:741,815`; `eval_exp.py:249`) in a KPI-gated DEFT controller (pattern from `tao-run-automl-deft-pipeline` + `tao-analyze-gaps-vlm-bcq`). Keep in-training `failure_rate` (200-step) and eval `failed_keys` as **separate** signals blended under `uniform_sampling_rate=0.1`. | **now** (loop exists; TAO adds orchestration) | Faster iterations-to-target on hard motions without easy-motion regression |
| **P2** | **Cosmos-Reason2-2B as task-QA evaluator for the GR00T backbone** | Build a held-out SONIC task-QA set; run cosmos-reason `evaluate` for BERTScore-F1/accuracy (`cosmos-reason-evaluate.md:44–47`); pair with `open_loop_eval` Action MSE/MAE. | **now** | First quantitative, automatable VLM-half quality signal (GR00T has **no** in-repo `success_rate`); unblocks measuring any later backbone change |
| **P3** | **PEFT branch in GR00T backbone + relax strict loader** | Inject `LoraConfig` on kept 12 layers in `set_trainable_parameters` (`qwen3_backbone.py:174`, after `:159`); either `merge_and_unload` before save **or** extend `mask_token` whitelist to exempt `lora_*` keys (`setup.py:97–120`). Mirror TAO `spec_template_train.yaml:69–80`. | **spike** (capacity-vs-`tune_top_llm_layers` unproven; base compat = P4) | Parameter-efficient VLM adaptation; enables reward-model & language-conditioning spikes |
| **P4** | **Verify Cosmos3-Nano→Qwen3-VL conversion for a 2B base** | `prepare_cosmos3_vlm_checkpoint.py` → 2B; validate qwen3_vl dir; round-trip through GR00T loader (`_reset_rotary_inv_freq` rebuilds non-persistent RoPE at load). | **spike** (prereq for P3's value; negative kills P3/P5 cheaply) | De-risks the entire shared-backbone LoRA story |
| **P5** | **Snap-to-lattice / quantization-aware BC guard on any GR00T→SONIC token path** | Nearest-codebook projection (FSQ `levels=32`, `max_num_tokens=2`) + explicit off-lattice decode-error metric; freeze interface. `run_vla_inference.py`, `universal_token_modules.py`. | **spike** (**the load-bearing risk** — this is Experiment #1 above) | Without it, *every* backbone gain can still fail to drive SONIC |
| **P6** | **Agentic AutoML/PBT adapter for SONIC HPO/curriculum** | Build SONIC-as-model-skill shim (synthetic `skill_info.yaml`/`train.schema.json` over `algo.config.*` + `motion.yaml`), custom `accelerate launch` job backend, `eval_fn→im_eval success_rate` (the one clean hook). Resolve LR/KL confound (pin `schedule='constant'` during LR sweeps); pbt needs checkpoint co-copy surgery. | **spike** (substantial adapter; value unquantified vs `sonic_release` defaults) | Automated curriculum/HPO — only after a proxy↔full-scale correlation study |
| **P7** | **MotionBricks targeted data generation** | — | **infeasible-today** (text-conditioning disabled; 4096-d vs 2048-d mismatch; G1Skeleton34↔29-DoF gap, no conversion pipeline) | Keep the gap-map (P1); mark generation future work |

**Cosmos leverage, feasibility-split:** *feasible-today* = task-QA evaluator (P2) + offline annotation/gap-analysis for the DEFT loop; *research-spike* = reward-model/verifier, LoRA-for-token-emission, language-conditioning; *infeasible-today* = MotionBricks-generated targeted data at the shared backbone.

---

## 5. GR00T → TAO: incorporating RL + curriculum *into* TAO agentic skills

**The asymmetry (the whole point):** SONIC hands TAO a complete, verified RL **chassis** but **not the engine (the reward)**. SONIC's reward is a dense analytic tracking kernel; a VLM reward is a sparse/learned **verifier** — net-new.

**Verified transfer ledger:**
| SONIC component | Transfers to TAO as | Verdict |
|---|---|---|
| GAE / clipped PPO surrogate | Correctness reference (Cosmos-RL is FSDP, not TRL — patterns only) | code-pattern only |
| Analytic-KL adaptive-LR controller (`_adjust_learning_rate_based_on_kl`, ~20 lines) | Portable domain-agnostic LR/pbt-mutation rule | **feasible transfer** |
| Anti-over-concentration sampler (failure cap + uniform floor + max-prob caps) | RL-prompt curriculum sampler | feasible-as-new-component (needs per-example difficulty) |
| `schedule_dict` (`@`-path, linear/segment) | Curriculum-schedule serialization format | **adopt as-is** |
| Dense analytic reward | — | **does NOT transfer — must be replaced by a verifier** |
| SONIC's TRL-PPO library itself | — | infeasible (Cosmos-RL ≠ HF TRL) |

**Two curriculum seams that map cleanly today [verified]:**
- **`data.ds_weights_alpha`** (`factory.py:78–85`) — single-scalar curriculum over the data distribution; anneal it across epochs. Clean on the GR00T VLA half; a schema-extension task on the cosmos-reason VLM half (`mix_ratio`).
- **`Gr00tTrainer.compute_loss` override** (`trainer.py:254–275`) — the exact seam to inject a per-sample regret/difficulty/verifier loss weight.

**TAO's SFT lock is real and at the schema layer:** `train.train_policy.type` enum = exactly `["sft"]` (`train.schema.json:1139–1147`). RL is latent in Cosmos-RL (`dp_shard`/`dp_replicate`, `val/reward_avg` listed-but-latent). "Add RL to TAO" = unlock the enum **+** supply reward/rollout/curriculum.

**Proposed deliverable — a `tao-curriculum-rl` *workflow* skill** (alongside `tao-run-automl`, matching its `type: workflow`):
- **Curriculum-over-SFT (feasible now):** schedule `ds_weights_alpha`/dataset mix; ramp SFT hyperparameters via a `schedule_dict`-style spec; stage transitions gated on `eval_fn`.
- **RL mode (gated on reward_verifier):** explicitly reports RL as *blocked* until (a) the enum is unlocked and (b) a verifier is supplied — the honest "listed-but-latent" status.
- **Agentic loop:** `autoresearch` SpecPrescreener = the "no-PPO pre-screen"; `LLMAnalyzer(narrow_ranges=True)` tightens search every 5 recs; `ResearchProgram` phases with `carry_forward="best"` literally encode a staged curriculum; `eval_fn → success_rate` drives stage transitions (plateau detection) instead of open-loop `common_step_counter` milestones.

**The reward is the pivot.** Most concrete unlock path (feasible spike): a **rule/verifier reward over grounded/structured QA** (multiple-choice, numeric, IoU, referring-expression match) using TAO's *existing* data skills (`tao-generate-referring-expressions`, `tao-generate-image-grounding`, `tao-generate-video-reasoning-annotations`) — i.e. standard RLVR. BERTScore-F1 as a noisier dense objective. Learned reward model = higher research spike.

---

## 6. Consolidated caveats (must-verify before build)
- **No `success_rate` in any runnable GR00T/cosmos-reason eval script by default** — use BERTScore-F1/accuracy or a bespoke verifier success rate.
- **Verified TAO base is `Cosmos3-Nano`, conversion helper defaults to `Qwen3-VL-8B-Instruct`** — 2B-family compatibility is the P4 spike.
- **Compute dominates** — a single SONIC PPO trial is thousands of GPU-hours; agentic-overhead-vs-well-tuned-defaults is unquantified (needs a proxy-scale correlation study).
- **Eval/train threshold decoupling** — agent may tune strict *training* thresholds while `eval_fn` measures relaxed ones (partly-decoupled objective).
- **Two distinct curricula** — SONIC's online sampler (data curriculum) vs open-loop scheduler (`schedule_dict`) must be exposed as separate knob families.

---

## 7. Provenance
Three parallel leaf agents, each reading pinned source (WBC `0e35637`, Isaac-GR00T `ab88b50`) this pass; full agent reports archived under `~/.hermes/cache/delegation/` and `~/reverse-transfer-rl-curriculum-report.md`. Line numbers are best-effort against pinned commits per the repo's own provenance note (`06 §Verification provenance`). An earlier adversarial critique in this project caught hallucinated line numbers and one fabricated metric — the labels above are load-bearing, not decorative.
