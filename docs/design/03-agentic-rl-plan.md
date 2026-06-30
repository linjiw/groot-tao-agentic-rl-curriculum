# 03 — Agentic RL Skills Plan

"Agentic RL" splits into two non-competing tracks that share one substrate — the **FSQ universal token** and **HuggingFace-TRL PPO**.

- **Track (i) — VLA-as-agent:** GR00T (Cosmos-Reason VLM) composes a *frozen library of learned motor skills* by emitting FSQ latent tokens that SONIC decodes.
- **Track (ii) — agent-as-designer:** an LLM/agent (ported from TAO's `autoresearch`/`pbt`) *designs the RL itself* — proposing motion subsets, reward weightings, termination/DR schedules, and HPO, then closing the loop on `im_eval` metrics.

---

## Track (i) — VLA-as-agent

### B1 — Behavior-cloning baseline
GR00T N1.7 (`UNITREE_G1_SONIC` tag, projector index 11, action keys `[motion_token, left_hand_joints, right_hand_joints]`, `action_horizon=40`) emits the `motion_token`; SONIC decodes it with the base controller **frozen**.
- **Interface correction:** the deployed 50 Hz path is the **decoder-only ONNX taking a flat continuous (B, 64)** `encoded_tokens` vector — **not** `UniversalTokenModule.forward_with_external_tokens` (which expects `(B, 2, 32)` FSQ codes). Use the flat-64 deploy path.
- *Metric:* agent-driven `success_rate` vs teacher (SONIC running its own encoder) on held-out motions.

### ⚠️ The dominant structural risk (address, don't defer)
GR00T regresses **continuous** tokens via flow matching, but SONIC's decoder was trained on **FSQ-quantized tokens on a discrete 32-level lattice**. This is a representation mismatch, not mere distribution shift, and the 1.25 magnitude guard (`run_vla_inference.py:291`) does **not** fix it (it rejects outliers, not off-lattice points). **Mitigation:** add a snap-to-lattice / nearest-codebook projection or **quantization-aware BC**, and define an explicit *off-lattice decode-error* metric vs the teacher running its own quantized encoder.

### B2 — Latent-residual RL adapter
Add a residual-action `Actor` variant (action dim = 64) calling `UniversalTokenModule.forward(latent_residual=..., latent_residual_mode='post_quantization')` (`universal_token_modules.py`) with the base decoder frozen via the `distill_only`/`module_mapping` warm-start (`train_agent_trl.py`). Train with the **existing** `TRLPPOTrainer`, a tiny `actor_learning_rate` (the correct Hydra key, under `algo.config.*`), reusing the adaptive-KL controller (`desired_kl=0.01`).
- *Caveat:* a `post_quantization` residual added to already-decoded continuous tokens does **not** resolve the B1 lattice mismatch.
- *Metric:* `success_rate`/MPJPE on composite-task motions vs B1; bounded residual norm; no single-skill regression.

---

## Track (ii) — agent-as-curriculum/HPO/reward designer

### Scoping correction (applies to B3–B5)
This is **not** "reuse `AutoMLRunner` as-is." TAO's `AutoMLRunner` is architecturally bound to the TAO-SDK job path (needs a model skill with `skill_info.yaml` `network_arch`, a TAO `container_image`, `data_sources` mapped to train/evaluate/inference actions, gpu/node spec keys, platform-SDK job submission). SONIC training is `accelerate launch train_agent_trl.py` requiring IsaacSim/IsaacLab across separate venvs plus a C++/TensorRT deploy. **Driving SONIC through the TAO agentic loop requires a substantial new adapter** (custom job submission + a spec/schema shim packaging SONIC as a "model skill"). Drop the "rather than greenfield" framing for these stages.

### B3 — Agentic curriculum / HPO
Build the adapter. Searchable params are Hydra overrides under `algo.config.*` (`actor_learning_rate`, `entropy_coef`, `desired_kl`, `clip_param`, `lam`, `num_mini_batches`) plus `motion.yaml adaptive_sampling.{uniform_sampling_rate, adp_samp_failure_rate_max_over_mean, bin_size}` and the `filter_and_copy_bones_data.py` motion allowlist.
- TAO `autoresearch` stage-3 **training-free pre-screening** is the agent's *internal* candidate filter — implement the no-PPO frozen-checkpoint screen as a custom pre-launch gate.
- Use `eval_fn(rec, train_job_id) → float` (`automl-advanced-monitoring.md`) **only** for the after-training metric: it shells out to `eval_agent_trl.py` (`++eval_callbacks=im_eval`) and returns `success_rate`.
- **Confound to resolve:** searching `actor_learning_rate` AND `desired_kl` is redundant (the KL already drives the adaptive-LR controller) — either drop the LR or pin `schedule='constant'` during the LR sweep.
- **Decoupling caveat:** eval `success_rate` is computed under *fixed relaxed* eval thresholds while the agent may tune *strict training* thresholds — the objective is partly decoupled from that knob.

### B4 — Agentic reward design
Expose reward term weights/σ (`rewards/terms/*.yaml`) as searchable params; the agent edits a generated reward-override YAML (`RewardManager` reads `RewardTermCfg.weight`; live re-tune via `set/get_term_cfg`). Use TAO's LLM-analyzer range-narrowing (env `AUTOML_LLM_ANALYZER_NARROW_RANGES`). A `metric_extractor` parses per-term reward curves from wandb.
- *Metric:* MPJPE improvement with no penalty term dominating the tracking terms.

### B5 — PBT schedule mutation
Ground on TAO `pbt` + SONIC's `schedule_dict` (`ppo_trainer.py`) as the mutation surface (the override hook injects modify_fn once at setup, so `schedule_dict` is the more plausible *runtime*-mutation hook than `step/linear_curriculum`, which are open-loop on `common_step_counter`).
- **Custom state co-copy needed:** SONIC checkpoints + sampler `state_dict` (`motion_lib_base.py`) + `running_mean_std` persist via different paths, and TAO `pbt` mutates *spec params* — valid weight-copying requires custom checkpoint surgery, not arbitrary Python-state copy.
- *(Removed: the IsaacLab `rl_games`/pbt citation — SONIC does not use `isaaclab_rl`; it was a red herring.)*

### B6 — Agentic active sampler
Subclass the inherited `MotionLibBase.update_adaptive_sampling_probabilities`. **Separate the two signals:** use the in-training per-bin `failure_rate` tensor for the inline sampler at the 200-step sync; consume eval-time `failed_keys` (`im_eval_callback.py`, gated on `eval_only`) **only** at the eval-watcher / `im_resample` 250-step cadence (eval runs as a distinct watcher process via `eval_exp.py`, not inline). The agent **blends with** (not replaces) `failure_rate` under the existing `uniform_sampling_rate=0.1` / max-prob caps.
- *Metric:* iterations-to-target on a hard subset vs stock sampler; easy-motion non-regression.

### B7 — Future work: couple the tracks + LoRA the VLM half
**Demote** the "agentic RL on the VLM half" framing — GR00T finetune is flow-matching BC with no reward and no LoRA today. The concrete deliverable is **LoRA-SFT of the Cosmos-Reason backbone**, which requires *adding* a PEFT wrapper to `qwen3_backbone.set_trainable_parameters` (boolean gates only today, no `peft` import), constrained to the kept `select_layer=12` layers. Address the circular **B1↔B3 coupling** (every controller improvement invalidates the B1 BC token dataset) with an explicit interface-version freeze (FSQ `levels=32`, `max_num_tokens=2`) and a defined iteration cadence.

## Quantified comparison plan (before committing B3–B6)
Estimate GPU-hours per `autoresearch` run (one rec = a 4096-env, 64+-GPU PPO run; ~10–50 recs), validate that a reduced-scale proxy correlates with full-scale `success_rate`, and set a **target improvement margin** over the already-well-tuned `sonic_release` defaults — the headline agentic value proposition is currently unquantified.
