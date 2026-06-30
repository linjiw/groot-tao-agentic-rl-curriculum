# 04 — TAO Cross-Pollination (what transfers, both directions)

Two facts are confirmed at the file level:
1. **GR00T N1.7's VLM half IS `nvidia/Cosmos-Reason2-2B`** (Qwen3-VL, `qwen3_backbone.py`) — the same family TAO's `tao-finetune-cosmos-reason` post-trains. *But* the skill's **verified base everywhere is `nvidia/Cosmos3-Nano`** (`train.schema.json` default `hf_model://nvidia/Cosmos3-Nano`); the string "Cosmos-Reason2-2B" appears **nowhere** in the skill. So whether the skill (and its Cosmos3-Nano→Qwen3-VL conversion helper) accepts Cosmos-Reason2-2B is an **open prerequisite spike**, not an assumption.
2. **SONIC's RL core subclasses HuggingFace TRL's `PPOTrainer`** (`ppo_trainer.py:321`, `trl==0.28.0`).

---

## TAO → GR00T

### C1 — LoRA post-training of the GR00T VLM half
TAO's LoRA-SFT (`LoraConfig` `r=8`, `target_modules=['q_proj','v_proj']`) is a clean conceptual drop-in for GR00T's **missing** PEFT path. Add an opt-in `LoraConfig` branch in `Qwen3Backbone.set_trainable_parameters` (`qwen3_backbone.py`, boolean gates only today), constrained to the kept `select_layer=12` LLM layers (adapters on truncated layers 13..N and on the visual tower are discarded — a real capacity concern).
- **Hard part — the loader:** GR00T's `setup.py` strict-key validation raises on ANY missing/unexpected/mismatched key, so adapters must be `merge_and_unload`-ed before `AutoModel` re-validates, **or** the strict check relaxed for `lora_*` keys.
- **Unverified prerequisite:** that a converted/merged Cosmos3-Nano (or Cosmos-Reason2-2B) directory's `state_dict` is byte-compatible with GR00T's strict key validator (tokenizer/vocab/RoPE-buffer handling could trip it).
- **Metric correction:** there is **no `success_rate`** in any runnable eval script — `open_loop_eval.py` outputs only **Action MSE/MAE**; the README success rates are manually-logged real-robot trials. Use the cosmos-reason `evaluate` BERTScore-F1/accuracy on a held-out SONIC task-QA set as the backbone-only proxy, plus action-MSE from `open_loop_eval`, and flag end-to-end VLA success as requiring manual trials.

### C2 — Agentic AutoML as SONIC's HPO/curriculum brain
TAO `autoresearch` (RAP retrieval + LLM spec proposals + internal pre-screen + keep/discard) and `pbt` (mutate HPs during training) are the HPO / curriculum-schedule brain for SONIC's ~7 PPO knobs and motion-sampler params.
- **Correct keys:** `algo.config.actor_learning_rate` (NOT `algo.actor_lr`) and all knobs under `algo.config.*`; `motion.yaml` keys (`uniform_sampling_rate`, `adp_samp_failure_rate_max_over_mean`) are correct.
- `eval_fn(rec, train_job_id) → float` makes SONIC's own `success_rate`/MPJPE the objective.
- SONIC's checkpointed adaptive-sampling state makes `pbt` resumes coherent *in principle*, but co-copying that state + `running_mean_std` alongside weights needs custom surgery.
- *(See [`03-agentic-rl-plan.md`](03-agentic-rl-plan.md) B3–B5 for the adapter scoping — this is NOT drop-in reuse.)*

### C3 — DEFT failure-driven data loop → SONIC data engine (downgraded to a research spike)
SONIC's failure-weighted sampler (`motion_lib_base.py`) + `im_eval` `failed_keys` (`im_eval_callback.py`) **is a literal robot instance of TAO's DEFT loop** (RCA/gap-analysis → SDG → mining → retrain, KPI-gated). The `failed_keys` → per-bin gap-map mapping is real and well-grounded.
- **BUT the MotionBricks generation half is NOT feasible today:** text-conditioning is disabled in every shipped config (`text_embeddings: null`), out-of-the-box training uses `SyntheticMotionDataset` random tensors, the full GR00T/SONIC MotionBricks pipeline is roadmap, and there is a **dimensional mismatch** — the text slot is 4096-d (matching 7–8B LLMs) while Cosmos-Reason2's hidden size is 2048, so the upstream text-embedding source is unspecified. MotionBricks also uses **G1Skeleton34** (NOT FSQ; multi-head EMA-reset VQ), so generated clips need a non-existent skeleton-conversion + physical-feasibility filter before `convert_soma_csv_to_motion_lib.py` / `filter_and_copy_bones_data.py` ingest.
- **Verdict:** keep the gap-map (it works); mark targeted generation as not-yet-feasible future work.

---

## GR00T → TAO (reverse transfer)

### C4 — SONIC as the working blueprint for TAO's locked RL mode
TAO's `tao-finetune-cosmos-reason` exposes **only SFT** — `train.train_policy.type` is enum-locked to `['sft']`, with RL latent in the Cosmos-RL framework (dropping the key flips to "RL mode → rollout replica → multi-node"; `val/reward_avg` is a listed-but-latent monitoring metric at `SKILL.md:102` / `cosmos-data-specs.md:61`). SONIC is a **working reference implementation** of that RL mode: explicit reward terms (`mdp/rewards.py`), actor-critic (`actor_critic_modules.py`), GAE + clipped surrogate + entropy + analytic-KL-adaptive-LR (`ppo_trainer.py`), and a production anti-over-concentration adaptive sampler.

**Two crucial corrections:**
- This is a **code-pattern analogy, NOT a shared library** — Cosmos-RL is a separate FSDP framework (`dp_shard`/`dp_replicate` sharding, `redis:12800` rollout coordination); only SONIC uses HF TRL.
- The **unsolved crux SONIC does NOT give TAO is the reward.** SONIC's reward is a *dense analytic motion-tracking kernel*; a VLM RL reward is a *sparse/learned verifier*. SONIC transfers scaffolding (GAE, KL-LR, sampler), not the reward — which is the hardest missing piece.

### C5 — IsaacLab CurriculumManager as a template for a first-class TAO curriculum skill
IsaacLab's `CurriculumManager` (compute per-reset at `manager_based_rl_env.py:356`) with its three primitives (`modify_reward_weight` / `modify_env_param` / `modify_term_cfg`) plus SONIC's performance-gated `terrain_levels_vel` is a clean template for a **new TAO curriculum skill** over dataset-mixture knobs (e.g. `ds_weights_alpha`, `finetune_config.py`). The per-epoch curriculum/reward-loss injection seam in GR00T is a `Gr00tTrainer` subclass (`gr00t/experiment/trainer.py`).

### Integration boundary warning
Any `AutoMLRunner` orchestration of SONIC must cross SONIC's **multi-venv** (`.venv_sim` / `.venv_teleop` / `.venv_data_collection` / `.venv_inference`) **+ C++/TensorRT deploy** boundaries — SONIC is **not** a single Python entrypoint. The `FileLock`-guarded IsaacSim launch and wandb-dir parsing are brittle plumbing that any orchestrator must handle.
