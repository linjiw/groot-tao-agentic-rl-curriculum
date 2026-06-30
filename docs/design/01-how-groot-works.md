# 01 — How GR00T Whole-Body Control Works

A technical report on the two halves of the stack: **SONIC** (the low-level RL controller) and **GR00T N1.7** (the high-level VLA), and how they connect. All paths are relative to `external/` submodules pinned at WBC `0e35637` / Isaac-GR00T `ab88b50`.

## 1. Two WBC stacks ship for the Unitree G1

- **Decoupled WBC** (`GR00T-WholeBodyControl/decoupled_wbc/`) — used in GR00T N1.5/N1.6: an RL policy drives the lower body (legs + waist) while IK/interpolation drives the arms.
- **GEAR-SONIC** (`gear_sonic/`) — the current unified controller: a *single* motion-tracking foundation policy covering walking → crawling → teleoperation. "Motion tracking as a scalable training task."

## 2. SONIC RL core — PPO over HuggingFace TRL

**Entry point.** `gear_sonic/train_agent_trl.py` is a Hydra app (`config_name="base"`). It deliberately strips `gear_sonic/` from `sys.path[0]` (lines 16–26) so `from trl import ...` resolves to the **real HuggingFace TRL package**, while the local `gear_sonic/trl/` shadow is used via explicit imports. It parses `config.algo.trl` into TRL's `ScriptArguments`/`PPOConfig`/`ModelConfig` via `HfArgumentParser`, builds an Accelerate `Accelerator`, launches IsaacSim under a `FileLock`, creates a `ManagerBasedRLEnv` wrapped by `ManagerEnvWrapper`, instantiates the universal-token actor/critic, and calls `trainer.train()`.

**Trainer.** `class TRLPPOTrainer(PPOTrainer)` at `gear_sonic/trl/trainer/ppo_trainer.py:321` (with `trl==0.28.0`). The canonical SONIC config composes `TRLAuxLossPPOTrainer` (`trainer: trl_ppo_aux`), which adds universal-token auxiliary losses on top of the PPO loss.

**Training loop** (textbook on-policy clipped PPO):
1. `_rollout_step` collects `num_steps_per_env` transitions stepping the IsaacLab env into a `RolloutStorage`.
2. `_compute_returns` (`ppo_trainer.py:2084`) computes GAE (γ=0.99, λ=0.95), bootstrapping `time_out` terminations by adding `γ·V` to the reward.
3. `num_ppo_epochs` of mini-batch updates: `_compute_ppo_loss` does the clipped surrogate (`clip_param=0.2`), clipped value loss, and entropy bonus (`entropy_coef=0.01`), with an **analytic Gaussian KL** between new/old action distributions feeding `_adjust_learning_rate_based_on_kl` (`desired_kl=0.01`: KL > 2× → lr ÷ 1.5, KL < ½ → lr × 1.5, clamped to `[1e-5, 2e-4]`).
4. Policy + value run in **one** `PolicyAndValueWrapper.forward` call to keep DDP gradient sync correct.
5. `sync_running_mean_std` and `sync_adaptive_sampling` across GPUs, then `callback_handler.on_step_end`.

**Hyperparameters** (`config/algo/ppo_im_phc.yaml`): `num_learning_epochs=5`, `num_mini_batches=4`, `clip_param=0.2`, `gamma=0.99`, `lam=0.95`, `value_loss_coef=1.0`, `entropy_coef=0.01`, `actor_learning_rate=2e-5`, `critic_learning_rate=1e-3`, `desired_kl=0.01`, `schedule=adaptive`, `init_noise_std=0.05`, `num_learning_iterations=100000`. The `sonic_release` experiment overrides `num_steps_per_env=24`, `max_grad_norm=0.1`, `std_clamp_min/max=0.001/0.5`.

### Reward, observation, action

- **Reward** (`config/manager_env/rewards/tracking/base_5point_local_feet_acc.yaml` → `mdp/rewards.py`): per-frame tracking error through **Gaussian kernels** `exp(-err²/σ²)` ∈ [0,1]. The dominant term is `tracking_vr_5point_local` (weight 2.0, σ=0.1), plus body pos/ori/lin-vel/ang-vel terms and penalties (`action_rate_l2`, `feet_acc` w=−2.5e-6, `joint_limit`, `undesired_contacts`). **σ controls strictness → it is a natural curriculum axis.**
- **Observations** are asymmetric: the **policy** sees only deployable noisy proprioception + history (`observations/policy/local_dir_hist.yaml`); the **critic** sees privileged full-body reference geometry + true root linear velocity (`observations/critic/privileged_mf_hist.yaml`); a **tokenizer** group feeds the encoders.
- **Action** = `JointPositionActionCfg` over all 29 joints with `use_default_offset=true` (`actions/terms/joint_pos.yaml`).

### The "universal token" actor (the latent-action interface)

`actor_critic_modules.py:Actor` wraps a `UniversalTokenModule`: G1/SMPL/teleop encoders map their reference observations into a shared latent, an **FSQ** (Finite Scalar Quantization, `vector_quantize_pytorch.FSQ`, `config/actor_critic/quantizers/fsq.yaml`) discretizes it into **2 tokens × 32 levels = 64-dim**, and a decoder maps tokens + proprioception to joint targets. `act()` builds a `torch.distributions.Normal(mean, std)` and samples.

### Terminations & deploy

- **Terminations** (`terminations/tracking/base_adaptive_strict_ori_foot_xyz.yaml`) use **adaptive thresholds** that loosen for low reference-root-height motions (squat/sit) via a running root-height EMA (`terms/anchor_pos_adaptive.yaml`, `threshold_adaptive: True`, `down_threshold: 0.75`).
- **Deploy**: the actor is exported to **two ONNX files** (encoder + decoder). The C++/TensorRT stack runs only the `g1_dyn` decoder at 50 Hz (`sim_dt 0.005`, `decimation 4`); the deployed decoder-only ONNX takes a **flat continuous (B, 64)** `encoded_tokens` vector.

## 3. GR00T N1.7 VLA — Cosmos-Reason backbone + flow-matching head

Two-stage model in `Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py`:

1. **Backbone** (`gr00t/model/modules/qwen3_backbone.py`): `Qwen3VLForConditionalGeneration.from_pretrained("nvidia/Cosmos-Reason2-2B")`, truncated to `select_layer=12` LLM layers, emitting backbone features `[B, T, 2048]`. **Frozen by default** (`tune_llm`/`tune_visual=False`); only the projector / DiT / VL-LayerNorm train, with `tune_top_llm_layers` to unfreeze top-N. **There is no LoRA/PEFT path today** — only boolean `tune_*` gates.
2. **Action head** (`Gr00tN1d7ActionHead`): a flow-matching `AlternateVLDiT` (16 layers) that, at inference, starts from Gaussian noise `[B, action_horizon=40, max_action_dim=132]` and runs `num_inference_timesteps=4` Euler steps to regress **continuous** action chunks (loss = masked velocity-MSE). **No RL, no reward, no autoregressive tokens.**

**Cross-embodiment** = per-embodiment projector indices + `CategorySpecificMLP`s. The bridge tag `UNITREE_G1_SONIC` (projector index 11) has action keys `[motion_token, left_hand_joints, right_hand_joints]`.

## 4. How the two halves connect

GR00T emits the `motion_token`; SONIC's `UniversalTokenModule` decodes/tracks it into whole-body joint commands. The `latent_residual` interface (`post/pre_quantization` modes in `universal_token_modules.py`) and `forward_with_external_tokens` let an external planner steer the controller without retraining.

**⚠️ Representation mismatch (critical):** GR00T regresses *continuous* tokens via flow matching, but SONIC's decoder was trained on **FSQ-quantized tokens on a discrete 32-level lattice**. Predicted tokens need not lie on the lattice. The 1.25 action-bound guard (`gear_sonic/scripts/run_vla_inference.py:291`) rejects *outliers*, not *off-lattice* points. This is a representation mismatch, not mere distribution shift — see [`06-risks-and-open-questions.md`](06-risks-and-open-questions.md).

## 5. MotionBricks (sibling generative model)

`GR00T-WholeBodyControl/motionbricks/` is a real-time latent generative motion model that factorizes motion into "bricks" (root model + pose/tokenizer + smart primitives), tokenizing motion with a multi-head VQ-VAE on the **G1Skeleton34** skeleton (413-dim local representation, `docs/motion_representation.md`). It complements tracking-based SONIC and is a candidate **targeted-data generator** for the curriculum — but text-conditioning is disabled in shipped configs and there is a skeleton/representation gap to SONIC's 29-DoF motion lib (see cross-pollination doc).
