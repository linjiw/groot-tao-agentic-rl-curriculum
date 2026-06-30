# 02 — Curriculum-Learning Plan for Whole-Body Motion-Tracking RL

**Thesis: SONIC already implements ~80% of an automatic curriculum, but most hooks are disabled by default** (the experiment selects `manager_env/curriculum/empty.yaml`). This plan *formalizes and extends* existing machinery rather than building greenfield. Every stage is gated on the shipped `im_eval` metrics (`success_rate` and per-joint MPJPE from `gear_sonic/trl/callbacks/im_eval_callback.py`).

## What already exists (the 80%)

- **PHC-style per-bin failure-weighted motion sampler** — *live and active*. Motions are split into fixed `bin_size` (50-frame) bins; per-bin failure rates are tracked; sampling probability ∝ failure rate, with a `uniform_sampling_rate` exploration floor and an `adp_samp_failure_rate_max_over_mean` cap. Implemented in `gear_sonic/utils/motion_lib/motion_lib_base.py` (`update_adaptive_sampling*`), called every step from `mdp/commands.py`, configured in `config/manager_env/commands/terms/motion.yaml` (`adaptive_sampling:` block).
- **Adaptive termination thresholds** — `terminations/terms/anchor_pos_adaptive.yaml` (`threshold_adaptive`) loosen for low-height motions.
- **Staged event presets** — `events/tracking/level0_4.yaml` (push_robot, mass/CoM randomization).
- **Curriculum primitives present but OFF** — `mdp/curriculum.py` defines `terrain_levels_vel`, `step_curriculum`, `linear_curriculum`, and `force_push_*` slots in `CurriculumCfg`; the default config is `curriculum/empty.yaml`.
- **IsaacLab `CurriculumManager`** — three primitives `modify_reward_weight` / `modify_env_param` / `modify_term_cfg`, computed per-reset on reset `env_ids`.

## The 6-stage plan

### Stage 0 — Difficulty-feature scoring & easy→hard prior
**Do NOT seed `init_num_failures`** — the sampler *accumulates* episode/failure counts (never reset), so at 4096 envs a small prior is swamped within the first 200-step sync window. Instead compute an **offline** per-bin difficulty `D ∈ [0,1]` from features `MotionLibRobot` already loads (segment length, per-bin joint-vel/accel RMS, contact-transition count), save a `difficulty_prior` `.pt` sidecar, and inject `D` as a **persistent, decaying multiplicative term** inside `update_adaptive_sampling_probabilities`.
- *New key:* `motion.yaml adaptive_sampling.difficulty_prior_file`.
- *Drop for now:* the MotionBricks latent-density difficulty term (G1Skeleton34 vs SONIC 29-DoF mismatch; offline spike only).
- *Metric:* time-to-first-success-plateau and final `success_rate` vs flat-prior baseline at matched env-steps; log `Spearman(D, converged failure_rate)`.

### Stage 1 — PLR / regret acquisition function
Upgrade the sampler from raw failure-rate to **Prioritized Level Replay**. Subclass `MotionLibBase.update_adaptive_sampling_probabilities` and make the class configurable via `motion_lib_cfg`. Plumbing (the central, previously hand-wavy part):
1. Register a per-step `motion_time_step` / bin-id key in `RolloutStorage` (`register_key`), populated each rollout step.
2. **After** `_compute_returns` (`ppo_trainer.py:2084`), bincount `|GAE return − value|` against the bin id to build a per-bin **positive-value-loss regret EMA**.
3. Keep the existing pipeline (clip to `mean·max_over_mean`, blend with `uniform_sampling_rate`, multiply by bin weights, renormalize) but replace the failure-rate score with `score = (1−ρ)·regret + ρ·staleness`, optionally combined multiplicatively with failure-rate so impossible bins don't dominate.
4. Maintain accumulators over the **full** bin set but index into the GPU-resident active subset each call; sync at the 200-step cadence.
- *New keys:* `scoring_mode ∈ {failure_rate, plr_regret, hybrid}`, `plr_staleness_coef`, `regret_ema_decay`.
- *Metric:* held-out `success_rate`/MPJPE vs failure-rate baseline; fraction of compute spent on never-improving bins.

### Stage 2 — Progressive termination-threshold tightening ⭐ (the flagship)
Fill one empty `CurriculumCfg` slot with a new `config/manager_env/curriculum/threshold_tighten.yaml` instantiating `CurriculumTermCfg(func=mdp.modify_term_cfg, ...)` for three terms (`anchor_pos`, `ee_body_pos`, `foot_pos_xyz`), each with `modify_fn=step_curriculum` and `modify_params={values: [0.30, 0.22, 0.15], num_steps: [0, N1, N2]}`.
- Extend `CurriculumCfg` (`curriculum.py`, currently only `force_push_*`) with the three new fields, and extend the hardcoded `importlib` modify_fn injection block in `modular_tracking_env_cfg.py` to cover them.
- **Prerequisite to verify first:** confirm `terminations.py` reads `params.threshold` from the *live* term cfg each call (so `modify_term_cfg` writes propagate at episode boundaries via `TerminationManager.get/set_term_cfg`). If thresholds are cached at setup, this stage is inert → fall back to `schedule_dict`.
- Leave `threshold_adaptive=True` intact so squat/sit relaxation composes; eval always uses fixed relaxed thresholds for stable comparison.
- Full runnable spec in [`05-flagship-experiment.md`](05-flagship-experiment.md).
- *Metric:* longer early episodes, smoother early MPJPE drop, **same final strict `success_rate`** as the fixed-strict baseline.

### Stage 3 — Progressive domain randomization
Use the `force_push_linear_curriculum` slot (already wired to `linear_curriculum`) with `modify_env_param` on `event_manager.cfg.push_robot.params.velocity_range`. Because `velocity_range` is a dict, add a small `push_scale_curriculum` modify_fn (modeled on `linear_curriculum`) that interpolates a 0.3→1.0 scale and returns the new dict. For mass (`randomize_rigid_body_mass`, a **startup** event), schedule via `env.reinit_dr()` at `reinit_dr_freq` (coarse) or convert to a per-reset event.
- *Metric:* `success_rate` under held-out larger pushes/mass scales (generalization) + MPJPE; more stable early training.

### Stage 4 — Mastered-motion retirement + anti-forgetting replay
In the Stage-1 subclass: (1) **retirement** — multiply bins with EMA failure-rate < `retire_threshold` (~0.02) AND low moving-avg MPJPE by `retire_factor` (~0.1) before renormalizing (reversible — a retired bin re-enters if failure climbs). (2) **anti-forgetting** — replace the 2-way blend with `p = (1−u−r)·active + u·uniform + r·replay`, where `replay` covers retired bins (`r ≈ 0.05`). Tie into `ImResampleCallback` (`motion_resample_frequency=250`).
- **Unsolved plumbing to specify before build:** per-bin MPJPE is computed in the *separate* eval pass (`im_eval_callback` / spawned `eval_agent_trl.py`), not in the training rollout — define how it returns to the training-process `MotionLibRobot` at training cadence, or use only the in-training failure-rate signal for mastery.
- *Metric:* mastered-set `success_rate` must not regress while hard-set rises; track effective number of active bins.

### Stage 5 — `schedule_dict` fine curriculum + consolidation
Populate `config.trainer.schedule_dict` (consumed by `scheduler.update_scheduled_params`; `_navigate_object_path` uses `@`-delimited paths, **not** dotted, schema `{type, seg_vals, seg_steps}`).
- **Remove `init_noise_std` from annealing targets** — it is read only at `Actor` construction and `self.std` is a *learned* `nn.Parameter`; scheduling it does nothing and fights the gradient. Keep only `std_clamp_max` (read live at `get_std`) via the exact `@`-delimited path to `algo_config.std_clamp_max`.
- Ramp penalty weights (`feet_acc`, `action_rate_l2`) from near-zero.
- Add a **modality curriculum** by scheduling `encoder_sample_probs` (init in `commands.py`) from easiest modality (g1) toward harder (teleop/SMPL), leveraging the latent-alignment aux losses.
- Consolidate everything as `sonic_release_curriculum.yaml` and ablate each stage.

## New components to write
- An offline difficulty-scorer producing the `difficulty_prior` `.pt` sidecar (Stage 0).
- A `MotionLibBase.update_adaptive_sampling_probabilities` subclass implementing PLR-regret + retirement/replay (Stages 1, 4).
- `config/manager_env/curriculum/threshold_tighten.yaml` + `CurriculumCfg` field/injection extensions (Stage 2).
- A `push_scale_curriculum` modify_fn for dict-valued DR ranges (Stage 3).
- A consolidated `sonic_release_curriculum.yaml` experiment + per-stage ablation matrix (Stage 5).

## Compute budget (honest)
The per-stage ablation is **~6–12 full runs at `num_envs=4096` on 64+ GPUs for up to 1e5 iterations**. A reduced-scale protocol (fewer envs/iters or a motion subset) must be validated for proxy-to-full-scale correlation before committing. Autoresearch HPO over curriculum knobs and `failed_keys → MotionBricks` targeted synthesis are **future work** (interfaces/conversion pipelines don't exist today).
