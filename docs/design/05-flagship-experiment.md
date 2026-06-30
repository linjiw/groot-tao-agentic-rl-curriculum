# 05 — Flagship Experiment: Stage-2 Termination-Threshold Curriculum

**Why this one first.** It is the cheapest high-signal experiment that is fully runnable on this stack with **no new training infrastructure** (pure config + a small `CurriculumCfg` extension), **no LLM/AutoML adapter**, and **no representation-mismatch risk**. It isolates a single curriculum axis with zero changes to the PPO loop, the actor, or the FSQ token interface.

All paths relative to `external/GR00T-WholeBodyControl/`.

## Step 0 — Prerequisite verification (load-bearing — do this FIRST)
Confirm that `gear_sonic/envs/manager_env/mdp/terminations.py` reads `params.threshold` from the **live** term cfg on **each** call (so IsaacLab `modify_term_cfg` writes propagate at episode boundaries via `TerminationManager` get/set_term_cfg). `CurriculumManager.compute` runs **per-RESET** on reset `env_ids` (IsaacLab `manager_based_rl_env.py:356`).

➡️ **If thresholds are cached at setup, this experiment is inert** and must instead schedule via `config.trainer.schedule_dict`.

> **✅ VERIFIED (2026-06-30, against pinned source).** `TerminationManager.compute()`
> (`external/IsaacLab/.../managers/termination_manager.py:167`) calls
> `term_cfg.func(self._env, **term_cfg.params)` — reading `params` **fresh every step** — and
> `set_term_cfg`/`get_term_cfg` exist. The `gear_sonic` termination funcs take `threshold` as a
> live kwarg. **Stage 2 is viable.** Full record + two doc corrections in
> [`../../experiments/stage2-termination-curriculum/00-env-and-step0-verification.md`](../../experiments/stage2-termination-curriculum/00-env-and-step0-verification.md).
>
> **Corrections found during verification:** `modify_term_cfg` is an **IsaacLab** primitive
> (`isaaclab.envs.mdp.curriculums`), re-exported into the `gear_sonic` mdp namespace — not a
> `gear_sonic` function. Its `s.`-shorthand rewrites the first `terminations.` →
> `termination_manager.cfg.`. The `modify_fn` signature is `(env, env_ids, old_value,
> **modify_params)` and should return `NO_CHANGE` to skip redundant writes; the ready-to-apply
> patch ships a `step_curriculum_nochange` helper for that.

## Step 1 — Extend the curriculum dataclass
In `gear_sonic/envs/manager_env/mdp/curriculum.py` (currently only `force_push_curriculum` / `force_push_linear_curriculum`), add fields:
- `anchor_pos_threshold_curriculum`
- `ee_body_pos_threshold_curriculum`
- `foot_pos_xyz_threshold_curriculum`

## Step 2 — Extend the modify_fn injection
Extend the hardcoded `importlib` modify_fn injection block in `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` (the `# curriculum? WARNING HARDCODED` block) to inject `step_curriculum` (`curriculum.py`) as the `modify_fn` for the three new term names.

## Step 3 — Add the curriculum config
Create `gear_sonic/config/manager_env/curriculum/threshold_tighten.yaml` instantiating `gear_sonic.envs.manager_env.mdp.curriculum.CurriculumCfg` with three `CurriculumTermCfg(func=mdp.modify_term_cfg)` terms:

```yaml
_target_: gear_sonic.envs.manager_env.mdp.curriculum.CurriculumCfg

anchor_pos_threshold_curriculum:
  _target_: isaaclab.managers.CurriculumTermCfg
  func: gear_sonic.envs.manager_env.mdp:modify_term_cfg
  params:
    address: 'terminations.anchor_pos.params.threshold'   # 's.'->'_manager.cfg.' shorthand
    modify_fn: gear_sonic.envs.manager_env.mdp:step_curriculum
    modify_params: { values: [0.30, 0.22, 0.15], num_steps: [0, 30000, 80000] }

ee_body_pos_threshold_curriculum:
  _target_: isaaclab.managers.CurriculumTermCfg
  func: gear_sonic.envs.manager_env.mdp:modify_term_cfg
  params:
    address: 'terminations.ee_body_pos.params.threshold'
    modify_fn: gear_sonic.envs.manager_env.mdp:step_curriculum
    modify_params: { values: [0.30, 0.22, 0.15], num_steps: [0, 30000, 80000] }

foot_pos_xyz_threshold_curriculum:
  _target_: isaaclab.managers.CurriculumTermCfg
  func: gear_sonic.envs.manager_env.mdp:modify_term_cfg
  params:
    address: 'terminations.foot_pos_xyz.params.threshold'
    modify_fn: gear_sonic.envs.manager_env.mdp:step_curriculum
    modify_params: { values: [0.35, 0.28, 0.20], num_steps: [0, 30000, 80000] }
```

> `num_steps` milestones (`N1=30k`, `N2=80k`) are placeholders — tune to the total `num_learning_iterations` (default 1e5). Leave `threshold_adaptive=True` on `anchor_pos`/`ee_body_pos` so the squat/sit `down_threshold` relaxation still composes.

## Step 4 — Repoint the env defaults
In `gear_sonic/config/manager_env/base_env.yaml`, change the curriculum default from `curriculum: empty` to `curriculum: threshold_tighten`.

## Run (unchanged invocation)
```bash
# Curriculum run
accelerate launch gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  num_envs=4096 headless=True
  # (AppLauncher args appended; 64+ GPUs for full scale)

# Baseline run — identical, but with curriculum: empty
accelerate launch gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  num_envs=4096 headless=True \
  manager_env/curriculum=empty
```

## Eval (both runs identical, fixed relaxed thresholds)
```bash
accelerate launch gear_sonic/eval_agent_trl.py \
  +checkpoint=<experiment_dir>/model_step_*.pt \
  +headless=True ++eval_callbacks=im_eval
# eval_exp.py watches model_step_*.pt and backfills wandb.
```

## Metrics
- **Primary:** final `success_rate` from `im_eval_callback` (computed under the **fixed** `eval.yaml` thresholds, so curriculum vs baseline are directly comparable). The curriculum run must **match** the fixed-strict baseline's final `success_rate`.
- **Secondary:** longer early episode length and a smoother/faster early MPJPE drop (`im_eval_callback.py`).

## Reduced-scale fallback (affordability)
Lower `num_envs` and `num_learning_iterations`, or restrict to a motion subset, and **report whether the proxy ordering matches** before any full-scale run.

## Hypothesis
Tightening termination thresholds on a schedule lets the policy first learn coarse tracking on long, survivable episodes, then refine to strict precision — reaching the same final precision as the fixed-strict baseline but with faster, more stable early learning and fewer wasted early terminations.
