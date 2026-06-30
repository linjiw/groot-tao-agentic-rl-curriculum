# Stage-2 — How to Run

Status: **code written + statically validated on a 1×A10G dev box (CPU-only checks).**
Training has **not** been run here — this box lacks Isaac Lab and the GPUs for it. The
artifacts below are ready to launch on a proper Isaac-Lab multi-GPU host.

## Prerequisites (on the training host)
- Isaac Lab 2.3.2 + Isaac Sim installed (per `external/GR00T-WholeBodyControl` README).
- `pip install -e "gear_sonic/[training]"` in the Isaac Lab env.
- Bones-SEED motion lib converted/filtered (`convert_soma_csv_to_motion_lib.py`, `filter_and_copy_bones_data.py`).
- SONIC release checkpoint downloaded (`python download_from_hf.py --training`).
- Multi-GPU host (the released recipe uses `num_envs=4096`, 64+ GPUs). For a first **smoke run**, reduce `num_envs` and `num_learning_iterations` and use a motion subset (see below).

## Step 1 — apply the Stage-2 edits into the WBC checkout
From the repo root:
```bash
bash experiments/stage2-termination-curriculum/apply_and_validate.sh external/GR00T-WholeBodyControl
```
This applies the patch (idempotent), drops `threshold_tighten.yaml` into
`gear_sonic/config/manager_env/curriculum/`, and runs the CPU-only static checks.
(Equivalently: `git -C external/GR00T-WholeBodyControl apply experiments/.../patch/stage2_termination_curriculum.patch` then copy the config.)

## Step 2 — VERIFY the mechanism once on the real host (cheap)
The Step-0 source check already confirmed `TerminationManager.compute()` reads
`params.threshold` live (see `00-env-and-step0-verification.md`). On the real host, confirm
the curriculum term actually fires by logging the live threshold once per N steps for a few
hundred steps and checking it steps 0.30 → 0.22 → 0.15 at the milestones. If it never changes,
the address/shorthand is wrong — fix before spending GPU.

## Step 3 — launch (curriculum vs baseline, matched everything else)
```bash
# CURRICULUM run
accelerate launch gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  num_envs=4096 headless=True \
  manager_env/curriculum=threshold_tighten \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered

# BASELINE run — identical, default empty curriculum
accelerate launch gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  num_envs=4096 headless=True \
  manager_env/curriculum=empty \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
```

**Reduced-scale smoke (single small node):** lower `num_envs` (e.g. `num_envs=256`),
shorten `++algo.config.num_learning_iterations`, and point the motion lib at a small subset,
then check the proxy ordering (does curriculum beat baseline on early episode length / MPJPE
slope?) before committing to a full-scale run.

> Tune the `num_steps` milestones in `threshold_tighten.yaml` to your actual
> `num_learning_iterations`. `common_step_counter` counts env steps across all envs, so with
> 4096 envs the milestones [0, 30000, 80000] are reached very quickly — scale them up for a
> real schedule (e.g. fractions of total iterations × num_steps_per_env × num_envs).

## Step 4 — evaluate (both runs, fixed relaxed eval thresholds)
```bash
accelerate launch gear_sonic/eval_agent_trl.py \
  +checkpoint=<experiment_dir>/model_step_*.pt \
  +headless=True ++eval_callbacks=im_eval
```
`eval_exp.py` can watch the experiment dir and backfill wandb.

## Metrics (the actual readouts)
- **Primary:** final `success_rate` from `im_eval_callback` under **fixed** `eval.yaml`
  thresholds (so curriculum vs baseline are directly comparable). The curriculum run should
  **match** the fixed-strict baseline's final success_rate.
- **Secondary:** longer early episode length; smoother/faster early MPJPE drop.

## Hypothesis
Tightening termination thresholds on a schedule lets the policy learn coarse tracking on
long, survivable episodes first, then refine to strict precision — reaching the baseline's
final precision with faster, more stable early learning and fewer wasted early terminations.

## What this does NOT change
Zero changes to the PPO loop, the actor, the FSQ token interface, or the reward. One curriculum
axis isolated. This is intentional — it is the minimal, lowest-risk first experiment from the
6-stage plan (`docs/design/02-curriculum-rl-plan.md`).
