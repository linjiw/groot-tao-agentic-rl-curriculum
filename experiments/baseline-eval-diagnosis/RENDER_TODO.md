# RENDER_TODO — commands to run when the GPU frees

Context: the metrics-eval path works (no cameras). The render path
(`render_results=True` → `enable_cameras=True` → experience file
`isaaclab.python.headless.rendering.kit`) segfaults **during Isaac Sim app
startup** in `librtx.scenedb.plugin.so` (crash reports:
`/isaac-sim/kit/data/Kit/Isaac-Sim/5.1/crash_2026-07-02_18-48-00_26198.txt`
and `crash_2026-07-02_18-53-41_26602.txt`). A minimal repro (bare
`AppLauncher(headless=True, enable_cameras=True)`, no eval code) crashes
identically, so this is an Isaac Sim RTX/driver issue on this box (driver
595.71.05, A10G), not a bad eval command. Command-level fix already found:
use `manager_env/recorders=render` (NO leading `+`) — the saved training
config already defines `recorders`, and `+` makes Hydra error with
"Multiple values for manager_env/recorders".

Preflight for every step: `docker exec isaac-lab-base bash -c "pgrep -f eval_agent_trl.py || true"`
must be empty, and run ONE command at a time.

## Step 0 — cheap infra probe (~2 min, do this first)

Confirms whether `--rendering_mode performance` (simplified RTX pipeline: no
reflections, no sampled lighting, no DLSS-G — see
`/workspace/isaaclab/apps/rendering_modes/performance.kit`) gets the camera
app past startup. `eval_agent_trl.py` forwards unknown CLI flags to
AppLauncher's argparse (eval_agent_trl.py:217-221), but for this probe use the
bare launcher:

```bash
docker exec isaac-lab-base bash -c "cd /workspace/GR00T-WholeBodyControl && timeout 240 /isaac-sim/python.sh -c \"
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args,_ = p.parse_known_args([])
args.headless = True; args.enable_cameras = True; args.rendering_mode = 'performance'
app = AppLauncher(args); print('RENDERING_APP_STARTED_OK'); app.app.close()
\" 2>&1 | tail -3"
```

- Prints `RENDERING_APP_STARTED_OK` → proceed to A/B with
  `--rendering_mode performance` kept in the command.
- Still segfaults → skip A/B, go straight to the CPU-safe fallback (C), and
  file the RTX crash as an infra issue (likely driver 595.71.05 vs Isaac Sim
  5.1; the container's render path was never verified).

## A — render OUR baseline (model_step_010000), 2 envs = both motions

```bash
docker exec isaac-lab-base bash -c "cd /workspace/GR00T-WholeBodyControl && nohup /isaac-sim/python.sh gear_sonic/eval_agent_trl.py --rendering_mode performance +checkpoint=/workspace/wbc-training-logs/baseline/wbc_baseline_10k-20260701_232851/model_step_010000.pt +headless=True ++eval_callbacks=im_eval ++run_eval_loop=False ++num_envs=2 ++metrics_file=/workspace/wbc-training-logs/eval_baseline/step_010000/metrics_eval.json ++manager_env.config.save_rendering_dir=/workspace/wbc-training-logs/diagnosis/render_baseline ++manager_env.config.render_results=True ++manager_env.config.env_spacing=10.0 manager_env/recorders=render ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False > /workspace/wbc-training-logs/diagnosis/render_baseline/render.log 2>&1 &"
```

## B — render the RELEASE checkpoint (reference for "what good looks like")

Requires `/workspace/GR00T-WholeBodyControl/sonic_release/config.yaml`
(already downloaded from HF this session — present) and the release metrics
file from `/workspace/wbc-training-logs/diagnosis/eval_release/metrics_eval.json`.

```bash
docker exec isaac-lab-base bash -c "cd /workspace/GR00T-WholeBodyControl && nohup /isaac-sim/python.sh gear_sonic/eval_agent_trl.py --rendering_mode performance +checkpoint=/workspace/GR00T-WholeBodyControl/sonic_release/last.pt +headless=True ++eval_callbacks=im_eval ++run_eval_loop=False ++num_envs=2 ++metrics_file=/workspace/wbc-training-logs/diagnosis/eval_release/metrics_eval.json ++manager_env.config.save_rendering_dir=/workspace/wbc-training-logs/diagnosis/render_release ++manager_env.config.render_results=True ++manager_env.config.env_spacing=10.0 manager_env/recorders=render ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False > /workspace/wbc-training-logs/diagnosis/render_release/render.log 2>&1 &"
```

Notes for A/B:
- Poll with `tail` on the render.log; a healthy run prints the boxed GPU table,
  then `=== Start recording video to ... ===`, runs the render loop
  (release: full 2002 steps ≈ several min at 2 envs; baseline: ~110 steps),
  then `Rendering only. Reached the end of the evaluation loop.` and
  `All video writers closed`. Expected outputs: `000000.mp4`, `000001.mp4` in
  each save_rendering_dir (25 fps: 50 Hz step / render_frame_skip=2).
- The `metrics_file` drives text overlay (motion key, mpjpe, success) and
  sorts/filters which motions render (eval_agent_trl.py:265-369) — keep it.
- num_envs=2 covers both motions on box; VRAM is not the concern (crashes
  happened at app boot with 0 MiB used). If A/B OOM later in the loop, they
  will die visibly in render.log — do NOT raise num_envs above 8 on this A10G.
- `python -u` vs `python.sh`: eval_exp.py:389 uses `python -u` only to skip
  accelerate; inside this container plain `python` does not exist and
  `/isaac-sim/python.sh` is the same interpreter with IsaacLab paths — the
  crash is unrelated to this choice.

## C — fallback if RTX still crashes: CPU-safe trajectory dump + offline render

Idea: keep `render_results=False` (NO cameras → no RTX → uses the proven
`isaaclab.python.headless.kit` path) and attach only the
TrajectoryRecorder, which records joint positions + root pose per frame to
.pkl (gear_sonic/envs/manager_env/mdp/recorders.py:193-386). Then render the
.pkl offline with MuJoCo (CPU) from the G1 MJCF.

C1 — baseline trajectory dump:

```bash
docker exec isaac-lab-base bash -c "cd /workspace/GR00T-WholeBodyControl && nohup /isaac-sim/python.sh gear_sonic/eval_agent_trl.py +checkpoint=/workspace/wbc-training-logs/baseline/wbc_baseline_10k-20260701_232851/model_step_010000.pt +headless=True ++eval_callbacks=im_eval ++run_eval_loop=False ++num_envs=2 ++metrics_file=/workspace/wbc-training-logs/eval_baseline/step_010000/metrics_eval.json ++manager_env.config.save_rendering_dir=/workspace/wbc-training-logs/diagnosis/traj_baseline +manager_env.recorders.trajectory._target_=gear_sonic.envs.manager_env.mdp.recorders.TrajectoryRecorderCfg +manager_env.recorders.trajectory.save_path=/workspace/wbc-training-logs/diagnosis/traj_baseline ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False > /workspace/wbc-training-logs/diagnosis/traj_baseline/traj.log 2>&1 &"
```

C2 — release trajectory dump: same, with
`+checkpoint=/workspace/GR00T-WholeBodyControl/sonic_release/last.pt`,
`++metrics_file=/workspace/wbc-training-logs/diagnosis/eval_release/metrics_eval.json`,
and dirs `.../traj_release`.

Caveats for C (uncertainties, since this combination is unexercised):
- `metrics_file` sets `max_render_envs`, which flips the im_eval callback into
  render_only mode and calls `end_render_results()` at loop end — that closes
  trajectory writers too (manager_env_wrapper.py:988-992), so the .pkl files
  get flushed. If Hydra rejects the struct-add of `manager_env.recorders.trajectory`,
  wrap both overrides once more with `++`.
- Offline render (CPU, after docker cp of the .pkl): load
  `dof_pos`/`root_pos_w`/`root_quat_w` (wxyz) at fps from the pkl into a
  MuJoCo scene built from
  `gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml`,
  set qpos per frame, render with `mujoco.Renderer` offscreen (EGL not needed:
  use `MUJOCO_GL=osmesa` for pure CPU) and write mp4 with imageio. ~30 lines;
  can be run on the host or container without touching the GPU.

## Copy videos out (after A/B or the offline render)

```bash
mkdir -p /home/ec2-user/work/groot-tao-agentic-rl-curriculum/experiments/baseline-eval-diagnosis/videos
docker cp isaac-lab-base:/workspace/wbc-training-logs/diagnosis/render_baseline/. /home/ec2-user/work/groot-tao-agentic-rl-curriculum/experiments/baseline-eval-diagnosis/videos/baseline_step_010000/
docker cp isaac-lab-base:/workspace/wbc-training-logs/diagnosis/render_release/.  /home/ec2-user/work/groot-tao-agentic-rl-curriculum/experiments/baseline-eval-diagnosis/videos/sonic_release/
```

Then delete the copied `render.log` files from videos/ if only mp4s are wanted.

## Bonus (cheap, 2 min GPU each): identify the eval killer termination

Re-run the verified metrics eval on model_step_010000 three times, each time
raising ONE eval threshold to effectively disable it, and see which change
moves progress_rate:
`++manager_env.terminations.ee_body_pos.params.threshold=10.0`, then
`++manager_env.terminations.anchor_ori_full.params.threshold=100.0`, then
`++manager_env.terminations.anchor_pos.params.threshold=10.0`
(output dirs `/workspace/wbc-training-logs/diagnosis/ablate_{ee,ori,pos}`).
This resolves the main open question in RESULTS.md §3.
