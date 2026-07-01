# Infra Guide — running SONIC training on this box

How to use the verified training setup on this machine (A10G, Amazon Linux
2023). Everything below was run and verified on 2026-07-01 — see
[`progress/2026-07-01-infra.md`](progress/2026-07-01-infra.md) for the
verification record.

## The one-paragraph mental model

All training happens **inside the running `isaac-lab-base` docker container**
(host glibc is too old for IsaacLab pip wheels). The container has Isaac Sim
5.1 at `/isaac-sim`, IsaacLab at `/workspace/isaaclab`, and a WBC clone at
`/workspace/GR00T-WholeBodyControl` pinned to the **same commit as our
submodule (`0e35637`)** — our `file:line` citations are valid there. Always
use `/isaac-sim/python.sh` (the wrapper injects IsaacLab paths; the bare kit
python does not). Logs and checkpoints go under `/workspace/wbc-training-logs`.

## Quick reference

```bash
# is the container up?
docker ps --filter name=isaac-lab-base

# environment pre-flight (should be all green in Training section)
docker exec isaac-lab-base bash -c \
  "cd /workspace/GR00T-WholeBodyControl && /isaac-sim/python.sh check_environment.py --training"

# GPU from host
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Launch a training run

```bash
docker exec isaac-lab-base bash -c "cd /workspace/GR00T-WholeBodyControl && \
  nohup /isaac-sim/python.sh gear_sonic/train_agent_trl.py \
    +exp/manager/universal_token/all_modes=sonic_bones_seed \
    num_envs=256 headless=true use_wandb=false \
    project_name=demo experiment_name=myrun \
    base_dir=/workspace/wbc-training-logs \
    ++algo.config.num_learning_iterations=100 \
    ++algo.config.num_steps_per_env=16 \
    ++callbacks.model_save.save_last_frequency=10 \
    > /workspace/wbc-training-logs/myrun.log 2>&1 &"
```

- Output dir: `<base_dir>/<project_name>/<experiment_name>-<YYYYmmdd_HHMMSS>/`
  containing `config.yaml` (the fully-resolved config — check your overrides
  landed here), `meta.yaml`, and `last.pt` once the save cadence hits.
- Startup takes ~60–90 s (Isaac Sim boot) before iteration 1 prints.
- Measured: ~3.3 s/iter at 64 envs, ~3.7 s/iter at 256 envs (16 steps/env).
- 256 envs fits comfortably in the 23 GB A10G.

## Resume / rollback

```bash
# resume from an explicit checkpoint (= the manager's rollback primitive)
... train_agent_trl.py ... checkpoint=/workspace/wbc-training-logs/demo/myrun-20260701_212450/last.pt
```

Verified: prints `Loading checkpoint from ... / Loaded checkpoint from step N`
and continues. `last.pt` (~1.1 GB) contains model + optimizer + env state
(incl. the adaptive-sampler state). Omitting `checkpoint=` starts fresh unless
an `experiment_dir` glob matches (`train_agent_trl.py:73–91`).

## Knob overrides (the manager's action space, verified paths)

| Registry knob | Hydra override |
|---|---|
| `uniform_sampling_rate` | `++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.uniform_sampling_rate=0.25` ✅ verified live |
| cap β | `++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.adp_samp_failure_rate_max_over_mean=100` |
| `bin_size` | `++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.bin_size=50` |
| `termination_threshold.anchor_pos` | `++manager_env.terminations.anchor_pos.params.threshold=0.20` (config.yaml:763–771) |
| `termination_threshold.ee_body_pos` | `++manager_env.terminations.ee_body_pos.params.threshold=0.20` |
| `termination_threshold.foot_pos_xyz` | `++manager_env.terminations.foot_pos_xyz.params.threshold=0.25` |
| `desired_kl` / `entropy_coef` | `++algo.config.desired_kl=0.012` / `++algo.config.entropy_coef=0.01` |

Always confirm in the run's saved `config.yaml`. **Mutation model:** knobs
change **per run-segment** — stop, relaunch from `last.pt` with new
overrides. (Within-process mutation is future work.)

## Reading the console log (digest input)

Each iteration prints a boxed block:
`Learning iteration N` → `Mean rewards/length/entropy` →
`Env/Episode_Reward/<term>` → `Env/Metrics/motion/error_*` →
`Env/Episode_Termination/<term>` → `Env/adp_samp/*` (failure-rate min/max/mean,
prob stats, effective bins — the sampler-health stream) → `Total time/ETA`.
The `sonic-job-adapter` skill parses this into the digest JSONL streams —
prefer it over ad-hoc grep.

## Traps (each cost real time once)

1. `python` doesn't exist in the container; the bare kit python
   (`/isaac-sim/kit/python/bin/python3`) misses IsaacLab. **Use
   `/isaac-sim/python.sh`.**
2. Iteration count is `++algo.config.num_learning_iterations`, **not**
   `trainer.max_iterations` (the 06-30 failure).
3. `vector_quantize_pytorch` was pip-installed into the kit python this
   session; a container rebuild loses it.
4. The container is a running container, not an image — `docker restart`
   preserves it, but it is **not reproducible from a Dockerfile in this
   repo**. Treat `/workspace` as the durable state.
5. Motion data: only **2 motions** on box (`data/motion_lib_bones_seed/
   robot_filtered/`). bones-seed is HF-gated (access requested). Fine for
   smoke/adapter work; not for curriculum conclusions.
6. HF auth lives in `/workspace/hf-cache` (set `HF_HOME=/workspace/hf-cache`).
7. Checkpoints: `last.pt` overwrites in place; step-numbered files
   (`save_frequency`, default 2000) accumulate at ~1.1 GB each — watch
   `/workspace` (284 GB free as of setup).

## Cross-references

- Verification record: `progress/2026-07-01-infra.md`
- Manager design: `design/08-curriculum-manager-agent.md`
- Job adapter skill: `../skills/agentic/sonic-job-adapter/`
- Knob registry (bounds/cooldowns): `../skills/agentic/sonic-knob-registry/`
