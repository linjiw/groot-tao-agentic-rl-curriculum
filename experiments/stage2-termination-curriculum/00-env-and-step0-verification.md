# Stage-2 Experiment — Environment Recon & Step-0 Verification

Recorded 2026-06-30 on the dev box. This logs (a) what hardware/software is actually
available here, and (b) the result of the load-bearing Step-0 check from
[`docs/design/05-flagship-experiment.md`](../../docs/design/05-flagship-experiment.md).

## A. Hardware / software reality on this box

| Resource | Found | Implication |
|---|---|---|
| GPU | **1× NVIDIA A10G, 23 GB** (driver 595.71, CUDA 13.2) | The full-scale flagship (`num_envs=4096`, 64+ GPUs) **cannot run here**. Only a heavily reduced-scale smoke run is conceivable, and only after Isaac Lab is installed. |
| CPU / RAM | 8 vCPU / 30 GB | Fine for code work and CPU-only checks; marginal for sim. |
| Disk | `/workspace` 294 GB free; `/` 36 GB free | Enough for the repos; Isaac Sim install is large. |
| `.venv_sim` | Python 3.10, **torch 2.12.1+cu130 (CUDA OK)**, imports `gear_sonic` | This is the **MuJoCo sim** venv. |
| `isaaclab` / `isaacsim` | **NOT installed** in any venv on this box | The **training** path (`train_agent_trl.py`) hard-fails without Isaac Lab. Training is out of scope on this box until Isaac Lab is installed on a multi-GPU host. |

**Conclusion:** on this box we can do design-faithful, **zero-GPU** progress — verify the
mechanism the plan hinges on, and implement + statically validate the Stage-2 code so it is
ready to launch the moment a proper training host (Isaac Lab + multi-GPU) is available. We do
**not** fabricate training results here.

## B. Step-0 verification — does Stage 2 actually work?

**Question (from the flagship doc):** does `terminations.py` read `params.threshold` from the
*live* term cfg on each call, so that an IsaacLab `modify_term_cfg` curriculum write propagates
at episode boundaries? If thresholds were cached at setup, Stage 2 would be inert.

**Answer: YES — verified in source. Stage 2 is viable.**

Evidence (`external/IsaacLab/source/isaaclab/isaaclab/managers/termination_manager.py`):
```python
# TerminationManager.compute()  (line ~167)
for i, term_cfg in enumerate(self._term_cfgs):
    value = term_cfg.func(self._env, **term_cfg.params)   # <-- params read FRESH every step
```
- `compute()` re-reads `term_cfg.params` from `self._term_cfgs` on **every** step.
- `set_term_cfg(name, cfg)` (line ~229) and `get_term_cfg(name)` (line ~246) exist, so a
  curriculum term can mutate the live cfg.
- The `gear_sonic` termination funcs take `threshold` as a plain kwarg
  (`exceeded_anchor_pos(env, command_name, threshold)`, `terminations.py:54`; `exceeded_anchor_height(... threshold, threshold_adaptive, down_threshold ...)`, `:97`), i.e. the value is consumed live from `params`, not snapshotted.

➡️ Mutating `terminations.<term>.params.threshold` via a curriculum term takes effect. ✓

## C. Corrections to the design doc (found during verification)

The original `docs/design/05-flagship-experiment.md` had two inaccuracies, now corrected here
and to be reconciled into the design doc:

1. **`modify_term_cfg` is an IsaacLab primitive, not a `gear_sonic` function.** It lives at
   `isaaclab.envs.mdp.curriculums:modify_term_cfg` and is re-exported into the `gear_sonic` MDP
   namespace via `from isaaclab.envs.mdp import *` (`gear_sonic/envs/manager_env/mdp/__init__.py:3`).
   So both `mdp:modify_term_cfg` and `mdp:step_curriculum` resolve from the same `mdp` module.
2. **Address shorthand:** `modify_term_cfg` rewrites the **first** `s.` in the address to
   `_manager.cfg.`. So `"terminations.anchor_pos.params.threshold"` →
   `"termination_manager.cfg.anchor_pos.params.threshold"`. (Confirmed in the
   `modify_term_cfg` docstring, `curriculums.py`.)
3. **`modify_fn` signature** is `(env, env_ids, old_value, **modify_params)` and should return
   `mdp.modify_env_param.NO_CHANGE` when it does not change the value (to skip a redundant
   setter call). `gear_sonic`'s `step_curriculum(env, env_ids, original_value, values, num_steps)`
   matches the positional signature but returns `original_value` rather than `NO_CHANGE` — so it
   will call the setter every step. That is functionally correct (writes the same value back) but
   incurs a tiny per-step overhead. The patch ships a `step_curriculum_nochange` variant that
   returns `NO_CHANGE` between milestones to avoid the redundant write.

## D. What ships in this experiment folder

- [`patch/stage2_termination_curriculum.patch`](patch/stage2_termination_curriculum.patch) — a
  unified diff against the WBC submodule adding the `CurriculumCfg` fields, the `modify_fn`
  injection, and the `NO_CHANGE` helper.
- [`config/threshold_tighten.yaml`](config/threshold_tighten.yaml) — the new curriculum config
  (drop into `gear_sonic/config/manager_env/curriculum/`).
- [`apply_and_validate.sh`](apply_and_validate.sh) — applies the patch + config into the
  submodule working tree and runs a CPU-only static import/parse validation (no GPU, no Isaac Sim).
- [`RUN.md`](RUN.md) — the exact commands to launch on a proper Isaac-Lab multi-GPU host.

> The patch and config are deliberately kept **in our repo** rather than committed into the
> submodule, so the submodule stays a clean by-reference pin and the changes remain reviewable.
