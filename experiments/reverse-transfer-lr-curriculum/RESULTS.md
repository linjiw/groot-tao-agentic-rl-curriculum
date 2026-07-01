# Reverse-transfer: SONIC LR controller + curriculum scheduler → TAO

Ports two RL utilities OUT of the pinned `gear_sonic` submodule (GR00T
whole-body-control) INTO standalone, dependency-light, unit-tested,
domain-agnostic Python for TAO agentic post-training adoption.

All claims labeled: **[verified]** = read in pinned source this task ·
**[measured]** = from running code · **[speculative]** = inference.

---

## 1. KL-adaptive learning-rate controller — `kl_adaptive_lr.py`

**[verified] Source:**
`external/GR00T-WholeBodyControl/gear_sonic/trl/trainer/ppo_trainer.py`,
lines **2142-2166**, method
`_adjust_learning_rate_based_on_kl(self, kl_mean, optimizer)`.

**[verified] Exact SONIC logic (copied from source this task):**

```python
# ppo_trainer.py:2154-2166
if self.desired_kl is None:
    return
if kl_mean > self.desired_kl * 2.0:
    new_lr = max(self.adaptive_lr_min, self.args.learning_rate / 1.5)
elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
    new_lr = min(self.adaptive_lr_max, self.args.learning_rate * 1.5)
else:
    new_lr = self.args.learning_rate
self.args.learning_rate = new_lr
for param_group in optimizer.param_groups:
    param_group["lr"] = self.args.learning_rate
```

**[verified] The early-return guard** (the one flagged in the task) is
`if self.desired_kl is None: return` at ppo_trainer.py:2154-2155 — captured
faithfully as the "inert when `desired_kl is None`" behavior.

**What was ported (faithful):**
- The three-way band rule: shrink `/factor` above `desired_kl*2`, grow
  `*factor` below `desired_kl/2` **and** `kl_mean > 0.0`, else hold. **[verified]**
- Clamping to `[lr_min, lr_max]` via `max(...)` / `min(...)`. **[verified]**
- The `> 0.0` guard on the grow branch (non-positive KL never grows lr). **[verified]**
- In-place mutation of `optimizer.param_groups[*]["lr"]` when an optimizer is
  passed; always returns the new lr. **[verified]**

**Generalizations (labeled [speculative] design choices, behavior identical to SONIC by default):**
- SONIC's `self.desired_kl`, `self.adaptive_lr_min/max`, `self.args.learning_rate`
  become plain constructor args (`desired_kl, lr_min, lr_max, lr`).
- SONIC hard-codes `1.5` as both shrink divisor and grow multiplier; exposed as
  `factor` (default **1.5**) so the out-of-box rule is byte-identical to SONIC.
- Pure-python core — **no torch required** for the LR math. Works with any
  object exposing `.param_groups` (a real `torch.optim.Optimizer` works
  unchanged), or with `optimizer=None`.

**API:**
```python
KLAdaptiveLR(desired_kl, lr_min, lr_max, lr, factor=1.5)
    .update(kl_mean, optimizer=None) -> new_lr
```

---

## 2. Curriculum scheduler (`schedule_dict`) — `curriculum_schedule.py`

**FOUND** — a real implementation exists (not invented).

**[verified] Source:**
`external/GR00T-WholeBodyControl/gear_sonic/trl/utils/scheduler.py`,
function `update_scheduled_params(obj, scheduler_dict, step, split_char="@")`
lines **296-353**, plus the `@`-path navigation helpers lines **17-293**.

**[verified] Usage confirming it is the real `schedule_dict` format:**
`gear_sonic/eval_agent_trl.py:466-470` calls
`scheduler.update_scheduled_params(schedule_wrapper, config.trainer.schedule_dict, state.global_step)`,
and `ppo_trainer.py:361,389,413` threads `schedule_dict` through the trainer.
This matches design-doc `docs/design/07-review-and-revised-roadmap.md` line 89's
description ("@-path, linear/segment curriculum-schedule serialization").

**What was ported (faithful):**
- **`interpolate_schedule`** — the `linear` (breakpoint linear interp, clamp/hold
  at final segment) and `segment` (piecewise-constant hold) logic from
  scheduler.py:306-320. **[verified]** Pure-python, testable with no host object.
- **`@`-path navigation** — attribute access, `['key']`/`[idx]` bracket access,
  and `method('arg')` call resolution (scheduler.py:17-293). **[verified]**
- **`update_scheduled_params`** driver — target resolution, `val_type` coercion,
  dict merge vs. `overwrite_dict`, and `trigger_func` firing on exact breakpoints
  (scheduler.py:296-353). **[verified]**

**Generalizations (labeled — dependency-light + safety):**
- SONIC coerces via `eval(val_type)(val)`; replaced with a **safe type map**
  (`float/int/bool/str`). Identical behavior for documented types. **[speculative]**
- SONIC's `DictConfig` (omegaconf) branch collapses to plain-`dict` handling;
  omegaconf is **not** a dependency here. dict-merge / `overwrite_dict`
  semantics preserved for plain dicts. **[speculative]**
- SONIC's bare-`eval` fallback for exotic bracket keys / args is dropped in favor
  of literal parsing (safety). Standard string/int/bool/None args unaffected. **[speculative]**
- Below-first-breakpoint index clamped to 0 to avoid negative indexing (SONIC's
  `while step < seg_steps[i]: i -= 1` assumes `step >= seg_steps[0]`). Behavior
  for `step >= seg_steps[0]` is identical. **[speculative]**

---

## 3. Tests — `test_kl_adaptive_lr.py`

19 tests. KL controller coverage (required cases): KL above 2× band shrinks by
`/1.5` incl. `lr_min` clamp; KL below 0.5× band grows by `*1.5` incl. `lr_max`
clamp; in-band hold; strict-boundary hold; the `>0.0` guard (zero **and**
negative KL); `desired_kl=None` inert; optimizer `param_groups` mutation;
`optimizer=None` path; a 5-step trajectory (shrink→shrink→hold→grow→guard-hold);
custom `factor`. Plus curriculum interp/`@`-path smoke tests.

### [measured] Real pytest output

Command:
```
/workspace/Isaac-GR00T/.venv/bin/python -m pytest test_kl_adaptive_lr.py -v
```

```
============================= test session starts ==============================
platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0 -- /workspace/Isaac-GR00T/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/ec2-user/work/groot-tao-agentic-rl-curriculum/experiments/reverse-transfer-lr-curriculum
plugins: typeguard-4.5.1
collected 19 items

test_kl_adaptive_lr.py::test_kl_above_2x_band_shrinks_by_factor PASSED   [  5%]
test_kl_adaptive_lr.py::test_shrink_clamped_at_lr_min PASSED             [ 10%]
test_kl_adaptive_lr.py::test_kl_below_half_band_grows_by_factor PASSED   [ 15%]
test_kl_adaptive_lr.py::test_grow_clamped_at_lr_max PASSED               [ 21%]
test_kl_adaptive_lr.py::test_kl_in_band_holds PASSED                     [ 26%]
test_kl_adaptive_lr.py::test_grow_branch_zero_kl_guard_holds PASSED      [ 31%]
test_kl_adaptive_lr.py::test_grow_branch_negative_kl_guard_holds PASSED  [ 36%]
test_kl_adaptive_lr.py::test_band_boundaries_are_exclusive_hold PASSED   [ 42%]
test_kl_adaptive_lr.py::test_desired_kl_none_is_inert PASSED             [ 47%]
test_kl_adaptive_lr.py::test_optimizer_param_groups_updated PASSED       [ 52%]
test_kl_adaptive_lr.py::test_optimizer_none_still_returns_lr PASSED      [ 57%]
test_kl_adaptive_lr.py::test_multi_step_trajectory PASSED                [ 63%]
test_kl_adaptive_lr.py::test_custom_factor PASSED                        [ 68%]
test_kl_adaptive_lr.py::test_linear_interpolation_midpoint PASSED        [ 73%]
test_kl_adaptive_lr.py::test_linear_holds_at_and_after_final PASSED      [ 78%]
test_kl_adaptive_lr.py::test_linear_multi_segment PASSED                 [ 84%]
test_kl_adaptive_lr.py::test_segment_step_function PASSED                [ 89%]
test_kl_adaptive_lr.py::test_update_scheduled_params_simple_attr PASSED  [ 94%]
test_kl_adaptive_lr.py::test_update_scheduled_params_at_path_and_bracket PASSED [100%]

============================== 19 passed in 0.05s ==============================
```

**[measured] Summary line: `19 passed in 0.05s`**

> Note: `pytest` was not present in `/workspace/Isaac-GR00T/.venv` initially; it
> was installed into that venv with `uv pip install pytest` (pytest 9.1.1) to
> run the suite. The venv's interpreter is Python 3.10.20.

---

## 4. How TAO adopts this

**[speculative]** The KL-adaptive LR rule is **domain-agnostic**: it only needs a
scalar "distance between successive policies." Nothing about it is specific to
whole-body control or SONIC's PPO.

- **Drop-in for any optimizer loop.** After each policy-update round, compute a
  KL (or KL-proxy) between the pre- and post-update policy on a batch of
  prompts/states, then:
  ```python
  ctrl = KLAdaptiveLR(desired_kl=0.01, lr_min=1e-6, lr_max=1e-3, lr=optimizer.defaults["lr"])
  # ... each step:
  ctrl.update(kl_mean=measured_kl, optimizer=optimizer)   # mutates param_groups, returns lr
  ```
- **KL-proxy for agentic post-training.** For LLM/agent RL where an exact KL is
  costly, feed a proxy: mean token-level KL on a fixed eval batch, sequence-level
  logprob divergence between old/new policy, or an importance-ratio-derived
  estimate. The controller only cares that larger = policy moved more.
- **Stability guarantee carried over.** The `[lr_min, lr_max]` clamps and the
  `>0.0` grow guard prevent runaway lr and spurious growth from noisy/zero KL —
  the same safety SONIC relies on, now available to TAO independent of framework.
- **`curriculum_schedule.py`** lets TAO serialize curriculum knobs (difficulty,
  reward weights, DR ranges) as a `schedule_dict` of `@`-paths with
  `linear`/`segment` interpolation over training steps, and apply them to any
  config object each step — decoupled from Isaac/omegaconf.

---

## Files created (all under `experiments/reverse-transfer-lr-curriculum/`)
- `kl_adaptive_lr.py` — `KLAdaptiveLR` controller (SONIC port).
- `curriculum_schedule.py` — `schedule_dict` parse + interpolate port.
- `test_kl_adaptive_lr.py` — 19 pytest tests.
- `RESULTS.md` — this file.

No files under `external/` were modified (pinned, read-only). No commits made.
