---
name: sonic-knob-registry
description: >-
  Typed knob registry + static decision validator for the SONIC
  Curriculum-Manager Agent (design doc 08). Defines the manager's entire
  action space — data-curriculum knobs (uniform_sampling_rate, failure-rate
  cap, bin_size), competence-gated schedule knobs (termination thresholds,
  DR push scale, penalty ramps), optimizer meta-knobs (desired_kl,
  entropy_coef, KL-LR clamps) — each with hard bounds, max per-decision step,
  and cooldown. Validates a manager decision BEFORE anything is applied.
  Use when validating or authoring curriculum-manager decisions, defining
  new knobs, or asking what the manager is allowed to touch.
license: Apache-2.0
compatibility: >-
  Pure Python 3.9+ + PyYAML. No torch/IsaacLab/GPU. Knobs with status
  "patch" additionally require the stage-2 termination-curriculum patch on
  the WBC submodule; status "design" knobs are validator-rejected outside
  replay mode.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- tao
- sonic
- curriculum
- rl
- agentic
- guardrails
---

# SONIC Knob Registry (Curriculum-Manager action space)

Phase-0 component of the Curriculum-Manager Agent
(`docs/design/08-curriculum-manager-agent.md`, §5 action space + §3 step-3
"Validate"). The registry is the **whitelist**: anything not in
`registry.yaml` is outside the manager's action space by construction.

## Files

- `registry.yaml` — the knob registry as data. Three disjoint families
  (review caveat `07:114` — data curriculum ≠ hyperparameter schedule):
  `data_curriculum`, `schedule`, `optimizer`. Every knob carries a
  `verified_source` citation into the pinned WBC submodule (`0e35637`) or a
  design doc, a `status` (`available` / `patch` / `design`), `hard_range`,
  `max_step` (multiplicative / additive / notch), and `cooldown_ticks`.
- `knob_registry.py` — loader + `validate_decision(decision, state)`:
  whitelist, typed value, hard range, max step vs current value, per-knob
  cooldown, one-atomic-change, required `rationale`/`expected_effect`/
  `tripwire` fields. Returns structured errors/warnings; never mutates state.
- `test_knob_registry.py` — 22 tests (pytest).

## Decision format (what the manager LLM must emit)

```yaml
action: set            # or "none" — the default for a healthy run
knob: uniform_sampling_rate
value: 0.15
rationale: "failure-rate entropy dropped 40% over 3 ticks; cap saturated"
expected_effect: "sampler concentration falls; held-out success flat or up"
tripwire: {metric: heldout_success_rate, drop_pct: 5, evals: 3}
```

## Quick start

```bash
cd skills/agentic/sonic-knob-registry
python3 -m pytest test_knob_registry.py -q     # 22 passed

python3 - <<'PY'
from knob_registry import load_registry, RunState
reg = load_registry()
state = RunState(tick=10)
d = {"action": "set", "knob": "uniform_sampling_rate", "value": 0.15,
     "rationale": "cap saturated", "expected_effect": "entropy up",
     "tripwire": {"metric": "heldout_success_rate", "drop_pct": 5, "evals": 3}}
res = reg.validate_decision(d, state)
print(res.ok, res.errors, res.warnings)
PY
```

## Rules encoded (from design doc 08 axioms 5–6)

1. **One atomic change per tick** (`meta.global_rules.max_changes_per_tick: 1`);
   list-shaped knob payloads are rejected outright.
2. **`action: none` is always valid** and is the documented default.
3. **Bounded steps**: PBT-style multiplicative bounds (×1.5/×2), additive
   notches for thresholds (0.02–0.05 m per design), adjacent-notch-only for
   choices.
4. **Cooldowns** are per-knob, in manager ticks, checked against the
   caller-owned `RunState`.
5. **`status: design` knobs** (Stage-3 DR ramp, Stage-4 retire/replay,
   penalty ramps) are rejected unless `allow_design=True` — pass that only
   in the replay harness, never against a live run.
6. **Explicitly absent** (outside the action space, do not add without a
   design-doc revision): per-clip sampling weights, eval thresholds
   (`terminations/tracking/eval.yaml` — the relaxed 0.25 set), held-out set
   composition, reward function code, anything in the eval watcher.

## Verified anchors (pinned WBC `0e35637`)

- Sampler knobs live-read at the 200-step sync:
  `gear_sonic/config/manager_env/commands/terms/motion.yaml:16–25`
  (`bin_size: 50`, `uniform_sampling_rate: 0.1`,
  `adp_samp_failure_rate_max_over_mean: 50.0`; `sonic_release.yaml:71`
  overrides the cap to `200`), consumed in
  `gear_sonic/utils/motion_lib/motion_lib_base.py:2531–2577` (cap =
  mean × `adp_samp_failure_rate_max_over_mean`, clip, renorm).
- Strict training thresholds:
  `terminations/tracking/base_adaptive_strict_ori_foot_xyz.yaml`
  (`anchor_pos 0.15`, `ee_body_pos 0.15`, `foot_pos_xyz 0.2`); relaxed eval
  set: `terminations/tracking/eval.yaml` (0.25) — **eval thresholds are not
  knobs** (protected-metric rule).
- Optimizer defaults: `gear_sonic/config/algo/ppo_im_phc.yaml:16–22`
  (`entropy_coef 0.01`, `desired_kl 0.01`).
- `schedule_dict` mechanism: `gear_sonic/trl/utils/scheduler.py:296–353`,
  consumed at `ppo_trainer.py` step loop; CPU port already validated at
  `experiments/reverse-transfer-lr-curriculum/curriculum_schedule.py`.

## Related skills

- `sonic-run-digest` — builds the observation the manager reads before
  deciding.
- `sonic-curriculum-manager` (planned) — the playbook that consumes both.
- `tao-curriculum-rl` — the reverse-transfer skill; its ported
  `kl_adaptive_lr.py` is the inner controller Family-C knobs parameterize.
