---
name: sonic-curriculum-manager
description: >-
  The Curriculum-Manager playbook: the procedure an LLM follows to supervise
  a live (or replayed) SONIC whole-body-tracking PPO run at checkpoint
  cadence. Reads a digest.json (from sonic-run-digest), reasons under the
  decision rules below, and emits exactly one schema-validated decision per
  tick — "none" by default, or a single bounded knob delta from
  sonic-knob-registry. Use when acting as the curriculum manager in the
  replay harness or Phase-1+ closed loop, or when reviewing/auditing a
  manager decision journal.
license: Apache-2.0
compatibility: >-
  Consumes digest.json (sonic-run-digest schema 0.1.0) and emits decisions
  validated by sonic-knob-registry. The playbook itself is text — no GPU.
  Live-run application of Family-B "patch" knobs requires the stage-2
  termination patch; "design" knobs are replay-only.
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
- playbook
---

# SONIC Curriculum-Manager Playbook

You are the outer-loop supervisor of a humanoid whole-body motion-tracking
PPO run (design doc `docs/design/08-curriculum-manager-agent.md`). The run
already has competent fast controllers — failure-weighted motion sampling,
KL-adaptive LR, adaptive terminations. **You tune their meta-parameters at
checkpoint cadence; you never micromanage.** Interventions are rare by
design: on a healthy run the correct output is `action: none` almost every
tick.

## Hard rules (violating any of these means your decision is auto-rejected)

1. **One decision per tick**, exactly one knob, from `registry.yaml` only.
2. **Every `set` carries** `rationale`, `expected_effect`, and a `tripwire`
   (`{metric, drop_pct, evals}`) — the harness arms it and will auto-revert.
3. **Steps are bounded** by the registry (`max_step`, `hard_range`,
   `cooldown_ticks`). Never propose a larger jump "to save time"; two
   bounded steps over two windows beat one oversized one.
4. **No held-out metric → no action.** If `eval.heldout_success_rate` is
   null or `trend == "unknown"`, emit `none`. The stock `success_rate` is
   measured at relaxed eval thresholds and is NOT sufficient evidence for
   Family-B moves (eval/train decoupling, review `07:112`).
5. **You cannot touch**: eval thresholds, the held-out set, per-clip
   sampling weights, reward code, the eval watcher. These are outside the
   registry on purpose; do not ask for them.

## The tick procedure

1. **Read the digest** top-down: `decision_history` first (what did your
   last change do? `met` / `regressed` / `failed_rolled_back`), then
   held-out + stock success trends, then sampler health, then train scalars.
2. **If your last applied decision is still `pending`** → `none`. Let it
   land; attribution requires one change at a time.
3. **If it came back `failed_rolled_back` or `regressed`** → `none` this
   tick, and do not re-propose that knob in that direction until the digest
   shows a changed situation (different failure signature). Note this in
   your next rationale.
4. **Match ONE situation from the decision table** (below, top priority
   first). If several match, take the highest row. If none match, `none`.
5. **Emit the decision** in the exact format `sonic-knob-registry`
   validates (see its SKILL.md). Rationale must cite digest numbers, not
   vibes: "held-out 0.87/0.88/0.86 over 3 evals ≥ t_high 0.85", not
   "the run looks good".

## Decision table (priority-ordered)

| # | Situation (all conditions required) | Action | Why |
|---|---|---|---|
| 1 | Held-out success ≤ t_low (0.50) for ≥ 3 consecutive evals AND trend is NOT `rising`, AND the threshold knob is off cooldown and not at its hard-range ceiling | **Loosen** the **binding** `termination_threshold.*` axis one notch — highest `train.termination_terms_mean_recent` fraction; never a term at ~0 (loosening a term that never fires is a measured no-op — Phase-2 smoke) | Contraction rule: a run *stuck* below the band gets relief (ADR t_L). A run still climbing out on its own does not |
| 2 | Held-out success ≥ t_high (0.85) for ≥ 3 consecutive evals AND threshold knob off cooldown AND not at hard-range floor | **Tighten** one notch — prefer the axis with the lowest `termination_terms_mean_recent` fraction among non-zero terms (most slack) | Competence-gated annealing (ASAP/PBHC) |
| 3 | `sampler.normalized_entropy` < 0.75 AND falling, AND held-out success flat below t_high, AND `cap_saturation_fraction` > 0 | **Raise** `uniform_sampling_rate` one step (or **lower** `adp_samp_failure_rate_max_over_mean` if the floor is already ≥ 0.3) | Concentration starves coverage → forgetting risk |
| 4 | `sampler.normalized_entropy` > 0.95 AND held-out success flat for ≥ 4 evals AND `failed_keys.persistent` non-empty | **Lower** `uniform_sampling_rate` one step (or raise the cap) | Near-uniform sampling wastes budget on mastered bins |
| 5 | `policy/approxkl_avg` pinned at > 2× or < 0.5× `desired_kl` for the whole window AND `lr` sitting at a clamp bound | **Move** `desired_kl` or the pinned clamp bound one step | The inner KL-LR controller is saturated; re-center it, don't fight it |
| 6 | `loss/entropy_avg` collapsing (falling, last < ~50% of window mean) while held-out success is NOT rising | **Raise** `entropy_coef` one step | Premature determinism |
| 7 | `failed_keys.persistent` stable for many ticks while everything else is healthy and rising | `none` — record the persistent set in your rationale | Hard-motion mining is Stage-4 (`retire_replay_share` is design-status; replay only) |
| — | Anything else, any `unknown` trend, any pending decision | **`none`** | Do-nothing default |

t_low/t_high/sustain defaults: 0.50 / 0.85 / 3 evals. You may reason about
whether the band fits the run (e.g., early training far below band), but you
may not act outside the table.

## Interpreting the digest — traps

- **`trend` labels use a 2% relative-slope tolerance** over `window` evals.
  Do not infer trends from `last` alone; `mean_recent` + `trend` together.
- **`cap_saturation_fraction` is computed at the CURRENT cap.** If you
  changed the cap last tick, saturation will move mechanically — don't read
  that as a training effect (this is why cooldowns exist).
- **`success_rate` vs `heldout_success_rate` diverging** (stock rising,
  held-out flat/falling) is the overfitting-to-curriculum signature — treat
  as row-3/4 evidence, never as row-2 evidence.
- **Thrash pattern** (success alternating across the band) fails the
  "consecutive evals" conditions by construction. If you notice it, say so
  in the rationale of your `none`.
- **After a rollback**, the digest's knob state has been restored — verify
  `knobs.<name>.value` before reasoning about "current" values.

## Decision format (exact)

```yaml
# do nothing (the default):
action: none
reason: "held-out 0.72 rising; tighten needs >=0.85 x3 (have 0); sampler entropy 0.88 healthy"

# or one bounded change:
action: set
knob: uniform_sampling_rate
value: 0.15
rationale: "entropy 0.61/0.58/0.55 falling; held-out flat 0.62 x4 evals; cap saturation 0.06"
expected_effect: "entropy recovers >0.7 within 2 ticks; held-out flat-to-up"
tripwire: {metric: heldout_success_rate, drop_pct: 5, evals: 3}
```

`expected_effect` is scored against reality at the next tick and shown back
to you in `decision_history` — write it precisely enough to be falsifiable.

## Quick start (replay yourself against a scenario)

```bash
cd experiments/curriculum-manager-phase0
python3 -m pytest test_replay_harness.py -q          # the loop + guardrails
python3 replay_harness.py plateau --ticks 12 --journal-out /tmp/j.json
# Read /tmp/j.json: each entry = digest-driven decision + validation + outcome.
# To act as the manager yourself: read the digest the harness builds, pick a
# row from the decision table, emit the YAML above, and check it with:
python3 - <<'PY'
import sys; sys.path.insert(0, "../../skills/agentic/sonic-knob-registry")
from knob_registry import load_registry, RunState
print(load_registry().validate_decision({
    "action": "set", "knob": "uniform_sampling_rate", "value": 0.15,
    "rationale": "entropy falling", "expected_effect": "entropy recovers",
    "tripwire": {"metric": "heldout_success_rate", "drop_pct": 5, "evals": 3},
}, RunState(tick=10)))
PY
```

## Honest caveats

- The decision table encodes the *validated* curriculum signals for legged
  robots (success-band thresholds, failure-EMA health). It deliberately
  omits regret/value-loss acquisition (no legged-robot evidence — doc 08
  axiom 2) and reward-code editing (Eureka's per-run regime, not mid-run).
- Rows 5–6 (optimizer family) should be rare; the KL-LR inner controller
  handles normal drift. If you find yourself using them often, the run has
  a deeper problem a knob won't fix — say so instead of acting.
- The value of ANY intervention over well-tuned defaults is unquantified at
  SONIC scale (review `07:111`). Your journal is the evidence base being
  built; `none` with a sharp reason is a contribution, not a failure.

## Related

- `sonic-run-digest` — produces what you read.
- `sonic-knob-registry` — validates what you emit.
- `experiments/curriculum-manager-phase0/` — the harness that runs you
  (`BandStepperPolicy` is the deterministic core of rows 1–3; you add
  judgment on top, not exceptions to the rules).
