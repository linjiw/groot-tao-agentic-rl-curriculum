---
name: sonic-run-digest
description: >-
  Digest builder for the SONIC Curriculum-Manager Agent (design doc 08).
  Parses a run's training-metrics, eval, and adaptive-sampler JSONL streams
  into a single trend-annotated digest.json — success-rate trends (train-side
  AND the protected held-out metric), failed_keys diff/persistence,
  failure-rate-vector entropy + cap-saturation, KL/entropy/loss trends,
  per-term episode rewards, current knob state, and recent decision outcomes.
  Use when assembling the manager's per-tick observation, analyzing a SONIC
  run's curriculum health, or generating test fixtures for the replay harness.
license: Apache-2.0
compatibility: >-
  Pure Python 3.9+ stdlib (no numpy/torch/wandb). Consumes JSONL exported
  from the run (wandb export, stdout parse, or the replay harness's
  synthetic generators). Metric names match pinned WBC 0e35637.
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
- observability
---

# SONIC Run Digest (Curriculum-Manager observation)

Phase-0 component of the Curriculum-Manager Agent
(`docs/design/08-curriculum-manager-agent.md` §4 observation space, §3 step-1
"Digest"). Turns raw run logs into the **only** thing the manager LLM reads:
a compact, trend-annotated `digest.json`. Nothing per-step reaches the LLM.

## Files

- `digest_builder.py` — `build_digest(...)` + CLI. Stdlib-only stats
  (least-squares trend labels, normalized entropy, cap-saturation computed
  with the sampler's own semantics — cap = mean × `max_over_mean`, mirroring
  `motion_lib_base.py:2570–2577`).
- `test_digest_builder.py` — 13 tests (pytest).

## Input streams (JSONL, one record per line, SONIC's own metric names)

| Stream | Cadence | Keys (verified source) |
|---|---|---|
| `train.jsonl` | per logged PPO iteration | `it`, `policy/approxkl_avg`, `loss/entropy_avg`, `loss/value_avg`, `loss/policy_avg`, `lr`, `Policy/mean_noise_std`, `fps`, `Episode/<term>`, `scheduled_params/<name>` (`ppo_trainer.py:1578–1631`, `1874–1905`) |
| `eval.jsonl` | per eval pass | `it`, `success_rate`, `progress_rate`, `failed_keys`, optional `mpjpe_all_mean`, optional `eval/success/*` (`im_eval_callback.py:747`, `811–815`); **plus `heldout_success_rate`** — the protected metric from the separate held-out watcher (not stock SONIC; doc 08 axiom 5) |
| `sampler.jsonl` | per 200-step sampling sync | `it`, `failure_rate` (per-bin vector = `adp_samp_failure_rate`, `motion_lib_base.py:2531–2552`) |

Producing these streams from a live run is the job of the (Phase-2)
`sonic-job-adapter`; in Phase 0/1 the replay harness generates them
synthetically, and a wandb CSV/JSON export can be reshaped with a few lines.

## Digest contents (what the manager sees)

- `eval.success_rate` / `eval.heldout_success_rate` — `{last, mean_recent,
  slope_recent, trend, n_points}` over the last `window` evals. Both are
  present so the playbook can enforce **never act on eval-side evidence
  alone** (doc 08 §9).
- `eval.failed_keys` — `count`, `newly_failing`, `newly_recovered`,
  `persistent` (failing in every eval in the window).
- `sampler` — `normalized_entropy` trend, `cap_saturation_fraction`
  (fraction of bins clipped at the current cap — pass the **current**
  `adp_samp_failure_rate_max_over_mean`), `top_k_share`.
- `train` — trends for KL/entropy/losses/LR/noise-std, latest per-term
  `Episode/*` reward means, latest `scheduled_params/*`.
- `knobs` — current value + `ticks_since_change` per registry knob
  (supplied by the manager loop from `sonic-knob-registry` state).
- `decision_history` — recent journal entries with outcomes, so the LLM
  sees what its last interventions did (doc 08 §6.5, AURA-style memory).

## Quick start

```bash
cd skills/agentic/sonic-run-digest
python3 -m pytest test_digest_builder.py -q     # 13 passed

# From exported logs:
python3 digest_builder.py \
  --train train.jsonl --eval eval.jsonl --sampler sampler.jsonl \
  --max-over-mean 200 --window 5 --out digest.json
```

```python
from digest_builder import build_digest
digest = build_digest(train_records=..., eval_records=..., sampler_records=...,
                      knob_state=..., decision_history=..., max_over_mean=200.0)
```

## Caveats (honest)

- `heldout_success_rate` does **not** exist in stock SONIC — it is this
  project's protected metric, produced by a separate eval watcher over a
  frozen motion subset at fixed relaxed thresholds. Until that watcher
  exists, the field is absent and the digest reports `null` (the playbook
  must then refuse Family-B tightening decisions).
- Stock `success_rate` is computed at the **relaxed eval thresholds**
  (`terminations/tracking/eval.yaml`, 0.25), not the strict training ones —
  the decoupling the review flagged (`07:112`). The digest keeps both sides
  visible; it does not paper over the gap.
- Trend labels use a relative-slope tolerance (2%); with < 2 points in the
  window the trend is `"unknown"` — the playbook treats `unknown` as
  "do nothing".

## Related skills

- `sonic-knob-registry` — the action space + validator this digest's
  `knobs` section reflects.
- `sonic-curriculum-manager` (planned) — the playbook that reads this
  digest and emits decisions.
