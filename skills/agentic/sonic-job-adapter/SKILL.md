---
name: sonic-job-adapter
description: >-
  Launch/observe/rollback lifecycle for SONIC training runs on this box's
  isaac-lab-base container — the Curriculum-Manager's hands. Builds the
  verified docker-exec training invocation with manager knob overrides
  (registry names → verified Hydra paths), parses the training console log
  into digest train/sampler JSONL streams, and manages the per-segment
  checkpoint/snapshot/rollback bookkeeping. Use when launching managed
  SONIC run-segments, converting training logs to digest inputs, or
  rolling back a bad knob change.
license: Apache-2.0
compatibility: >-
  Command building, log parsing, and rollback bookkeeping are pure Python
  3.9+ stdlib (tested offline). launch/wait/parse-from-container require
  docker + the running isaac-lab-base container described in
  docs/infra-guide.md.
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
- infra
---

# SONIC Job Adapter (the manager's hands)

Phase-2 component of the Curriculum-Manager Agent
(`docs/design/08-curriculum-manager-agent.md` §7). Everything here wraps the
infra verified in `docs/infra-guide.md` — same container, same launch
invocation, same override paths.

## The segment model

The manager's unit of change is a **run-segment**: a knob-constant stretch
of training. Apply a decision = end segment N, snapshot its `last.pt`,
launch segment N+1 from that checkpoint with the new knob override.
Rollback = relaunch from segment N's **input** checkpoint with segment
N-1's knob values (`JobAdapter.rollback_launch`, tested). Within-process
knob mutation is explicitly out of scope.

```
seg1 (knobs A) ──last.pt──▶ snapshot_seg1.pt ──▶ seg2 (knobs B, ckpt=snapshot_seg1)
                                   │
                     tripwire fires ▼
                seg2_rollback (knobs A, ckpt=snapshot_seg1)   ← pre-change state
```

Snapshots exist because `last.pt` **overwrites in place** — without the
copy, segment N+1's progress destroys the rollback point.

## The eval pass (the manager's scoreboard)

`JobAdapter.eval_segment(seg, it)` runs an eval-only `im_eval` pass on a
segment's snapshot at the RELAXED FIXED thresholds
(`+manager_env/terminations=tracking/eval`: 0.25/0.25/1.0,
`threshold_adaptive=false`) and returns one digest eval-stream record.
Verified live 2026-07-02: ~1–2 min/checkpoint at 64 envs on the A10G, and
**deterministic** — the same checkpoint reproduces byte-identical
`metrics_eval.json`, so per-segment arm comparisons are noise-free.

Two properties this buys the manager (doc 08 §9 mitigation):
- **Training-mutable thresholds are neutralized at eval — but NOT by a
  free process boundary** (adversarial review 2026-07-02, finding M1):
  `eval_agent_trl.py:79–112` loads the CHECKPOINT-SIBLING `config.yaml`
  (which carries manager-applied overrides) and merges the eval config on
  top; `tracking/eval.yaml` only names anchor_pos/anchor_ori_full/
  ee_body_pos, so any other training-mutated term (foot_pos_xyz) leaks
  through the merge. `build_eval_command` therefore **explicitly re-pins
  foot_pos_xyz at stock 0.2**, and a structural test
  (`test_eval_command_pins_every_actionable_term_eval_yaml_misses`) fails
  if a termination knob is ever added to the action space without its
  eval pin. In the first v3 comparison run this leak was live — the
  manager's s3–s6 evals ran with its own foot_pos_xyz=0.25 — and the
  affected segments were re-evaluated with the pin (see
  `experiments/curriculum-manager-phase2/COMPARISON_V3_RESULTS.md`).
- `parse_metrics_eval` maps the honest smoke-scale metrics:
  `success_rate` (0.0 until a policy completes a full motion — measured
  0.0 across the whole 10k baseline), `progress_rate` (moves first),
  `mpjpe_all_mean` (= eval/all/mpjpe_g; **executed-frame survivor bias** —
  read jointly with progress_rate, never alone), per-motion
  progress/mpjpe/terminated, `failed_keys`. NaN aggregates (success-side
  stats when nothing succeeds) are dropped, not propagated.

## Files

- `job_adapter.py` — `KNOB_TO_HYDRA` (registry knob → verified Hydra path;
  raises on unmapped knobs — the adapter never invents config paths),
  `build_train_command`, `build_eval_command`, `parse_console_log`,
  `parse_metrics_eval`, `JobAdapter` (launch_segment / wait /
  parse_segment / eval_segment / rollback_launch), CLI.
- `test_job_adapter.py` — 17 tests, including parsing a **real log
  excerpt** and a **real metrics_eval.json** from verified runs, plus
  end-to-end checks that parsed records flow into `sonic-run-digest`.
- `testdata/train_log_excerpt.txt` — iterations 4–5 of the real
  `wbc_smoke` run (the parser's ground truth).
- `testdata/metrics_eval_real.json` — the real eval output from the 10k
  baseline checkpoint (2026-07-02; the eval parser's ground truth,
  including its NaN literals).

## Quick start

```bash
cd skills/agentic/sonic-job-adapter
python3 -m pytest test_job_adapter.py -q      # 17 passed

# dry-run: print the exact launch command for a knob override
python3 job_adapter.py command --name seg1 --iterations 20 --num-envs 64 \
  --knob uniform_sampling_rate=0.25

# parse a training log (host copy or straight from the container)
python3 job_adapter.py parse --log /workspace/wbc-training-logs/myrun.log \
  --container --out-prefix myrun
# -> myrun_train.jsonl + myrun_sampler.jsonl for sonic-run-digest
```

```python
from job_adapter import JobAdapter
ad = JobAdapter(project="manager", num_envs=256)
s1 = ad.launch_segment("seg1", iterations=50, knobs={"uniform_sampling_rate": 0.1})
ad.wait(s1)                    # polls; snapshots last.pt on success
s2 = ad.launch_segment("seg2", iterations=50,
                       knobs={"uniform_sampling_rate": 0.15},
                       checkpoint_in=s1.snapshot)
ad.wait(s2)
# tripwire fired? →
rb = ad.rollback_launch(s2, "seg2_rollback", iterations=50)
```

## Console-log → digest mapping

One train + one sampler record per `Learning iteration N` block:

| Console label | Digest key |
|---|---|
| `Mean rewards` / `Mean length` / `Mean entropy` | `Episode/rew_mean` / `Episode/len_mean` / `loss/entropy_avg` |
| `Env/Episode_Reward/<term>` | `Episode/<term>` |
| `Env/Episode_Termination/<term>`, `Env/Metrics/motion/error_*` | same, `Env/` stripped |
| `Env/adp_samp/<stat>` | sampler record `<stat>` (failure_rate_min/max/mean, prob_*, effective_num_bins, num_concentrated_bins) |

Also extracted: `Loaded checkpoint from step N` (resume verification),
experiment dir, Traceback count (segment health).

### Sampler stream caveat (honest)

The console prints **aggregate** `adp_samp` stats, not the per-bin
failure-rate vector, so `digest.sampler.normalized_entropy` /
`cap_saturation_fraction` cannot be computed from this stream.
`effective_num_bins` and `prob_max_over_uniform` are the working
concentration proxies (effective bins ↓ + prob_max_over_uniform ↑ =
concentration ↑). Options if the true vector is needed later: extend the
digest builder to accept these proxies natively, or add a tiny logging
callback in-container that dumps `_motion_lib.adp_samp_failure_rate` to
JSONL. For the ON-vs-OFF smoke, the proxies suffice.

## Traps

- One training process at a time — `launch_segment` refuses if
  `train_agent_trl.py` is already running (single GPU).
- `wait()` polls process existence, then parses; a segment with any
  Traceback is marked `failed` and gets no snapshot.
- Knob values land in the **next** segment only; verify in the new run's
  saved `config.yaml` (the adapter's parse reports the experiment dir).
- Container is durable state, not reproducible from a Dockerfile —
  see `docs/infra-guide.md` traps 3–4.

## Related

- `docs/infra-guide.md` — the verified infra this wraps.
- `sonic-run-digest` — consumes the JSONL this produces.
- `sonic-knob-registry` — validates decisions BEFORE they become segments;
  `KNOB_TO_HYDRA` covers exactly the registry's `available` + `patch`
  Family-A/B/C knobs that map to config (schedule_dict knobs excluded).
- `sonic-curriculum-manager` — the playbook whose decisions drive this.
