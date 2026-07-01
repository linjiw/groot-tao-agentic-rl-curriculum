# Curriculum-Manager Phase 2 (part 1) — job adapter validated against real SONIC training

**Status: adapter DONE + live-validated (2026-07-01). 11/11 unit tests
[measured]; full segment→knob-change→rollback lifecycle executed against
real 64-env SONIC training on this box's A10G [measured, run dirs below].
The manager ON-vs-OFF smoke is now mechanically unblocked.**

Design doc: `docs/design/08-curriculum-manager-agent.md` §7 (skill table:
`sonic-job-adapter`) and §8 Phase 2. Infra: `docs/infra-guide.md`.

## What was validated live (all [measured])

Three real training segments via `skills/agentic/sonic-job-adapter`,
project `adapter_val`, 64 envs, `save_last_frequency=5`:

| Segment | What it proved | Evidence |
|---|---|---|
| `seg1` (8 iters, `uniform_sampling_rate=0.1`, fresh) | launch → wait → parse → snapshot | status `done`, 8 train + 8 sampler records, 0 tracebacks, `snapshot_seg1.pt` created (`seg1-20260701_214522/`) |
| `seg2` (8 iters, `uniform_sampling_rate=0.15`, from `snapshot_seg1.pt`) | knob-change segment: resume + override both land | log: "Loaded checkpoint from step 5"; saved `config.yaml:332` = `0.15` |
| `seg2_rb` (4 iters, rollback of seg2) | `rollback_launch` restores **pre-change state** | relaunched from `snapshot_seg1.pt` with knobs `{uniform_sampling_rate: 0.1}`; log confirms step-5 load; saved `config.yaml:332` = `0.1` |

Also verified: parsed train records flow into `sonic-run-digest`'s
`build_digest` (unit-tested end-to-end on the real log excerpt), and the
parser extracts per-iteration reward terms, termination fractions, motion
errors, and `Env/adp_samp/*` sampler-health stats from the rich console box.

## Bugs found by live validation (both fixed)

1. `is_training_running()` matched its **own pgrep wrapper** (`bash -c
   "pgrep -f train_agent_trl.py"` contains the pattern) → adapter refused
   every launch. Fix: `[t]rain` bracket pattern + require a python process.
2. `latest_checkpoint()` raised on absent `last.pt` (ls exit 2) — but a
   segment shorter than the save cadence legitimately has none. Fix:
   `|| true`, return None.

Both are exactly the class of plumbing bug the unit tests can't catch —
which is why the live validation step exists.

## Sampler stream caveat (carried into SKILL.md)

The console prints aggregate `adp_samp` stats (min/max/mean failure rate,
prob stats, `effective_num_bins`, `num_concentrated_bins`) — not the
per-bin vector. So the digest's `normalized_entropy`/`cap_saturation_fraction`
are unavailable from this stream; `effective_num_bins` +
`prob_max_over_uniform` are the concentration proxies for the smoke. If the
true vector is needed: small logging callback in-container dumping
`_motion_lib.adp_samp_failure_rate` to JSONL (future work).

## What's next (the actual ON-vs-OFF smoke)

1. Wire the loop: digest from `parse_segment` → `sonic-curriculum-manager`
   policy (band or LLM) → `sonic-knob-registry` validation → next
   `launch_segment`. All pieces now exist and are individually validated;
   the composition is a ~100-line driver.
2. Held-out watcher wiring (`filter_motion_keys` + eval terminations) —
   **currently limited by the 2-motion library** (bones-seed gated): with 2
   motions a held-out split is degenerate (`select_holdout` correctly
   refuses n=1 splits at sane fractions). The smoke can run manager-ON vs
   OFF on training-side metrics, but the protected-metric discipline needs
   the real motion library → bones-seed access remains the gating external
   dependency for a *meaningful* comparison.
3. Reproduce: `python3 -m pytest skills/agentic/sonic-job-adapter -q`;
   live lifecycle per the SKILL.md Quick start (needs the container).

## Run artifacts (in-container)

```
/workspace/wbc-training-logs/adapter_val/
  seg1-20260701_214522/     (config.yaml, last.pt, snapshot_seg1.pt)
  seg2-20260701_2147xx/     (config.yaml: rate=0.15)
  seg2_rb-20260701_214951/  (config.yaml: rate=0.1 — rollback verified)
/workspace/wbc-training-logs/adapter_val_seg{1,2,2_rb}.log
```
