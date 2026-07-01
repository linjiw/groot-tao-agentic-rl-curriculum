# Curriculum-Manager Phase 0 — replay harness + decision-loop validation

**Status: DONE (2026-07-01). 50/50 new tests passing [measured]; full repo CPU
suite 93/93 [measured]. No GPU, no LLM, no live trainer — by design.**

Design doc: `docs/design/08-curriculum-manager-agent.md` (Phase 0 of §8).

## What was built

Three components, mirroring the doc-08 decision loop (§3):

| Component | Where | Tests |
|---|---|---|
| Knob registry + static decision validator | `skills/agentic/sonic-knob-registry/` (`registry.yaml`, `knob_registry.py`) | 22 |
| Digest builder (logs → `digest.json`) | `skills/agentic/sonic-run-digest/` (`digest_builder.py`) | 13 |
| Replay harness + deterministic playbook core | here (`replay_harness.py`, `scenarios.py`) | 15 |

The harness wires them into the full tick loop — **digest → decide →
validate → apply → tripwire-watch → journal** — with `BandStepperPolicy`
standing in for the LLM: a deterministic implementation of the playbook's
mechanical core (ADR-style dual-threshold stepping with hysteresis, §6.1;
sampler-health regulation via failure-vector entropy, §6.2). The real LLM
policy plugs into the same `propose(digest, state, registry)` interface later;
the harness owns validation/tripwire/journal, so the policy is never trusted.

## Acceptance behaviors verified (all [measured])

| Scenario | Expected (doc 08 Phase 0) | Result |
|---|---|---|
| `healthy` | mostly do-nothing, then ONE bounded tighten after sustained held-out success ≥ t_high | 1 applied (anchor_pos 0.30→0.25), 0 rollbacks, 0 rejected |
| `thrash` (success oscillates across the band) | hysteresis + sustained-trend rules → **zero** interventions | 0 applied |
| `plateau` (stuck success + concentrated failure mass) | Family-A sampler floor moves; threshold **never** tightens; cooldowns respected | uniform_sampling_rate stepped ×1.5 per move up to the 0.5 hard cap; no threshold change |
| `regression` (held-out collapses after an applied change) | tripwire fires after N sustained breaches → auto-revert + decision marked `failed_rolled_back` | 1 rollback, value restored, journal annotated |
| no `heldout_success_rate` in logs | protected-metric rule: **no action at all** | 0 applied |
| rogue policy (out-of-registry knob / oversized step) | every proposal statically rejected with auditable errors | 6/6 rejected |

Also verified: journal is JSON-serializable and carries rationale/outcome per
entry; scenarios are seed-deterministic; outcome attribution (`met` /
`regressed` / `failed_rolled_back`) feeds back into the next tick's digest
(`decision_history`) — the AURA-style memory of §6.5.

## Registry ground-truthing (pinned WBC `0e35637`)

Knob defaults/paths re-verified against source this session, not copied from
design docs: `motion.yaml:16–25` (`bin_size: 50`, `uniform_sampling_rate: 0.1`,
cap `50.0`; `sonic_release.yaml:71` overrides cap to `200`),
`motion_lib_base.py:2531–2577` (failure-rate EMA + cap/clip/renorm semantics —
the digest's `cap_saturation_fraction` mirrors these lines),
`base_adaptive_strict_ori_foot_xyz.yaml` (strict 0.15/0.15/0.2) vs `eval.yaml`
(relaxed 0.25 — **excluded** from the action space),
`ppo_im_phc.yaml:16–22` (`entropy_coef 0.01`, `desired_kl 0.01`),
trainer metric names `policy/approxkl_avg`, `loss/entropy_avg`, `Episode/*`,
`scheduled_params/*` (`ppo_trainer.py:1578–1631`, `1874–1905`),
eval names `success_rate`/`failed_keys` (`im_eval_callback.py:741–815`).

## Run it

```bash
PY=/home/ec2-user/.local/bin/python3.10   # any py3.9+ with pytest+pyyaml
$PY -m pytest . ../../skills/agentic/sonic-knob-registry ../../skills/agentic/sonic-run-digest -q
$PY replay_harness.py healthy --ticks 12 --journal-out /tmp/journal.json
$PY replay_harness.py regression --ticks 12   # watch the tripwire fire
```

## Honest limits / what Phase 0 does NOT show

- The digest's JSONL inputs are **synthetic**; producing them from a live run
  (wandb export / stdout parse) is the Phase-2 `sonic-job-adapter`'s job.
- `heldout_success_rate` does not exist in stock SONIC — the separate held-out
  eval watcher is still to be built; until then the policy correctly refuses
  all action (tested).
- `BandStepperPolicy` validates the *loop and guardrails*, not LLM judgment.
  Phase 1 swaps in the LLM behind the same interface against a toy live run.
- Family-B "patch"-status knobs assume the stage-2 termination patch; "design"
  knobs (DR ramp, retire/replay, penalty ramps) are validator-rejected outside
  replay mode (`allow_design=True` is replay-only, tested).
