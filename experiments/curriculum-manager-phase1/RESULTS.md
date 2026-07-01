# Curriculum-Manager Phase 1 — real LLM in the closed loop (toy live run)

**Status: DONE (2026-07-01). 14/14 tests [measured]; LLM arm run manually
against the live toy loop — decisions correct in both regimes [measured,
journals below]. CPU-only (the "A10G" phase needed no GPU: the toy run is
numpy).**

Design doc: `docs/design/08-curriculum-manager-agent.md` (Phase 1 of §8):
*"validate the full tick loop end-to-end with a live optimizer. Measures the
manager's behavior, not SONIC performance."*

## What was built

- `toy_tracking_run.py` — a knob-responsive live training loop (numpy,
  seconds per run): 60 synthetic motions with difficulties, per-motion skill
  that grows with sampling mass, a failure-weighted sampler with **SONIC's
  floor/cap semantics** (`uniform_sampling_rate`,
  `adp_samp_failure_rate_max_over_mean`), a termination-threshold knob that
  trades success margin against training pressure, and a real held-out
  subset (never sampled; improves only via generalization spillover —
  tested). Emits the exact train/eval/sampler record shapes
  `sonic-run-digest` consumes.
- `live_loop.py` — Phase-0 guardrail layout (digest → decide → validate →
  apply → tripwire → journal; the policy is never trusted) driving the live
  run. Three policies behind the same `propose(digest, state, registry)`
  interface:
  - `none` — control arm;
  - `band` — Phase-0 `BandStepperPolicy` (deterministic baseline);
  - `llm` — **`LLMPolicy`: shells out to `claude -p` with the
    `sonic-curriculum-manager` playbook + the digest**, parses one fenced
    JSON decision; any subprocess/parse failure degrades to `action: none`
    with the failure recorded in the journal (fail-safe, tested).
- `test_live_loop.py` — 14 tests: toy-run determinism/record shapes/knob
  responsiveness (floor→entropy, threshold→pressure, held-out spillover),
  live-loop guardrails (null/band/rogue arms), LLM output parser fail-safes.

## Measured results

Toy run, seed 0, 16 ticks (250 iters/tick):

| Arm | held-out first → last | applied | rejected | rollbacks |
|---|---|---|---|---|
| `none` | 0.330 → 0.611 | 0 | 0 | 0 |
| `band` | 0.330 → 0.603 | 3 (anchor_pos loosened 0.30→0.45 via row-1 contraction) | 0 | 0 |

**LLM arm (the Phase-1 deliverable), 6-tick runs:**

1. **Low-band rising run** (seed 0 fresh start; held-out 0.33→0.45, below
   t_low but climbing): LLM emitted `none` **all 6 ticks**, with rationales
   citing exact digest numbers — tick 1–2 correctly blocked by the
   ≥3-consecutive-evals sustain rule, ticks 3–6 correctly matching no table
   row ("held-out ≤ t_low but trend rising — row 1 requires NOT rising").
   0 applied, 0 rejected. [journal: /tmp/llm_journal3.json of the session]
2. **In-band competent run** (skill≈0.85 start; held-out 0.88→0.95): LLM
   correctly did nothing for 2 ticks (sustain not yet met), **tightened
   `anchor_pos` 0.30→0.25 at tick 3** (row 2, citing "0.8811/0.8935/0.9017
   ≥ t_high 0.85 ×3"), held during cooldown at ticks 4–5 *explicitly citing
   the cooldown*, then **tightened `ee_body_pos` at tick 6**. Both decisions
   passed the validator unmodified; first decision scored `met`.

Every LLM rationale was digest-grounded (quoted the actual numbers), and no
LLM output ever needed the validator to save it — but the rogue-policy test
confirms the loop would reject illegal output anyway.

## Playbook fix found by the LLM arm (the point of Phase 1)

The first LLM run exposed a real spec bug: decision-table **row 1
(contraction)** originally required "a Family-B threshold was tightened
within the last ~6 ticks" — so a run that *starts* below the band could
never get relief; and without a trend condition, a run climbing out on its
own would get pointlessly loosened. Row 1 now reads: held-out ≤ t_low ×3
**AND trend NOT rising** → loosen. The deterministic `band` baseline had the
same blind spot; its journal now shows sensible contraction on the low-band
run.

## Reproduce

```bash
PY=~/.local/bin/python3.10          # needs numpy (pip install --user)
$PY -m pytest test_live_loop.py -q  # 14 passed
$PY live_loop.py --policy none --ticks 16
$PY live_loop.py --policy band --ticks 16
$PY live_loop.py --policy llm  --ticks 6 --journal-out journal.json  # needs `claude` CLI
```

## Honest limits

- The toy dynamics are hand-made; they demonstrate the **loop** (actions
  have consequences the manager must live with), not curriculum value.
  Whether managing beats defaults is Phase 2/3's question, on real SONIC.
- The LLM arm ran 12 ticks total across two regimes — enough to verify
  procedure-following and digest-grounded reasoning, not a statistical
  claim about LLM-vs-band quality.
- `claude -p` latency is ~30–60 s/tick; fine at checkpoint cadence (the
  real cadence is ~minutes-hours), irrelevant for the toy.
