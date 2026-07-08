# E5b — Chaos-Divergence Floor (final verdict, 2026-07-08)

Question: with the SAME snapshot/seed/config, how much do the training
metrics move when the loss is perturbed at ~fp32-ULP scale? That number
is the pure chaotic-divergence floor that any run-equivalence tolerance
τ must sit above.

All numbers below were extracted by the adjudicator directly from raw
logs on the box (`/workspace/wbc-training-logs/`), not from subagent
reports. Setup: SONIC motion-tracking, 50-iter segment, 256 envs,
seed 42, A10G, same starting snapshot as the E5c reference chain.

## Takes 1–2: two ways an "epsilon perturbation" can be a no-op

| Take | Knob perturbed | Outcome | Why |
|---|---|---|---|
| 1 | `0.15 → 0.15000001` | **BIT_IDENTICAL** (1250 metric lines) | delta below fp32 resolution — rounds back to the same float |
| 2 | `0.15 → 0.1500001` (fp32-distinct) | **BIT_IDENTICAL** (1250 metric lines) | knob only acts in a comparison; no sample ever landed inside the shifted band |
| 3 | `entropy_coef 0.01 → 0.0100001` | **DIVERGED** | coefficient multiplies the loss every update — continuously-acting path |

Lesson (now codified in `core/equivalence.py` docstring): a chaos probe
must inject into a **continuously-acting path** (loss coefficient, lr),
and the perturbed value must survive fp32 round-trip. Takes 1–2 also
double as extra evidence for gate 0: the process is bit-deterministic
when the effective inputs are unchanged.

## Take 3: the measured floor

Reference: `cmp_control_seed42_rep3_control_s8.log` (bit-verified twice
in E5c). Chaos run: `cmp_control_seed42_chaos3_control_s1.log`.
50 paired `Mean rewards:` values (+ 50 `Mean length:` values).

| Statistic | Mean rewards | Mean ep-length |
|---|---|---|
| per-iter max rel dev | **2.77e-1** (iter 46) | 1.79e-1 (iter 30) |
| per-iter mean rel dev | 6.6e-2 | 6.9e-2 |
| full-50 window-mean rel dev | **1.31e-2** | — |
| tail-10 window-mean rel dev | 3.75e-2 | — |

Cross-check (sanity): rep4-s8 — the E5c kernel-event run, config
truly identical — sits at full-50 mean rel dev **4.18e-3** vs the same
reference, i.e. *below* the chaos floor, exactly as it must be.
(Its per-iter max is 3.04e-1 — again unusable pointwise.)

## Verdict and consequences

1. **Pointwise gating is dead.** The chaos floor of the per-iteration
   max-rel-dev statistic is ~28%; any τ above that swallows every
   effect of interest. `EquivalenceGate` was rewritten to gate the
   **window-mean** deviation; pointwise max is diagnostic-only.
2. **Measured floor: 1.31e-2** (full-50 window mean), exported as
   `E5B_CHAOS_FLOOR_MEAN`.
3. **τ = 3 × 1.31e-2 ≈ 3.9e-2** via `calibrate_tau(safety_factor=3)`.
   Checks out on both sides: passes the rep4 kernel-event pair
   (4.18e-3 < τ) and does not swallow a 10% effect (τ < 0.10).
   Status upgrade: τ gate **[speculative] → [measured]**.
4. Caveat: floor measured at one horizon (50 iters) and one env count
   (256). Chaotic divergence grows with horizon — re-measure before
   applying τ to much longer windows.

Tests: `experiments/run-manager-core/tests/` — 138 passed (was 135;
+3 covering mean-based gating and the measured calibration path).

Unblocked: E6 tier-0 gate can now use the measured τ.
