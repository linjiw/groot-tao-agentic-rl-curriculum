# E6 — tier-0 journal equivalence gate: wired & verified (2026-07-09)

## What shipped

`core/equivalence.py` gains the tier-0 journal gate (E6), closing the
chain E5 → E5c → E5b → **E6**:

- `measured_tau(min_effect_dev=0.10)` — the production τ:
  3 × E5B_CHAOS_FLOOR_MEAN(1.31e-2) = **3.93e-2 [measured]**, guarded
  against swallowing a 10% effect. SONIC/A10G evidence, not a universal
  law; re-measure on engine/horizon/env-count change.
- `journal_series(journal, field)` — per-segment metric series from a
  RunManager journal; lifecycle event entries skipped, None → NaN so the
  gate's NaN discipline applies.
- `compare_journals(a, b, tau, fields)` → `JournalGateReport` — the E6
  gate proper: each field (default `rew_mean_last`, `len_mean_last`)
  through the two-gate EquivalenceGate; composite verdict = worst field
  (severity: bit_identical < within_tau < incomparable < diverged).
  Segment-count mismatch is INCOMPARABLE, never truncated.

## Verification (all [measured], this box)

- `uv run --python 3.13 --with pytest --with pyyaml python -m pytest
  tests/ -q` → **147/147 passed** (138 → 147, +9: measured_tau ×1,
  journal-gate unit ×6, real-journal ×2).
- Real Phase-2 journals judged by the gate itself:
  - `probe_a` vs `probe_b` (E5c determinism pair) →
    **bit_identical** — matches the E5c byte-for-byte ground truth.
  - `v4_seed42` vs `v4_seed1337` (different seeds, 10 segments) →
    NOT bit-identical (as required); mean_rel_dev **8.14e-3** <
    τ=3.93e-2 → within_tau; pointwise max 2.65e-1 (diagnostic only).

## Finding worth keeping

Different SEEDS land within τ at the 10-segment horizon
(8.14e-3 vs floor 1.31e-2 — seed-to-seed mean deviation is *below* the
single-run chaos floor measured over 50 iters; window widths differ, so
compare qualitatively only). This independently corroborates E1's
verdict that knob effects at this horizon are noise-dominated, and it
sets the correct expectation for the gate's role: **tier-0 equivalence
is for replay/re-run/rollback verification, not for detecting arm
effects** — effect claims need the confidence-gated tier (doc 09 §7
amendment 6), not a tighter τ.

## Semantics fixed by evidence chain

| Gate | Statistic | Floor [measured] | Role |
|------|-----------|------------------|------|
| 0 | bit identity | — (E5c: deterministic given identical inputs) | short-circuit pass only |
| 1 | window-mean rel dev | 1.31e-2 (E5b take-3) | THE gate, τ=3.9e-2 |
| — | pointwise max rel dev | 2.78e-1 (E5b take-3) | diagnostic only |
