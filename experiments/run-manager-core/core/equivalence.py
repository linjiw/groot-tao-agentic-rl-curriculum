# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run-equivalence gate (the "tau two-gate" verdict) — engine-agnostic.

Decides whether two runs of the same configuration should be treated as
"the same run" for reproducibility accounting. Semantics fixed by the
E5/E5c evidence chain (2026-07-08, curriculum-manager-phase2):

- Gate 0 (bit gate, SHORT-CIRCUIT ONLY): if the two metric series are
  bit-identical, the runs are equivalent — full stop. E5c measured that
  the training process is bit-deterministic given identical inputs
  (probe_a == probe_b == rep3-s8, byte-for-byte). However, rare one-off
  kernel-level events (rep4-s8; ~1/31 segments observed) make bit
  DIFFERENCE inconclusive, so a failed bit gate must NOT fail the
  verdict. It only forwards to gate 1.

- Gate 1 (numeric tolerance, THE REAL GATE): compare an AGGREGATE
  statistic of each series (window mean), not per-point deviations.
  E5b take-3 (2026-07-08, entropy_coef 0.01 -> 0.0100001, one fp32-ULP-
  scale perturbation of the loss, same snapshot/seed) measured the pure
  chaotic divergence floor against the bit-verified reference:

      per-iter max rel dev : 2.77e-1   (chaos swallows pointwise compare)
      full-50-iter mean    : 1.31e-2   <- the usable floor
      (cross-check: rep4 kernel-event run vs ref, same config,
       full-50 mean rel dev 4.18e-3 — below the floor, as it must be)

  So the pointwise statistic is USELESS as a gate (its chaos floor is
  ~28%); only window aggregates carry signal. tau gates the relative
  deviation of the window means and must sit above 1.31e-2. tau still
  has no default: callers inject it (see calibrate_tau).

  E5b history worth keeping: take-1 (0.15 -> 0.15000001) rounds back to
  the same fp32 value, and take-2 (0.15 -> 0.1500001, fp32-distinct)
  perturbed a COMPARISON knob that no sample ever landed inside — both
  came back bit-identical over 1250 metric lines. Epsilon-perturbing a
  knob does not necessarily perturb the dynamics; chaos probes must
  inject into a continuously-acting path (loss coefficients, lr).

Nothing here imports an engine. Series are plain float lists, typically
per-iteration `Episode/rew_mean` plus final eval scalars.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Sequence

VERDICT_BIT_IDENTICAL = "bit_identical"        # gate 0 pass
VERDICT_WITHIN_TAU = "within_tau"              # gate 0 fail, gate 1 pass
VERDICT_DIVERGED = "diverged"                  # both gates fail
VERDICT_INCOMPARABLE = "incomparable"          # length/NaN mismatch

# Measured chaos floor for the FULL-WINDOW MEAN statistic (E5b take-3,
# 50-iter segment, 256 envs, seed 42, SONIC/A10G). Relative deviation of
# the 50-iter mean reward under a one-ULP-scale loss perturbation.
# Re-measure when the engine, horizon, or env count changes materially.
E5B_CHAOS_FLOOR_MEAN = 1.31e-2
# Same probe, per-iteration max relative deviation — kept as a warning
# sign only: any pointwise gate would need tau > this, i.e. ~28%, which
# would swallow every effect of interest. Do not gate pointwise.
E5B_CHAOS_FLOOR_POINTWISE = 2.78e-1


@dataclasses.dataclass
class GateReport:
    """Machine-readable verdict for one series pair."""

    verdict: str
    bit_identical: bool
    max_rel_dev: Optional[float]      # pointwise worst (diagnostic only)
    argmax_index: Optional[int]       # where the worst deviation sits
    mean_rel_dev: Optional[float]     # window-mean deviation (THE gated stat)
    tau: Optional[float]
    n_points: int
    detail: str = ""

    @property
    def equivalent(self) -> bool:
        return self.verdict in (VERDICT_BIT_IDENTICAL, VERDICT_WITHIN_TAU)

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)


def max_relative_deviation(
    a: Sequence[float], b: Sequence[float], eps: float = 1e-12,
) -> tuple[Optional[float], Optional[int]]:
    """max_i |a_i - b_i| / max(|a_i|, |b_i|, eps) and its argmax.

    Returns (None, None) for empty input. Raises ValueError on length
    mismatch (that is an INCOMPARABLE condition the caller must surface,
    not silently truncate)."""
    if len(a) != len(b):
        raise ValueError(f"series length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return None, None
    worst, worst_i = 0.0, 0
    for i, (x, y) in enumerate(zip(a, b)):
        if math.isnan(x) or math.isnan(y):
            if math.isnan(x) and math.isnan(y):
                continue  # both-NaN positions carry no signal
            return math.inf, i
        d = abs(x - y) / max(abs(x), abs(y), eps)
        if d > worst:
            worst, worst_i = d, i
    return worst, worst_i


def mean_relative_deviation(
    a: Sequence[float], b: Sequence[float], eps: float = 1e-12,
) -> Optional[float]:
    """|mean(a) - mean(b)| / max(|mean(a)|, |mean(b)|, eps).

    The window-mean statistic gated by gate 1. E5b take-3 measured its
    chaos floor at 1.31e-2 over a 50-iter window, vs ~2.8e-1 for the
    pointwise max — aggregates average the chaos out. NaNs poison the
    mean by design (returns inf on any NaN mismatch; both-NaN positions
    are skipped)."""
    if len(a) != len(b):
        raise ValueError(f"series length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return None
    sa = sb = 0.0
    n = 0
    for x, y in zip(a, b):
        if math.isnan(x) or math.isnan(y):
            if math.isnan(x) and math.isnan(y):
                continue
            return math.inf
        sa += x
        sb += y
        n += 1
    if n == 0:
        return None
    ma, mb = sa / n, sb / n
    return abs(ma - mb) / max(abs(ma), abs(mb), eps)


class EquivalenceGate:
    """Two-gate run-equivalence verdict. tau is injected, not defaulted.

    Gate 1 verdicts are decided on the WINDOW-MEAN deviation (see
    mean_relative_deviation); the pointwise max is reported as a
    diagnostic only, per the E5b take-3 measurement."""

    def __init__(self, tau: Optional[float] = None):
        if tau is not None and (tau <= 0 or math.isnan(tau)):
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau

    def compare(self, a: Sequence[float], b: Sequence[float]) -> GateReport:
        n = len(a)
        if len(b) != n:
            return GateReport(
                verdict=VERDICT_INCOMPARABLE, bit_identical=False,
                max_rel_dev=None, argmax_index=None, mean_rel_dev=None,
                tau=self.tau, n_points=n,
                detail=f"length mismatch: {n} vs {len(b)}")
        # gate 0: bit identity (short-circuit pass only)
        if all((x == y) or (math.isnan(x) and math.isnan(y))
               for x, y in zip(a, b)):
            return GateReport(
                verdict=VERDICT_BIT_IDENTICAL, bit_identical=True,
                max_rel_dev=0.0 if n else None,
                argmax_index=None, mean_rel_dev=0.0 if n else None,
                tau=self.tau, n_points=n)
        # gate 1: numeric tolerance on the window mean
        if self.tau is None:
            raise RuntimeError(
                "series are not bit-identical and no tau was injected; "
                "calibrate tau from the E5b chaos floor before asking for "
                "a numeric verdict")
        dev, idx = max_relative_deviation(a, b)
        mdev = mean_relative_deviation(a, b)
        if mdev is not None and not math.isinf(mdev) and mdev <= self.tau:
            return GateReport(
                verdict=VERDICT_WITHIN_TAU, bit_identical=False,
                max_rel_dev=dev, argmax_index=idx, mean_rel_dev=mdev,
                tau=self.tau, n_points=n)
        return GateReport(
            verdict=VERDICT_DIVERGED, bit_identical=False,
            max_rel_dev=dev, argmax_index=idx, mean_rel_dev=mdev,
            tau=self.tau, n_points=n,
            detail="window-mean relative deviation exceeds tau"
                   if not math.isinf(mdev or 0) else "NaN mismatch")


def calibrate_tau(
    chaos_floor_dev: float,
    min_effect_dev: Optional[float] = None,
    safety_factor: float = 3.0,
) -> float:
    """tau := safety_factor * chaos floor, sanity-checked against the
    smallest effect size we want to DETECT (tau must stay below it).

    chaos_floor_dev: max relative deviation measured by an E5b-style
        probe (same snapshot, epsilon-perturbed knob) against the
        deterministic reference.
    min_effect_dev: optional smallest real-effect deviation that must
        NOT be swallowed; raises if tau would swallow it.
    """
    if chaos_floor_dev < 0 or math.isnan(chaos_floor_dev):
        raise ValueError(f"bad chaos floor: {chaos_floor_dev}")
    tau = safety_factor * chaos_floor_dev
    if tau == 0.0:
        raise ValueError(
            "chaos floor of exactly 0 is not credible for a chaotic "
            "training process; re-measure before calibrating tau")
    if min_effect_dev is not None and tau >= min_effect_dev:
        raise ValueError(
            f"tau={tau:.3g} would swallow the smallest effect of interest "
            f"({min_effect_dev:.3g}); chaos floor too high or safety "
            f"factor too generous")
    return tau


def measured_tau(min_effect_dev: Optional[float] = 0.10) -> float:
    """The production tau: 3x the E5b take-3 measured window-mean chaos
    floor -> ~3.93e-2 [measured 2026-07-08, SONIC/A10G, 50-iter window,
    256 envs]. Guards by default against swallowing a 10% effect (the
    smallest arm-level effect Phase-2 treated as real).

    Re-measure the floor (an E5b-style probe on a continuously-acting
    knob) when the engine, horizon, or env count changes materially —
    this constant is SONIC/A10G evidence, not a universal law."""
    return calibrate_tau(E5B_CHAOS_FLOOR_MEAN, min_effect_dev=min_effect_dev)


# ── E6: tier-0 journal gate ──────────────────────────────────────────
# Journal-level wiring of the two-gate verdict: two RunManager journals
# (core.journal entry lists) in, one machine-readable equivalence
# verdict out. This is the tier-0 reproducibility gate: replays,
# control-arm re-runs and rollback re-executions are judged here before
# any higher-tier (analytic-controller / LLM) comparison is allowed to
# claim an effect.

# journal fields gated by default: the two training-side series every
# per-segment entry carries (byte-compatible with Phase-2 journals)
DEFAULT_GATED_FIELDS = ("rew_mean_last", "len_mean_last")


def journal_series(journal: Sequence[Dict],
                   field: str = "rew_mean_last") -> List[float]:
    """Per-segment metric series from a journal entry list.

    Only per-segment entries count (entries carrying `segment` AND the
    field key); lifecycle event entries (`segment_failed`,
    `disk_gate_failed`, rollback events) are skipped — two runs that
    diverge in WHICH segments ran will differ in series length and land
    on INCOMPARABLE, which is the honest verdict. A present-but-None
    value becomes NaN so the gate's NaN discipline applies (paired
    None/None positions carry no signal; None vs number poisons)."""
    out: List[float] = []
    for e in journal:
        if "segment" not in e or field not in e:
            continue
        v = e[field]
        out.append(float("nan") if v is None else float(v))
    return out


# severity order for the composite verdict: the worst field wins
_VERDICT_RANK = {VERDICT_BIT_IDENTICAL: 0, VERDICT_WITHIN_TAU: 1,
                 VERDICT_INCOMPARABLE: 2, VERDICT_DIVERGED: 3}


@dataclasses.dataclass
class JournalGateReport:
    """Composite verdict over one journal pair: per-field GateReports
    plus the worst-field overall verdict."""

    verdict: str                       # worst field verdict
    fields: Dict[str, GateReport]      # per-field reports
    tau: Optional[float]

    @property
    def equivalent(self) -> bool:
        return self.verdict in (VERDICT_BIT_IDENTICAL, VERDICT_WITHIN_TAU)

    def to_dict(self) -> Dict:
        return {"verdict": self.verdict, "tau": self.tau,
                "equivalent": self.equivalent,
                "fields": {k: r.to_dict() for k, r in self.fields.items()}}


def compare_journals(
    journal_a: Sequence[Dict], journal_b: Sequence[Dict],
    tau: Optional[float] = None,
    fields: Sequence[str] = DEFAULT_GATED_FIELDS,
) -> JournalGateReport:
    """E6 tier-0 gate: judge two RunManager journals equivalent or not.

    Each field's per-segment series goes through the two-gate
    EquivalenceGate (gate 0 bit identity; gate 1 window-mean tolerance
    at `tau`). The composite verdict is the WORST field verdict —
    a single diverged series fails the pair, and a length mismatch
    (different segment counts) is INCOMPARABLE, never silently
    truncated. `tau=None` still allows a BIT_IDENTICAL composite but
    raises (via EquivalenceGate) as soon as any field needs a numeric
    verdict; pass measured_tau() for the calibrated production gate."""
    if not fields:
        raise ValueError("compare_journals needs at least one field")
    gate = EquivalenceGate(tau=tau)
    reports: Dict[str, GateReport] = {}
    for field in fields:
        reports[field] = gate.compare(journal_series(journal_a, field),
                                      journal_series(journal_b, field))
    worst = max(reports.values(), key=lambda r: _VERDICT_RANK[r.verdict])
    return JournalGateReport(verdict=worst.verdict, fields=reports, tau=tau)
