# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tier-0 analytic inner controllers (engine-agnostic).

Implements doc 09 §7 amendment 1 (three-tier control): the fast knobs are
owned by cheap analytic controllers running at per-step / per-rollout
cadence; the LLM (tier 2) supervises only their META-parameters, which is
why every constructor argument here is a plain float/int suitable for the
knob registry. Provenance of the two rules:

- `SigmaEMAController` — KungfuBot/PBHC (arXiv 2506.12851, NeurIPS 2025,
  real Unitree G1): adapt the tracking-reward tolerance sigma in
  r(x) = exp(-x/sigma) via  sigma <- min(sigma, EMA(tracking_error)),
  derived from bi-level optimization. Doc-09 verified [2-0, 3-0]: no
  fixed sigma is optimal across motions; the adaptive rule is
  near-optimal on all motion types. We add a floor (sigma_floor) so the
  monotone shrink cannot collapse the reward to a delta function — the
  floor is a tier-2 meta-knob.

- `BinLPSampler` — EGM (arXiv 2512.19043, G1 sim; medium confidence,
  sim-only): segment motions into fixed-duration bins keyed by
  (motion, bin); per-bin EMA of composite tracking error; sampling
  probability from normalized learning progress with power smoothing and
  a uniform-mix ratio. Doc-09 [verified 3-0]: removing the module
  degrades test E_mpkpe 57.35 -> 71.04 mm. Learning progress uses the
  two-timescale form |fast_EMA - slow_EMA| (Matiisen-style absolute LP,
  signal hierarchy: doc 09 §7 amendment 7 makes per-bin LP the primary
  signal).

Grounding guardrail (doc 09 §7 amendment 4): the sampler enforces a HARD
lower bound `uniform_floor` on the uniform sampling mass. The scheduled
`uniform_mix` above the floor is teacher-controlled; the floor itself is
not overridable downward at runtime.

Both controllers are deterministic pure-state machines: no RNG inside
(the sampler returns a probability vector; drawing from it is the
caller's job with the caller's seeded RNG), no wall-clock, no I/O — so
they inherit the E5c determinism result and pass through the E6
journal-equivalence gate unchanged.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Hashable, List, Optional, Sequence, Tuple


# ── sigma-EMA reward-tolerance controller (PBHC rule) ────────────────
@dataclasses.dataclass
class SigmaEMAController:
    """Adaptive tracking-reward tolerance: sigma <- max(floor, min(sigma, EMA(err))).

    Meta-parameters (tier-2 supervised, registry-ready):
      sigma_init: starting tolerance (> 0).
      ema_rate:   EMA smoothing rate alpha in (0, 1]; ema <- (1-a)*ema + a*err.
      sigma_floor: hard lower bound (>= 0, < sigma_init); prevents the
                   monotone shrink from collapsing r(x)=exp(-x/sigma).

    Invariants (tested):
      - sigma is monotone non-increasing over updates;
      - sigma >= sigma_floor always;
      - sigma never increases even if the error EMA rises again
        (min() keeps the tightest tolerance reached — the PBHC form).
    """

    sigma_init: float
    ema_rate: float = 0.001
    sigma_floor: float = 1e-4

    def __post_init__(self) -> None:
        if not (self.sigma_init > 0):
            raise ValueError(f"sigma_init must be > 0, got {self.sigma_init}")
        if not (0 < self.ema_rate <= 1):
            raise ValueError(f"ema_rate must be in (0, 1], got {self.ema_rate}")
        if not (0 <= self.sigma_floor < self.sigma_init):
            raise ValueError(
                f"sigma_floor must be in [0, sigma_init), got {self.sigma_floor}")
        self.sigma: float = float(self.sigma_init)
        self._ema: Optional[float] = None
        self.n_updates: int = 0

    def update(self, tracking_error: float) -> float:
        """Feed one tracking-error observation; returns the new sigma.

        Non-finite or negative errors are rejected (raise) rather than
        silently clamped — a NaN error means the caller's pipeline is
        broken and must not shrink the tolerance.
        """
        if not math.isfinite(tracking_error) or tracking_error < 0:
            raise ValueError(f"tracking_error must be finite and >= 0, "
                             f"got {tracking_error}")
        if self._ema is None:
            self._ema = float(tracking_error)
        else:
            a = self.ema_rate
            self._ema = (1.0 - a) * self._ema + a * float(tracking_error)
        self.sigma = max(self.sigma_floor, min(self.sigma, self._ema))
        self.n_updates += 1
        return self.sigma

    def reward(self, tracking_error: float) -> float:
        """Current tracking reward r(x) = exp(-x/sigma) (does NOT update)."""
        return math.exp(-tracking_error / self.sigma)

    # -- tier-2 meta-knob setters (registry-mediated) ------------------
    def set_ema_rate(self, value: float) -> float:
        """Meta-knob: EMA smoothing rate. Must stay in (0, 1]."""
        v = float(value)
        if not (0 < v <= 1) or not math.isfinite(v):
            raise ValueError(f"ema_rate must be in (0, 1], got {value}")
        self.ema_rate = v
        return v

    def set_sigma_floor(self, value: float) -> float:
        """Meta-knob: hard lower bound on sigma.

        Raising the floor ABOVE the current sigma would force sigma UP,
        violating the monotone-non-increasing invariant (the PBHC rule's
        core property) — so the applied value is clamped to
        min(requested, current sigma). Returns the value actually
        applied; the caller journals requested vs applied.
        """
        v = float(value)
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"sigma_floor must be finite and >= 0, got {value}")
        applied = min(v, self.sigma)
        self.sigma_floor = applied
        return applied

    def state_dict(self) -> Dict[str, float]:
        return {"sigma": self.sigma,
                "ema": float("nan") if self._ema is None else self._ema,
                "n_updates": self.n_updates}


# ── bin-LP sampler (EGM rule + LP signal + grounding floor) ──────────
@dataclasses.dataclass
class _BinStat:
    fast: Optional[float] = None
    slow: Optional[float] = None
    n: int = 0


class BinLPSampler:
    """Per-bin learning-progress sampler with uniform-mix grounding floor.

    Bins are opaque hashable keys, typically (motion_id, start_bin) —
    the sampler never interprets them (engine-agnostic).

    Meta-parameters (tier-2 supervised, registry-ready):
      fast_rate / slow_rate: two-timescale EMA rates, 0 < slow < fast <= 1.
          LP_i = |fast_i - slow_i| (absolute learning progress).
      power: smoothing exponent on LP before normalization (EGM's power
          smoothing); 0 < power. power->0 flattens, 1 = proportional.
      uniform_mix: scheduled uniform mass in [uniform_floor, 1].
      uniform_floor: HARD grounding lower bound on uniform mass
          (doc 09 §7 amendment 4). `set_uniform_mix` clamps UP to it.

    Unseen bins carry no LP; they receive mass only through the uniform
    component — which is exactly why the grounding floor must be > 0 in
    any real run (default 0.05).
    """

    def __init__(self, bins: Sequence[Hashable], *,
                 fast_rate: float = 0.1, slow_rate: float = 0.01,
                 power: float = 1.0, uniform_mix: float = 0.2,
                 uniform_floor: float = 0.05) -> None:
        if len(bins) == 0:
            raise ValueError("bins must be non-empty")
        if len(set(bins)) != len(bins):
            raise ValueError("bins must be unique")
        if not (0 < slow_rate < fast_rate <= 1):
            raise ValueError(f"need 0 < slow_rate < fast_rate <= 1, got "
                             f"slow={slow_rate} fast={fast_rate}")
        if not (power > 0):
            raise ValueError(f"power must be > 0, got {power}")
        if not (0 <= uniform_floor <= 1):
            raise ValueError(f"uniform_floor must be in [0,1], got {uniform_floor}")
        self.bins: List[Hashable] = list(bins)
        self.fast_rate = float(fast_rate)
        self.slow_rate = float(slow_rate)
        self.power = float(power)
        self.uniform_floor = float(uniform_floor)
        self.uniform_mix = max(float(uniform_mix), self.uniform_floor)
        if self.uniform_mix > 1:
            raise ValueError(f"uniform_mix must be <= 1, got {uniform_mix}")
        self._stats: Dict[Hashable, _BinStat] = {b: _BinStat() for b in bins}

    # -- observation path --------------------------------------------
    def observe(self, bin_key: Hashable, error: float) -> None:
        """Feed one composite-tracking-error observation for a bin."""
        if bin_key not in self._stats:
            raise KeyError(f"unknown bin {bin_key!r}")
        if not math.isfinite(error):
            raise ValueError(f"error must be finite, got {error}")
        s = self._stats[bin_key]
        e = float(error)
        s.fast = e if s.fast is None else (1 - self.fast_rate) * s.fast + self.fast_rate * e
        s.slow = e if s.slow is None else (1 - self.slow_rate) * s.slow + self.slow_rate * e
        s.n += 1

    # -- control path (tier-2 meta-knob entry point) ------------------
    def set_uniform_mix(self, value: float) -> float:
        """Teacher-scheduled uniform mass; clamped to [uniform_floor, 1].

        Returns the value actually applied — the caller journals BOTH the
        requested and the applied value so a clamped request is visible.
        """
        applied = min(1.0, max(float(value), self.uniform_floor))
        self.uniform_mix = applied
        return applied

    def set_power(self, value: float) -> float:
        """Meta-knob: LP smoothing exponent. Must stay > 0 and finite."""
        v = float(value)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"power must be finite and > 0, got {value}")
        self.power = v
        return v

    # -- read path -----------------------------------------------------
    def learning_progress(self, bin_key: Hashable) -> float:
        s = self._stats[bin_key]
        if s.fast is None or s.slow is None:
            return 0.0
        return abs(s.fast - s.slow)

    def probabilities(self) -> Dict[Hashable, float]:
        """Sampling distribution: (1-u) * LP-proportional + u * uniform.

        If ALL bins have zero LP (cold start or fully plateaued), the
        LP component degenerates to uniform — total is uniform.
        Guaranteed: sums to 1 (within fp), every bin >= u/n_bins > 0
        whenever uniform_floor > 0.
        """
        n = len(self.bins)
        lp = [self.learning_progress(b) ** self.power
              if self.learning_progress(b) > 0 else 0.0 for b in self.bins]
        total = sum(lp)
        if total <= 0:
            lp_part = [1.0 / n] * n
        else:
            lp_part = [v / total for v in lp]
        u = self.uniform_mix
        return {b: (1 - u) * p + u / n for b, p in zip(self.bins, lp_part)}

    def state_dict(self) -> Dict[str, object]:
        return {
            "uniform_mix": self.uniform_mix,
            "uniform_floor": self.uniform_floor,
            "bins": {repr(b): {"fast": s.fast, "slow": s.slow, "n": s.n}
                     for b, s in self._stats.items()},
        }
