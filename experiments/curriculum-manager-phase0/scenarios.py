# SPDX-License-Identifier: Apache-2.0
"""Synthetic run scenarios for the Curriculum-Manager replay harness.

Each scenario is a generator of per-tick record bundles shaped exactly like
the three JSONL streams `sonic-run-digest` consumes (train / eval / sampler,
SONIC metric names). Deterministic: seeded `random.Random`, no wall clock.

Scenarios (design doc 08 Phase 0 acceptance behaviors):
  healthy     — success climbs steadily; manager should mostly do nothing,
                then tighten thresholds once the success band is sustained.
  plateau     — success stalls below the band with concentrated sampler
                failure mass; the correct action is a Family-A sampler knob,
                never a Family-B tighten.
  thrash      — success oscillates across the band each eval; hysteresis +
                sustained-trend rules must hold (no decisions).
  regression  — success collapses right after a (simulated) applied change;
                the tripwire must fire and roll back.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Dict, Iterator, List, Optional


@dataclasses.dataclass
class TickRecords:
    """One manager tick's worth of new log records."""

    it: int
    train: List[dict]
    eval: List[dict]
    sampler: List[dict]


MOTION_KEYS = [f"motion_{i:03d}" for i in range(40)]
ITERS_PER_TICK = 250  # one eval pass per manager tick (doc 08 §3)


def _train_rec(it: int, rng: random.Random, kl: float = 0.01) -> dict:
    return {
        "it": it,
        "policy/approxkl_avg": kl * rng.uniform(0.8, 1.2),
        "loss/entropy_avg": rng.uniform(0.9, 1.1),
        "loss/value_avg": rng.uniform(0.4, 0.6),
        "loss/policy_avg": rng.uniform(-0.02, 0.02),
        "lr": 2e-5,
        "Policy/mean_noise_std": 0.8,
        "fps": 50000,
        "Episode/tracking_anchor_pos": rng.uniform(0.7, 0.9),
        "Episode/feet_acc": rng.uniform(-0.02, -0.005),
    }


def _eval_rec(
    it: int,
    success: float,
    heldout: float,
    rng: random.Random,
    n_failed: Optional[int] = None,
) -> dict:
    if n_failed is None:
        n_failed = max(0, round(len(MOTION_KEYS) * (1 - success)))
    failed = rng.sample(MOTION_KEYS, min(n_failed, len(MOTION_KEYS)))
    return {
        "it": it,
        "success_rate": round(max(0.0, min(1.0, success)), 4),
        "heldout_success_rate": round(max(0.0, min(1.0, heldout)), 4),
        "progress_rate": round(max(0.0, min(1.0, success + 0.05)), 4),
        "failed_keys": sorted(failed),
        "mpjpe_all_mean": round(120 * (1.1 - success), 2),
    }


def _sampler_rec(it: int, rng: random.Random, concentration: float = 0.2) -> dict:
    """concentration 0 → uniform failure rates; 1 → mass on a few bins.

    At concentration c, the 4 hot bins carry ~c of the total normalized
    mass, so normalized entropy drops well below 1 for c ≥ ~0.7.
    """
    n = 64
    base = [rng.uniform(0.8, 1.2) * (1 - concentration) / n for _ in range(n)]
    hot = rng.sample(range(n), 4)
    rates = [
        b + (concentration / 4 * rng.uniform(0.8, 1.2) if i in hot else 0.0)
        for i, b in enumerate(base)
    ]
    return {"it": it, "failure_rate": rates}


def _tick(
    tick_idx: int,
    success: float,
    heldout: float,
    rng: random.Random,
    concentration: float = 0.2,
    kl: float = 0.01,
) -> TickRecords:
    it0 = tick_idx * ITERS_PER_TICK
    train = [_train_rec(it0 + j * 50, rng, kl) for j in range(1, 6)]
    sampler = [_sampler_rec(it0 + j * 100, rng, concentration) for j in range(1, 3)]
    ev = [_eval_rec(it0 + ITERS_PER_TICK, success, heldout, rng)]
    return TickRecords(it=it0 + ITERS_PER_TICK, train=train, eval=ev, sampler=sampler)


# ── scenarios ────────────────────────────────────────────────────────
def healthy(n_ticks: int = 12, seed: int = 0) -> Iterator[TickRecords]:
    """Success climbs 0.55 → ~0.95; held-out tracks a touch lower."""
    rng = random.Random(seed)
    for t in range(n_ticks):
        s = min(0.95, 0.55 + 0.04 * t) + rng.uniform(-0.005, 0.005)
        yield _tick(t, s, s - 0.03, rng, concentration=0.15)


def plateau(n_ticks: int = 12, seed: int = 1) -> Iterator[TickRecords]:
    """Success stuck ~0.62 with heavy sampler concentration."""
    rng = random.Random(seed)
    for t in range(n_ticks):
        s = 0.62 + rng.uniform(-0.01, 0.01)
        yield _tick(t, s, s - 0.02, rng, concentration=0.9)


def thrash(n_ticks: int = 12, seed: int = 2) -> Iterator[TickRecords]:
    """Success alternates 0.80 / 0.55 across the band every eval."""
    rng = random.Random(seed)
    for t in range(n_ticks):
        s = 0.80 if t % 2 == 0 else 0.55
        yield _tick(t, s + rng.uniform(-0.01, 0.01), s - 0.02, rng)


def regression(
    n_ticks: int = 12, seed: int = 3, collapse_after_tick: int = 5
) -> Iterator[TickRecords]:
    """Healthy until `collapse_after_tick`, then held-out success collapses."""
    rng = random.Random(seed)
    for t in range(n_ticks):
        if t <= collapse_after_tick:
            s = min(0.95, 0.70 + 0.03 * t)
            h = s - 0.02
        else:
            s = 0.65 - 0.05 * (t - collapse_after_tick)
            h = s - 0.15  # held-out collapses harder
        yield _tick(t, s, h, rng)


SCENARIOS: Dict[str, callable] = {
    "healthy": healthy,
    "plateau": plateau,
    "thrash": thrash,
    "regression": regression,
}
