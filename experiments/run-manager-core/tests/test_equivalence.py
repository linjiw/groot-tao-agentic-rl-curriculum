# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the two-gate run-equivalence verdict (core/equivalence.py).

Fixture values mirror the real E5c evidence: rep3-s8 vs rep4-s8 first-iter
rewards (0.69315 vs 0.66776 -> rel dev ~3.7e-2) and bit-identical
probe_a/probe_b series.
"""

import math

import pytest

from core.equivalence import (
    E5B_CHAOS_FLOOR_MEAN,
    E5B_CHAOS_FLOOR_POINTWISE,
    VERDICT_BIT_IDENTICAL,
    VERDICT_DIVERGED,
    VERDICT_INCOMPARABLE,
    VERDICT_WITHIN_TAU,
    EquivalenceGate,
    calibrate_tau,
    max_relative_deviation,
    mean_relative_deviation,
)

REP3_S8_HEAD = [0.69315, 1.10942, 1.15485]
REP4_S8_HEAD = [0.66776, 1.15485, 1.10942]
# mean(REP3)=0.985807, mean(REP4)=0.977343 -> mean rel dev ~8.59e-3


# ── gate 0: bit identity ─────────────────────────────────────────────
def test_bit_identical_short_circuits_without_tau():
    g = EquivalenceGate(tau=None)  # no tau injected on purpose
    r = g.compare(REP3_S8_HEAD, list(REP3_S8_HEAD))
    assert r.verdict == VERDICT_BIT_IDENTICAL and r.equivalent
    assert r.max_rel_dev == 0.0


def test_bit_identical_tolerates_paired_nans():
    g = EquivalenceGate(tau=None)
    r = g.compare([1.0, math.nan], [1.0, math.nan])
    assert r.verdict == VERDICT_BIT_IDENTICAL


# ── gate 1: numeric tolerance (window-mean statistic) ────────────────
def test_rep4_divergence_fails_tight_tau():
    # window-mean rel dev = |0.985807-0.977343|/0.985807 ~ 8.59e-3 > 1e-3
    g = EquivalenceGate(tau=1e-3)
    r = g.compare(REP3_S8_HEAD, REP4_S8_HEAD)
    assert r.verdict == VERDICT_DIVERGED and not r.equivalent
    assert math.isclose(r.mean_rel_dev, 8.585e-3, rel_tol=1e-3)
    # pointwise diagnostics still reported:
    # idx0 dev ~3.66e-2; idx1 dev = 0.04543/1.15485 ~3.93e-2 -> argmax 1
    assert r.argmax_index == 1
    assert math.isclose(r.max_rel_dev, (1.15485 - 1.10942) / 1.15485,
                        rel_tol=1e-9)


def test_rep4_divergence_passes_loose_tau():
    g = EquivalenceGate(tau=0.05)
    r = g.compare(REP3_S8_HEAD, REP4_S8_HEAD)
    assert r.verdict == VERDICT_WITHIN_TAU and r.equivalent


def test_gate_is_mean_based_not_pointwise():
    # E5b take-3 lesson: chaos makes pointwise deviations huge (~28%)
    # while window means stay close. A pair with one big pointwise spike
    # but near-identical means must PASS a tau far below the spike.
    a = [1.0, 1.0, 1.0, 1.0, 1.30]
    b = [1.0, 1.0, 1.0, 1.30, 1.0]  # same mean, pointwise dev ~23%
    g = EquivalenceGate(tau=1e-3)
    r = g.compare(a, b)
    assert r.verdict == VERDICT_WITHIN_TAU
    assert r.max_rel_dev > 0.2      # diagnostic shows the spike
    assert r.mean_rel_dev < 1e-9    # gated stat is clean


def test_no_tau_and_not_bit_identical_raises():
    g = EquivalenceGate(tau=None)
    with pytest.raises(RuntimeError, match="E5b chaos floor"):
        g.compare([1.0], [1.0 + 1e-9])


def test_nan_mismatch_diverges():
    g = EquivalenceGate(tau=0.5)
    r = g.compare([1.0, math.nan], [1.0, 2.0])
    assert r.verdict == VERDICT_DIVERGED and math.isinf(r.max_rel_dev)


def test_length_mismatch_incomparable():
    g = EquivalenceGate(tau=0.1)
    r = g.compare([1.0, 2.0], [1.0])
    assert r.verdict == VERDICT_INCOMPARABLE and not r.equivalent


def test_bad_tau_rejected():
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError):
            EquivalenceGate(tau=bad)


# ── helpers ──────────────────────────────────────────────────────────
def test_max_relative_deviation_basics():
    dev, idx = max_relative_deviation([1.0, 2.0], [1.0, 2.2])
    assert idx == 1 and math.isclose(dev, 0.2 / 2.2)
    assert max_relative_deviation([], []) == (None, None)
    with pytest.raises(ValueError):
        max_relative_deviation([1.0], [1.0, 2.0])


def test_calibrate_tau():
    assert math.isclose(calibrate_tau(1e-3), 3e-3)
    with pytest.raises(ValueError):        # zero floor not credible
        calibrate_tau(0.0)
    with pytest.raises(ValueError):        # swallows effect of interest
        calibrate_tau(1e-2, min_effect_dev=2e-2)
    # ok when effect safely above tau
    assert calibrate_tau(1e-3, min_effect_dev=1e-1) == pytest.approx(3e-3)


def test_mean_relative_deviation_basics():
    assert mean_relative_deviation([], []) is None
    assert mean_relative_deviation([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert math.isinf(mean_relative_deviation([1.0, math.nan], [1.0, 2.0]))
    with pytest.raises(ValueError):
        mean_relative_deviation([1.0], [1.0, 2.0])


def test_tau_from_measured_e5b_floor():
    # The production calibration path: 3x the measured E5b take-3
    # window-mean chaos floor -> tau ~ 3.93e-2. Must not swallow a 10%
    # effect; must sit above the kernel-event deviation (4.18e-3, rep4).
    tau = calibrate_tau(E5B_CHAOS_FLOOR_MEAN, min_effect_dev=0.10)
    assert math.isclose(tau, 3 * 1.31e-2)
    assert tau > 4.18e-3                       # rep4 kernel event passes
    assert E5B_CHAOS_FLOOR_POINTWISE > 0.2     # pointwise stat unusable
    # end-to-end: the rep4-style pair is equivalent under measured tau
    r = EquivalenceGate(tau=tau).compare(REP3_S8_HEAD, REP4_S8_HEAD)
    assert r.verdict == VERDICT_WITHIN_TAU
