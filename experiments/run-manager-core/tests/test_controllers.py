# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the tier-0 analytic controllers (core/controllers.py).

Invariants tested map 1:1 to the provenance claims in the module
docstring: PBHC sigma monotonicity/floor, EGM bin-LP normalization,
grounding-floor hardness (doc 09 §7 amendment 4), and determinism
(pure state machine — same feed => same state, no RNG inside).
"""

import math

import pytest

from core import BinLPSampler, SigmaEMAController


# ── SigmaEMAController ────────────────────────────────────────────────
class TestSigmaEMA:
    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            SigmaEMAController(sigma_init=0.0)
        with pytest.raises(ValueError):
            SigmaEMAController(sigma_init=-1.0)
        with pytest.raises(ValueError):
            SigmaEMAController(sigma_init=1.0, ema_rate=0.0)
        with pytest.raises(ValueError):
            SigmaEMAController(sigma_init=1.0, ema_rate=1.5)
        with pytest.raises(ValueError):
            SigmaEMAController(sigma_init=1.0, sigma_floor=1.0)  # floor >= init
        with pytest.raises(ValueError):
            SigmaEMAController(sigma_init=1.0, sigma_floor=-0.1)

    def test_sigma_monotone_nonincreasing(self):
        """PBHC rule: sigma <- min(sigma, EMA(err)) can never increase,
        even when the error rises again after a low stretch."""
        c = SigmaEMAController(sigma_init=1.0, ema_rate=0.5)
        errs = [0.8, 0.3, 0.1, 0.9, 2.0, 5.0, 0.05]
        prev = c.sigma
        for e in errs:
            s = c.update(e)
            assert s <= prev + 1e-15
            prev = s

    def test_sigma_tracks_low_error_down(self):
        c = SigmaEMAController(sigma_init=1.0, ema_rate=1.0)  # EMA = last err
        c.update(0.5)
        assert c.sigma == pytest.approx(0.5)
        c.update(0.2)
        assert c.sigma == pytest.approx(0.2)
        # error rebounds — sigma holds (min keeps tightest tolerance)
        c.update(0.7)
        assert c.sigma == pytest.approx(0.2)

    def test_floor_is_hard(self):
        c = SigmaEMAController(sigma_init=1.0, ema_rate=1.0, sigma_floor=0.1)
        for _ in range(20):
            c.update(0.0)
        assert c.sigma == pytest.approx(0.1)

    def test_rejects_nan_and_negative(self):
        c = SigmaEMAController(sigma_init=1.0)
        with pytest.raises(ValueError):
            c.update(float("nan"))
        with pytest.raises(ValueError):
            c.update(float("inf"))
        with pytest.raises(ValueError):
            c.update(-0.1)
        assert c.n_updates == 0  # rejected feeds must not mutate state

    def test_reward_shape(self):
        c = SigmaEMAController(sigma_init=0.5)
        assert c.reward(0.0) == pytest.approx(1.0)
        assert c.reward(0.5) == pytest.approx(math.exp(-1.0))
        # tighter sigma => same error scores lower
        c.update(0.1); c.update(0.1)
        assert c.reward(0.5) < math.exp(-1.0)

    def test_determinism_same_feed_same_state(self):
        errs = [0.31, 0.27, 0.44, 0.12, 0.09, 0.55]
        a = SigmaEMAController(sigma_init=1.0, ema_rate=0.05)
        b = SigmaEMAController(sigma_init=1.0, ema_rate=0.05)
        for e in errs:
            a.update(e); b.update(e)
        assert a.state_dict() == b.state_dict()  # bit-identical, E5c-style


# ── BinLPSampler ─────────────────────────────────────────────────────
BINS = [("m1", 0), ("m1", 1), ("m2", 0), ("m2", 1)]


class TestBinLPSampler:
    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            BinLPSampler([])
        with pytest.raises(ValueError):
            BinLPSampler(["a", "a"])
        with pytest.raises(ValueError):
            BinLPSampler(BINS, fast_rate=0.01, slow_rate=0.1)  # slow >= fast
        with pytest.raises(ValueError):
            BinLPSampler(BINS, power=0.0)
        with pytest.raises(ValueError):
            BinLPSampler(BINS, uniform_mix=2.0)

    def test_cold_start_is_uniform(self):
        s = BinLPSampler(BINS, uniform_mix=0.2)
        p = s.probabilities()
        assert sum(p.values()) == pytest.approx(1.0)
        for v in p.values():
            assert v == pytest.approx(1.0 / len(BINS))

    def test_probabilities_sum_to_one_and_favor_lp(self):
        s = BinLPSampler(BINS, fast_rate=0.5, slow_rate=0.05, uniform_mix=0.2)
        # bin m1/0 improves fast (falling error => fast pulls away from slow)
        for e in [1.0, 0.8, 0.6, 0.4, 0.2]:
            s.observe(("m1", 0), e)
        # bin m2/0 is flat (no progress)
        for _ in range(5):
            s.observe(("m2", 0), 1.0)
        p = s.probabilities()
        assert sum(p.values()) == pytest.approx(1.0)
        assert p[("m1", 0)] > p[("m2", 0)]          # LP bin favored
        assert p[("m2", 1)] > 0                     # unseen bin still reachable

    def test_grounding_floor_every_bin_has_mass(self):
        """Doc 09 amendment 4: uniform floor guarantees min mass per bin."""
        s = BinLPSampler(BINS, uniform_mix=0.2, uniform_floor=0.05)
        for e in [1.0, 0.5, 0.1]:
            s.observe(("m1", 0), e)
        p = s.probabilities()
        for v in p.values():
            assert v >= 0.2 / len(BINS) - 1e-12

    def test_set_uniform_mix_clamps_to_floor(self):
        s = BinLPSampler(BINS, uniform_mix=0.3, uniform_floor=0.05)
        assert s.set_uniform_mix(0.5) == pytest.approx(0.5)
        # teacher tries to zero the grounding mass — HARD floor wins
        assert s.set_uniform_mix(0.0) == pytest.approx(0.05)
        assert s.uniform_mix == pytest.approx(0.05)
        assert s.set_uniform_mix(1.5) == pytest.approx(1.0)  # clamped up too

    def test_observe_validation(self):
        s = BinLPSampler(BINS)
        with pytest.raises(KeyError):
            s.observe(("nope", 9), 1.0)
        with pytest.raises(ValueError):
            s.observe(("m1", 0), float("nan"))

    def test_lp_is_two_timescale_abs_diff(self):
        s = BinLPSampler(BINS, fast_rate=1.0, slow_rate=0.5)
        s.observe(("m1", 0), 1.0)   # fast=1.0 slow=1.0
        s.observe(("m1", 0), 0.0)   # fast=0.0 slow=0.5
        assert s.learning_progress(("m1", 0)) == pytest.approx(0.5)
        assert s.learning_progress(("m2", 0)) == 0.0  # unseen

    def test_power_smoothing_flattens(self):
        def spread(power):
            s = BinLPSampler(BINS, fast_rate=1.0, slow_rate=0.5,
                             power=power, uniform_mix=0.05,
                             uniform_floor=0.05)
            s.observe(("m1", 0), 1.0); s.observe(("m1", 0), 0.0)   # LP 0.5
            s.observe(("m2", 0), 1.0); s.observe(("m2", 0), 0.9)   # LP 0.05
            p = s.probabilities()
            return p[("m1", 0)] - p[("m2", 0)]
        assert spread(0.5) < spread(1.0)  # lower power => flatter

    def test_determinism_same_feed_same_state(self):
        feed = [(("m1", 0), 0.9), (("m2", 1), 0.4), (("m1", 0), 0.7),
                (("m1", 1), 0.3), (("m2", 0), 0.8), (("m1", 0), 0.5)]
        a = BinLPSampler(BINS)
        b = BinLPSampler(BINS)
        for k, e in feed:
            a.observe(k, e); b.observe(k, e)
        assert a.state_dict() == b.state_dict()
        assert a.probabilities() == b.probabilities()
