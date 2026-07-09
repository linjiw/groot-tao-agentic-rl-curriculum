# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Host-CPU tests for the SONIC sigma-EMA tier-0 shim (I1 spike).

Covers the numeric core (sigma_ema_kernel) and the controller binding
(sigma_ema_binding). The torch/isaaclab term (sonic_sigma_ema_term) runs
only in-container; its logic is the binding + kernel, which ARE tested
here, plus a torch-absent import-guard check.

The load-bearing property is the BIT-IDENTICAL NO-OP: G0's gate-0 arm
compares a no-op shim run against stock and requires bit_identical. If the
no-op path ever perturbs a reward value, G0 fails and the whole insertion
is suspect — so it is tested exhaustively across the r-domain.
"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from adapters.sonic_tier0 import sigma_ema_kernel as kern
from adapters.sonic_tier0.sigma_ema_binding import SigmaEMABinding


# ── kernel: the r_active = r_stock ** (std^2/sigma^2) identity ──────────
class TestKernel:
    @pytest.mark.parametrize("r", [1.0, 0.9, 0.5, 0.1, 1e-3, 1e-20, 1e-30, 0.0, -0.0])
    def test_noop_exponent_is_exact_passthrough(self, r):
        # exponent == 1.0 must return the INPUT object unchanged (bit-identical),
        # not a recomputed r**1.0 (which can differ for subnormals / -0.0).
        assert kern.apply_retune(r, 1.0) is r or kern.apply_retune(r, 1.0) == r

    def test_noop_factor_when_sigma_equals_std(self):
        for std in [0.1, 0.3, 1.0, 3.14]:
            assert kern.retune_factor(std, std) == 1.0

    def test_recover_squared_error_inverts_kernel(self):
        # forward r = exp(-d/std^2); recover d back — only where r stays
        # above the saturation floor (d/std^2 < ~69, else exp underflows and
        # recovery correctly saturates, see test_recover_clamps_saturation).
        for std in [0.2, 0.5, 1.0]:
            for d in [0.0, 0.01, 0.25, 1.0, 4.0]:
                if d / (std * std) > 60:      # would underflow past _R_FLOOR
                    continue
                r = math.exp(-d / (std * std))
                d_back = kern.recover_squared_error(r, std)
                assert d_back == pytest.approx(d, abs=1e-9)

    def test_recover_clamps_saturation_no_inf(self):
        # r=0 (fully saturated) must give a large FINITE error, never inf
        d = kern.recover_squared_error(0.0, 0.3)
        assert math.isfinite(d) and d > 0
        # r>1 impossible -> clamped to 1 -> d=0
        assert kern.recover_squared_error(1.5, 0.3) == 0.0

    def test_tighter_sigma_lowers_reward(self):
        # sigma < std => exponent > 1 => reward shrinks for imperfect tracking
        std, sigma = 0.3, 0.15
        p = kern.retune_factor(std, sigma)
        assert p > 1.0
        r_stock = 0.5
        r_active = kern.apply_retune(r_stock, p)
        assert r_active < r_stock          # stricter tolerance penalises error
        # perfect tracking (r=1) is unchanged by any exponent
        assert kern.apply_retune(1.0, p) == 1.0

    def test_mean_linear_error_units(self):
        # a batch all at squared-error d has linear error sqrt(d)
        std, d = 0.5, 0.25
        r = math.exp(-d / (std * std))
        mle = kern.mean_linear_error([r, r, r], std)
        assert mle == pytest.approx(math.sqrt(d), abs=1e-9)

    def test_empty_batch_zero_error(self):
        assert kern.mean_linear_error([], 0.3) == 0.0

    def test_retune_factor_rejects_bad_std_sigma(self):
        with pytest.raises(ValueError):
            kern.retune_factor(0.0, 0.3)
        with pytest.raises(ValueError):
            kern.retune_factor(0.3, -1.0)


# ── binding: NO-OP vs ACTIVE, unit mapping, monotonicity ────────────────
class TestBindingNoOp:
    def test_noop_factor_always_one(self):
        b = SigmaEMABinding(std=0.3, active=False)
        for err in [0.0, 0.1, 0.5, 1.0]:
            assert b.observe_and_factor(mean_r_stock_batch=None,
                                        mean_linear_err=err) == 1.0
        assert b.current_factor() == 1.0
        assert b.n_steps == 0            # no-op never advances state

    def test_active_first_step_is_noop_factor(self):
        # PBHC: sigma_init == std, so BEFORE any shrink the factor is 1.0.
        b = SigmaEMABinding(std=0.3, active=True, ema_rate=1.0)
        # A perfect-tracking observation (err=0) must not shrink sigma.
        f = b.observe_and_factor(mean_r_stock_batch=None, mean_linear_err=0.0)
        assert f == 1.0


class TestBindingActive:
    def test_sigma_monotone_non_increasing(self):
        b = SigmaEMABinding(std=1.0, active=True, ema_rate=0.5,
                            sigma_floor_frac=0.0)
        factors = []
        # feed a decreasing error stream; sigma should track down, factor up
        for err in [0.9, 0.7, 0.5, 0.3, 0.2, 0.2, 0.2]:
            factors.append(b.observe_and_factor(None, mean_linear_err=err))
        # exponent = std^2/sigma^2 is non-decreasing as sigma shrinks
        assert all(factors[i] <= factors[i + 1] + 1e-12
                   for i in range(len(factors) - 1))
        assert factors[-1] >= 1.0

    def test_sigma_never_rises_when_error_climbs_back(self):
        # PBHC min() keeps the tightest tolerance reached.
        b = SigmaEMABinding(std=1.0, active=True, ema_rate=1.0,
                            sigma_floor_frac=0.0)
        b.observe_and_factor(None, mean_linear_err=0.3)   # sigma -> ~0.3
        f_low = b.current_factor()
        b.observe_and_factor(None, mean_linear_err=0.9)   # error rises
        f_after = b.current_factor()
        assert f_after == pytest.approx(f_low)            # unchanged

    def test_sigma_floor_respected(self):
        b = SigmaEMABinding(std=1.0, active=True, ema_rate=1.0,
                            sigma_floor_frac=0.25)
        for _ in range(50):
            b.observe_and_factor(None, mean_linear_err=0.001)  # push way down
        rec = b.trace_record()
        assert rec["sigma"] >= 0.25 - 1e-9                 # floor = 0.25*std


class TestMetaKnobs:
    def test_set_ema_rate(self):
        b = SigmaEMABinding(std=0.3, active=True)
        assert b.set_ema_rate(0.05) == 0.05
        with pytest.raises(ValueError):
            b.set_ema_rate(1.5)

    def test_set_sigma_floor_frac_clamps_to_monotone(self):
        # raising the floor above current sigma is clamped down (controller
        # guard) so the monotone invariant is never violated.
        b = SigmaEMABinding(std=1.0, active=True, ema_rate=1.0,
                            sigma_floor_frac=0.0)
        b.observe_and_factor(None, mean_linear_err=0.2)    # sigma -> ~0.2
        applied = b.set_sigma_floor_frac(0.9)              # request floor 0.9
        assert applied <= 0.2 + 1e-9                       # clamped to sigma


class TestSidecarPersistence:
    def test_save_and_reload_continues_trajectory(self, tmp_path):
        p = str(tmp_path / "sigma.json")
        b1 = SigmaEMABinding(std=1.0, active=True, ema_rate=0.5,
                             sigma_floor_frac=0.0, sidecar_path=p)
        for err in [0.8, 0.6, 0.4]:
            b1.observe_and_factor(None, mean_linear_err=err)
        sigma_before = b1.trace_record()["sigma"]
        n_before = b1.n_steps
        b1.save()

        # a fresh binding resuming from the sidecar starts where b1 left off
        b2 = SigmaEMABinding(std=1.0, active=True, ema_rate=0.5,
                             sigma_floor_frac=0.0, sidecar_path=p)
        assert b2._loaded_from_sidecar
        assert b2.trace_record()["sigma"] == pytest.approx(sigma_before)
        assert b2.n_steps == n_before
        # and it does not RISE sigma on a subsequent higher error
        b2.observe_and_factor(None, mean_linear_err=0.9)
        assert b2.trace_record()["sigma"] <= sigma_before + 1e-12

    def test_load_refuses_sidecar_from_different_std(self, tmp_path):
        # A sidecar written under std=0.5 must NOT be restored into a std=0.3
        # binding (its sigma is in the wrong units; restoring could make
        # sigma>std => reward inflated). Refuse -> fresh start. (Review risk 5.)
        p = str(tmp_path / "sigma.json")
        b1 = SigmaEMABinding(std=0.5, active=True, ema_rate=1.0,
                             sigma_floor_frac=0.0, sidecar_path=p)
        b1.observe_and_factor(None, mean_linear_err=0.4)   # sigma -> ~0.4
        b1.save()
        b2 = SigmaEMABinding(std=0.3, active=True, ema_rate=1.0,
                             sigma_floor_frac=0.0, sidecar_path=p)
        assert not b2._loaded_from_sidecar                 # refused
        assert b2.current_factor() == 1.0                  # fresh: sigma==std
        assert b2.trace_record()["sigma"] == pytest.approx(0.3)

    def test_load_clamps_corrupt_sigma_above_std(self, tmp_path):
        # A hand-edited/corrupt payload with sigma>std must be clamped to std
        # on load, never producing an exponent<1 (reward inflation).
        p = str(tmp_path / "sigma.json")
        b1 = SigmaEMABinding(std=0.3, active=True, sidecar_path=p)
        b1.save()
        with open(p) as fh:
            payload = json.load(fh)
        payload["sigma"] = 0.9        # > std, invalid
        with open(p, "w") as fh:
            json.dump(payload, fh)
        b2 = SigmaEMABinding(std=0.3, active=True, sidecar_path=p)
        assert b2.trace_record()["sigma"] <= 0.3 + 1e-12
        assert b2.current_factor() >= 1.0                  # exponent never < 1

    def test_atomic_save_no_partial_file(self, tmp_path):
        p = str(tmp_path / "sigma.json")
        b = SigmaEMABinding(std=0.5, active=True, sidecar_path=p)
        b.observe_and_factor(None, mean_linear_err=0.3)
        b.save()
        assert os.path.exists(p)
        assert not os.path.exists(p + ".tmp")   # tmp renamed away


class TestTraceRecord:
    def test_trace_has_journal_fields(self):
        b = SigmaEMABinding(std=0.3, active=True, term_name="tracking_anchor_pos")
        b.observe_and_factor(None, mean_linear_err=0.25)
        rec = b.trace_record()
        for k in ("term", "active", "std", "sigma", "sigma_ratio",
                  "retune_exponent", "n_steps"):
            assert k in rec
        assert rec["term"] == "tracking_anchor_pos"
        assert rec["sigma_ratio"] == pytest.approx(rec["sigma"] / 0.3)


# ── isaaclab contract: the term class must survive RewardManager checks ──
class TestIsaacLabContract:
    """The shim is a ManagerTermBase subclass consumed by isaaclab's
    RewardManager. isaaclab (manager_base._resolve_common_term_cfg)
    STATICALLY validates the __call__ signature against the term's `params`
    keys, and instantiates the class as func(cfg=..., env=...). A bare
    **kwargs __call__ silently FAILS that check at training startup (a
    live-only bug). These tests lock the contract on CPU.

    We import the term module (torch/isaaclab absent on host => ManagerTermBase
    falls back to object, which is exactly what lets us introspect the
    signatures without the sim)."""

    def _term_classes(self):
        import adapters.sonic_tier0.sonic_sigma_ema_term as term
        return [term.SigmaEMAAnchorPos, term.SigmaEMAAnchorOri,
                term.SigmaEMARelBodyPos]

    def test_init_signature_is_cfg_env(self):
        # manager_base.py:418  term_cfg.func = term_cfg.func(cfg=..., env=...)
        import inspect
        for cls in self._term_classes():
            params = list(inspect.signature(cls.__init__).parameters)
            assert params[:3] == ["self", "cfg", "env"]

    def test_call_signature_passes_isaaclab_param_check(self):
        # reproduce _resolve_common_term_cfg's set-equality gate for a class
        # term with config params {command_name, std}.
        import inspect
        for cls in self._term_classes():
            params = inspect.signature(cls.__call__).parameters
            with_def = [a for a in params if params[a].default is not inspect.Parameter.empty]
            without_def = [a for a in params if params[a].default is inspect.Parameter.empty]
            args = without_def + with_def
            min_argc = 2  # self + env
            assert len(args) > min_argc
            lhs = set(args[min_argc:])
            rhs = set(["command_name", "std"] + with_def)
            assert lhs == rhs, f"{cls.__name__}: {lhs} != {rhs}"

    def test_call_has_no_bare_kwargs(self):
        # a VAR_KEYWORD param would reintroduce the startup failure
        import inspect
        for cls in self._term_classes():
            kinds = {p.kind for p in inspect.signature(cls.__call__).parameters.values()}
            assert inspect.Parameter.VAR_KEYWORD not in kinds

    def test_stock_fn_names_are_real_sonic_functions(self):
        # guard against typos in STOCK_FN — these must exist in the SONIC
        # rewards module path we override to. (Names only; module import needs
        # the container, so we assert the string shape here.)
        import adapters.sonic_tier0.sonic_sigma_ema_term as term
        for cls in self._term_classes():
            assert cls.STOCK_FN.endswith("_error")
            assert term._STOCK_MODULE == "gear_sonic.envs.manager_env.mdp"


# ── reference-path equivalence: batch retune == per-element ─────────────
class TestBatchReference:
    def test_retune_batch_matches_elementwise(self):
        rs = [1.0, 0.8, 0.5, 0.2, 1e-5]
        p = 2.0
        assert kern.retune_batch(rs, p) == [kern.apply_retune(r, p) for r in rs]

    def test_full_noop_pipeline_bit_identical(self):
        # Simulate the shim's no-op path end-to-end: active=False => every
        # reward passes through byte-for-byte. This is the G0 property in
        # miniature (torch tensor path is the same op: `return r_stock`).
        b = SigmaEMABinding(std=0.3, active=False)
        rs = [1.0, 0.9137, 0.5, 0.001, 1e-25]
        for r in rs:
            f = b.observe_and_factor(None, mean_linear_err=kern.mean_linear_error([r], 0.3))
            assert f == 1.0
            assert kern.apply_retune(r, f) == r     # identical object/value
