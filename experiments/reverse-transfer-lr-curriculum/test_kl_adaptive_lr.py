"""Unit tests for the SONIC KL-adaptive LR controller port (and curriculum scheduler).

Run:
    /workspace/Isaac-GR00T/.venv/bin/python -m pytest test_kl_adaptive_lr.py -v
"""

import math

import pytest

from kl_adaptive_lr import KLAdaptiveLR
from curriculum_schedule import interpolate_schedule, update_scheduled_params


# --------------------------------------------------------------------------- #
# KLAdaptiveLR  -- the required cases
# --------------------------------------------------------------------------- #

DESIRED_KL = 0.01
LR_MIN = 1e-5
LR_MAX = 1e-2


def make_controller(lr=1e-3, factor=1.5):
    return KLAdaptiveLR(
        desired_kl=DESIRED_KL, lr_min=LR_MIN, lr_max=LR_MAX, lr=lr, factor=factor
    )


def test_kl_above_2x_band_shrinks_by_factor():
    c = make_controller(lr=1e-3)
    # kl_mean above desired_kl * 2.0 -> shrink
    new_lr = c.update(kl_mean=DESIRED_KL * 3.0)
    assert new_lr == pytest.approx(1e-3 / 1.5)


def test_shrink_clamped_at_lr_min():
    c = make_controller(lr=LR_MIN * 1.2)  # /1.5 would drop below lr_min
    new_lr = c.update(kl_mean=DESIRED_KL * 5.0)
    assert new_lr == pytest.approx(LR_MIN)


def test_kl_below_half_band_grows_by_factor():
    c = make_controller(lr=1e-3)
    new_lr = c.update(kl_mean=DESIRED_KL / 4.0)  # below desired_kl/2 and > 0
    assert new_lr == pytest.approx(1e-3 * 1.5)


def test_grow_clamped_at_lr_max():
    c = make_controller(lr=LR_MAX / 1.2)  # *1.5 would exceed lr_max
    new_lr = c.update(kl_mean=DESIRED_KL / 4.0)
    assert new_lr == pytest.approx(LR_MAX)


def test_kl_in_band_holds():
    c = make_controller(lr=1e-3)
    # exactly at desired_kl: inside band -> hold
    assert c.update(kl_mean=DESIRED_KL) == pytest.approx(1e-3)
    # anywhere strictly inside (d/2, d*2) holds
    assert c.update(kl_mean=DESIRED_KL * 1.5) == pytest.approx(1e-3)


def test_grow_branch_zero_kl_guard_holds():
    # kl_mean == 0.0 is below desired_kl/2 but the > 0.0 guard blocks the grow.
    c = make_controller(lr=1e-3)
    assert c.update(kl_mean=0.0) == pytest.approx(1e-3)


def test_grow_branch_negative_kl_guard_holds():
    # negative KL (numerical artifact) must NOT grow the lr.
    c = make_controller(lr=1e-3)
    assert c.update(kl_mean=-0.5) == pytest.approx(1e-3)


def test_band_boundaries_are_exclusive_hold():
    # SONIC uses strict > and < ; exactly on 2x or 0.5x boundary -> hold.
    c = make_controller(lr=1e-3)
    assert c.update(kl_mean=DESIRED_KL * 2.0) == pytest.approx(1e-3)  # not > 2x
    c2 = make_controller(lr=1e-3)
    assert c2.update(kl_mean=DESIRED_KL / 2.0) == pytest.approx(1e-3)  # not < 0.5x


def test_desired_kl_none_is_inert():
    c = KLAdaptiveLR(desired_kl=None, lr_min=LR_MIN, lr_max=LR_MAX, lr=1e-3)
    assert c.update(kl_mean=999.0) == pytest.approx(1e-3)
    assert c.update(kl_mean=0.0) == pytest.approx(1e-3)


def test_optimizer_param_groups_updated():
    class FakeOptimizer:
        def __init__(self, lr):
            self.param_groups = [{"lr": lr}, {"lr": lr}]

    opt = FakeOptimizer(1e-3)
    c = make_controller(lr=1e-3)
    new_lr = c.update(kl_mean=DESIRED_KL * 3.0, optimizer=opt)
    assert new_lr == pytest.approx(1e-3 / 1.5)
    for pg in opt.param_groups:
        assert pg["lr"] == pytest.approx(new_lr)


def test_optimizer_none_still_returns_lr():
    c = make_controller(lr=1e-3)
    assert c.update(kl_mean=DESIRED_KL * 3.0, optimizer=None) == pytest.approx(1e-3 / 1.5)


def test_multi_step_trajectory():
    """A multi-step trajectory exercising shrink, hold, grow and clamps."""
    c = make_controller(lr=1e-3, factor=1.5)
    lrs = []

    # step 1: high KL -> shrink
    lrs.append(c.update(kl_mean=0.05))            # 1e-3 / 1.5
    # step 2: still high -> shrink again
    lrs.append(c.update(kl_mean=0.05))            # /1.5 again
    # step 3: in band -> hold
    lrs.append(c.update(kl_mean=0.01))
    # step 4: low KL -> grow
    lrs.append(c.update(kl_mean=0.001))           # *1.5
    # step 5: zero KL -> guard holds
    lrs.append(c.update(kl_mean=0.0))

    expected = [
        1e-3 / 1.5,
        1e-3 / 1.5 / 1.5,
        1e-3 / 1.5 / 1.5,
        1e-3 / 1.5 / 1.5 * 1.5,
        1e-3 / 1.5 / 1.5 * 1.5,
    ]
    for got, exp in zip(lrs, expected):
        assert got == pytest.approx(exp)
    # every lr stayed within clamp bounds
    for lr in lrs:
        assert LR_MIN <= lr <= LR_MAX


def test_custom_factor():
    c = make_controller(lr=1e-3, factor=2.0)
    assert c.update(kl_mean=0.05) == pytest.approx(1e-3 / 2.0)
    c2 = make_controller(lr=1e-3, factor=2.0)
    assert c2.update(kl_mean=0.001) == pytest.approx(1e-3 * 2.0)


# --------------------------------------------------------------------------- #
# curriculum_schedule  -- interpolation + @-path smoke tests
# --------------------------------------------------------------------------- #

def test_linear_interpolation_midpoint():
    v = interpolate_schedule("linear", [0, 100], [5.0, 10.0], 50)
    assert v == pytest.approx(7.5)


def test_linear_holds_at_and_after_final():
    assert interpolate_schedule("linear", [0, 100], [5.0, 10.0], 100) == pytest.approx(10.0)
    assert interpolate_schedule("linear", [0, 100], [5.0, 10.0], 150) == pytest.approx(10.0)


def test_linear_multi_segment():
    steps = [0, 100, 200]
    vals = [0.0, 10.0, 30.0]
    assert interpolate_schedule("linear", steps, vals, 50) == pytest.approx(5.0)
    assert interpolate_schedule("linear", steps, vals, 150) == pytest.approx(20.0)


def test_segment_step_function():
    steps = [0, 100, 200]
    vals = [1.0, 2.0, 3.0]
    assert interpolate_schedule("segment", steps, vals, 50) == pytest.approx(1.0)
    assert interpolate_schedule("segment", steps, vals, 100) == pytest.approx(2.0)
    assert interpolate_schedule("segment", steps, vals, 250) == pytest.approx(3.0)


def test_update_scheduled_params_simple_attr():
    class Host:
        def __init__(self):
            self.lr = 0.0

    h = Host()
    sched = {"lr": {"type": "linear", "seg_steps": [0, 100], "seg_vals": [1.0, 2.0]}}
    out = update_scheduled_params(h, sched, 50)
    assert h.lr == pytest.approx(1.5)
    assert out["lr"] == pytest.approx(1.5)


def test_update_scheduled_params_at_path_and_bracket():
    class EventCfg(dict):
        pass

    class Host:
        def __init__(self):
            self.cfg = {"params": {"x": [5.0, 9.0]}}

    h = Host()
    # target: cfg['params']['x'][0] -> the last '@' split isolates the final accessor
    sched = {
        "cfg@['params']['x'][0]": {
            "type": "linear",
            "seg_steps": [0, 100],
            "seg_vals": [5.0, 10.0],
        }
    }
    update_scheduled_params(h, sched, 50)
    assert h.cfg["params"]["x"][0] == pytest.approx(7.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
