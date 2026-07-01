# SPDX-License-Identifier: Apache-2.0
"""Tests for the knob registry + decision validator (design doc 08 §3/§5)."""

import pytest

from knob_registry import KnobRegistry, RunState, load_registry


@pytest.fixture()
def reg():
    return load_registry()


@pytest.fixture()
def state():
    return RunState(tick=10)


def decision(knob, value, **over):
    d = {
        "action": "set",
        "knob": knob,
        "value": value,
        "rationale": "test",
        "expected_effect": "test",
        "tripwire": {"metric": "heldout_success_rate", "drop_pct": 5, "evals": 3},
    }
    d.update(over)
    return d


# ── registry integrity ────────────────────────────────────────────────
def test_registry_loads_and_has_three_families(reg):
    families = {k["family"] for k in reg.knobs.values()}
    assert families == {"data_curriculum", "schedule", "optimizer"}


def test_every_knob_cites_a_source(reg):
    for name, k in reg.knobs.items():
        assert k.get("verified_source"), f"{name} missing verified_source"


def test_global_rules_present(reg):
    assert reg.meta["global_rules"]["max_changes_per_tick"] == 1
    assert reg.meta["global_rules"]["default_action"] == "none"


# ── the default action ────────────────────────────────────────────────
def test_action_none_always_valid(reg, state):
    assert reg.validate_decision({"action": "none"}, state)


def test_unknown_action_rejected(reg, state):
    res = reg.validate_decision({"action": "restart_everything"}, state)
    assert not res and "unknown action" in res.errors[0]


# ── whitelist / atomicity ─────────────────────────────────────────────
def test_unknown_knob_rejected(reg, state):
    res = reg.validate_decision(decision("per_clip_weights", 0.5), state)
    assert not res and "outside action space" in res.errors[0]


def test_multi_knob_payload_rejected(reg, state):
    res = reg.validate_decision(
        decision(["uniform_sampling_rate", "desired_kl"], 0.2), state
    )
    assert not res and "atomic" in res.errors[0]


def test_missing_rationale_or_tripwire_rejected(reg, state):
    res = reg.validate_decision(decision("uniform_sampling_rate", 0.15, rationale=""), state)
    assert not res
    res = reg.validate_decision(decision("uniform_sampling_rate", 0.15, tripwire=None), state)
    assert not res


def test_malformed_tripwire_rejected(reg, state):
    res = reg.validate_decision(
        decision("uniform_sampling_rate", 0.15, tripwire={"metric": "x"}), state
    )
    assert not res and any("tripwire missing" in e for e in res.errors)


# ── bounds and steps ──────────────────────────────────────────────────
def test_hard_range_enforced(reg, state):
    assert not reg.validate_decision(decision("uniform_sampling_rate", 0.9), state)
    assert not reg.validate_decision(decision("uniform_sampling_rate", 0.01), state)


def test_multiplicative_step_enforced(reg, state):
    # default 0.1, factor 1.5 → 0.15 ok, 0.2 too big
    assert reg.validate_decision(decision("uniform_sampling_rate", 0.15), state)
    res = reg.validate_decision(decision("uniform_sampling_rate", 0.2), state)
    assert not res and any("step too large" in e for e in res.errors)


def test_multiplicative_step_down_enforced(reg, state):
    state.current_values["adp_samp_failure_rate_max_over_mean"] = 200.0
    # factor 2 → floor 100
    assert reg.validate_decision(decision("adp_samp_failure_rate_max_over_mean", 100.0), state)
    assert not reg.validate_decision(decision("adp_samp_failure_rate_max_over_mean", 50.0), state)


def test_additive_step_enforced(reg, state):
    state.current_values["termination_threshold.anchor_pos"] = 0.30
    assert reg.validate_decision(decision("termination_threshold.anchor_pos", 0.25), state)
    res = reg.validate_decision(decision("termination_threshold.anchor_pos", 0.18), state)
    assert not res and any("step too large" in e for e in res.errors)


def test_notch_step_enforced(reg, state):
    state.current_values["bin_size"] = 50
    assert reg.validate_decision(decision("bin_size", 25), state)
    assert reg.validate_decision(decision("bin_size", 100), state)
    state.current_values["bin_size"] = 25
    res = reg.validate_decision(decision("bin_size", 100), state)
    assert not res and any("skips a notch" in e for e in res.errors)


def test_non_finite_and_non_numeric_rejected(reg, state):
    assert not reg.validate_decision(decision("desired_kl", float("nan")), state)
    assert not reg.validate_decision(decision("desired_kl", "0.01"), state)
    assert not reg.validate_decision(decision("desired_kl", True), state)


# ── cooldown ──────────────────────────────────────────────────────────
def test_cooldown_blocks_then_releases(reg, state):
    state.apply("uniform_sampling_rate", 0.15)  # changed at tick 10
    state.tick = 11
    res = reg.validate_decision(decision("uniform_sampling_rate", 0.2), state)
    assert not res and any("cooldown" in e for e in res.errors)
    state.tick = 12  # cooldown_ticks: 2 elapsed
    assert reg.validate_decision(decision("uniform_sampling_rate", 0.12), state)


def test_cooldown_is_per_knob(reg, state):
    state.apply("uniform_sampling_rate", 0.15)
    state.tick = 11
    assert reg.validate_decision(decision("desired_kl", 0.012), state)


# ── design-status gating ──────────────────────────────────────────────
def test_design_knob_rejected_by_default(reg, state):
    res = reg.validate_decision(decision("dr_push_scale", 0.9), state)
    assert not res and any("'design'" in e for e in res.errors)


def test_design_knob_allowed_in_replay(reg, state):
    assert reg.validate_decision(decision("dr_push_scale", 0.9), state, allow_design=True)


# ── warnings ──────────────────────────────────────────────────────────
def test_noop_value_warns(reg, state):
    res = reg.validate_decision(decision("desired_kl", 0.01), state)
    assert res.ok and any("prefer action 'none'" in w for w in res.warnings)


def test_restart_required_warns(reg, state):
    res = reg.validate_decision(decision("bin_size", 25), state)
    assert res.ok and any("restart" in w for w in res.warnings)


# ── malformed registry specs rejected ─────────────────────────────────
def test_bad_specs_rejected():
    base = {"family": "optimizer", "type": "float", "hard_range": [0, 1],
            "max_step": {"kind": "additive", "step": 0.1}, "cooldown_ticks": 1,
            "status": "available"}
    with pytest.raises(ValueError):
        KnobRegistry({"knobs": {"k": {**base, "hard_range": [1, 0]}}})
    with pytest.raises(ValueError):
        KnobRegistry({"knobs": {"k": {**base, "max_step": {"kind": "weird"}}}})
    with pytest.raises(ValueError):
        KnobRegistry({"knobs": {"k": {**base, "type": "choice", "choices": [1]}}})
    with pytest.raises(ValueError):
        KnobRegistry({"knobs": {"k": {**base, "max_step": {"kind": "multiplicative", "factor": 0.5}}}})
