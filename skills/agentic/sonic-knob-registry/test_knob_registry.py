# SPDX-License-Identifier: Apache-2.0
"""Tests for the knob registry + decision validator (design doc 08 §3/§5)."""

import pytest

from knob_registry import ConfigDriftError, KnobRegistry, RunState, load_registry


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


# ── registry-level pending gate (doc 08 §11 amendment 2) ─────────────
def test_pending_gate_rejects_any_set(reg, state):
    state.arm_pending("uniform_sampling_rate")
    # even a perfectly-formed change to a DIFFERENT knob is rejected
    res = reg.validate_decision(decision("desired_kl", 0.012), state)
    assert not res and any("pending change" in e and "uniform_sampling_rate" in e
                           for e in res.errors)


def test_pending_gate_allows_none(reg, state):
    state.arm_pending("uniform_sampling_rate")
    assert reg.validate_decision({"action": "none"}, state)


def test_pending_gate_clears(reg, state):
    state.arm_pending("uniform_sampling_rate")
    state.clear_pending()
    assert state.pending is None
    assert reg.validate_decision(decision("desired_kl", 0.012), state)


def test_pending_gate_same_knob_also_rejected(reg, state):
    state.arm_pending("desired_kl")
    res = reg.validate_decision(decision("desired_kl", 0.012), state)
    assert not res


def test_legacy_state_without_pending_still_validates(reg):
    """Defense in depth must not break callers holding an old-shaped state."""
    class LegacyState:
        tick = 10
        current_values = {}
        last_changed_tick = {}
    assert reg.validate_decision(decision("desired_kl", 0.012), LegacyState())


# ── hard rule 4: Family-B needs a held-out metric (digest-supplied) ──
def _digest_with_heldout(last, trend):
    return {"eval": {"heldout_success_rate": {"last": last, "trend": trend}}}


def test_family_b_rejected_without_heldout(reg, state):
    state.current_values["termination_threshold.anchor_pos"] = 0.30
    d = decision("termination_threshold.anchor_pos", 0.25)
    # digest with no eval section at all
    res = reg.validate_decision(d, state, digest={"eval": None})
    assert not res and any("hard rule 4" in e for e in res.errors)
    # held-out present but null / unknown trend
    for bad in (_digest_with_heldout(None, "flat"),
                _digest_with_heldout(0.6, "unknown")):
        assert not reg.validate_decision(d, state, digest=bad)


def test_family_b_allowed_with_heldout(reg, state):
    state.current_values["termination_threshold.anchor_pos"] = 0.30
    d = decision("termination_threshold.anchor_pos", 0.25)
    assert reg.validate_decision(d, state, digest=_digest_with_heldout(0.42, "flat"))


def test_non_family_b_unaffected_by_missing_heldout(reg, state):
    # Family-A knob: hard rule 4 machine-check does not apply
    d = decision("uniform_sampling_rate", 0.15)
    assert reg.validate_decision(d, state, digest={"eval": None})


def test_no_digest_skips_heldout_check(reg, state):
    """Callers without a held-out stream (Phase-2 smoke) omit the digest;
    the gate stays behavioral there (scope-noted), not machine-enforced."""
    state.current_values["termination_threshold.anchor_pos"] = 0.30
    assert reg.validate_decision(decision("termination_threshold.anchor_pos", 0.25), state)


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


# ── verify against a run's resolved config.yaml (doc 08 §11 amendment 8) ─
_PATHS = {
    "termination_threshold.foot_pos_xyz":
        "manager_env.terminations.foot_pos_xyz.params.threshold",
    "uniform_sampling_rate":
        "manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.uniform_sampling_rate",
}


def _fake_cfg(foot=0.2, uniform=0.1):
    return {"manager_env": {
        "terminations": {"foot_pos_xyz": {"params": {"threshold": foot}}},
        "commands": {"motion": {"motion_lib_cfg": {
            "adaptive_sampling": {"uniform_sampling_rate": uniform}}}}}}


def test_verify_matching_beliefs_pass(reg, state):
    state.current_values["termination_threshold.foot_pos_xyz"] = 0.2
    state.current_values["uniform_sampling_rate"] = 0.1
    res = reg.verify_against_config(state, _fake_cfg(), _PATHS)
    assert res and res.ok and not res.drifts and not res.missing
    assert set(res.checked) == set(_PATHS)
    res.raise_on_drift()  # no-op when ok


def test_verify_divergence_flags_and_raises(reg, state):
    # THE v2 defect shape: registry default (0.35) believed, but the run's
    # resolved config actually has the stock strict value (0.2)
    state.current_values["termination_threshold.foot_pos_xyz"] = 0.35
    res = reg.verify_against_config(state, _fake_cfg(foot=0.2), _PATHS)
    assert not res
    drift = res.drifts["termination_threshold.foot_pos_xyz"]
    assert drift == {"believed": 0.35, "resolved": 0.2}
    with pytest.raises(ConfigDriftError, match="amendment 8"):
        res.raise_on_drift()
    # the wrong belief is NOT silently rewritten — the caller must refuse
    assert state.current_values["termination_threshold.foot_pos_xyz"] == 0.35


def test_verify_adopts_unseeded_beliefs_from_ground_truth(reg, state):
    # no beliefs yet: values are seeded from the config, never from the
    # registry.yaml defaults (the structural fix for the seeding defect)
    res = reg.verify_against_config(state, _fake_cfg(foot=0.2), _PATHS)
    assert res.ok
    assert res.adopted["termination_threshold.foot_pos_xyz"] == 0.2
    assert state.current_values["termination_threshold.foot_pos_xyz"] == 0.2
    assert reg.current_of("termination_threshold.foot_pos_xyz", state) == 0.2


def test_verify_no_adopt_compares_registry_default(reg, state):
    # adopt_unseeded=False: the registry default (0.35) is what current_of()
    # would answer, so default-vs-config drift must surface, not hide
    res = reg.verify_against_config(state, _fake_cfg(foot=0.2), _PATHS,
                                    adopt_unseeded=False)
    assert not res
    assert res.drifts["termination_threshold.foot_pos_xyz"]["believed"] == 0.35
    assert "termination_threshold.foot_pos_xyz" not in state.current_values


def test_verify_missing_path_flagged_not_fatal(reg, state):
    state.current_values["termination_threshold.foot_pos_xyz"] = 0.2
    cfg = _fake_cfg()
    del cfg["manager_env"]["commands"]  # uniform_sampling_rate path gone
    res = reg.verify_against_config(state, cfg, _PATHS)
    assert res.ok  # missing is a wiring gap, not drift
    assert res.missing == ["uniform_sampling_rate"]
    res.raise_on_drift()


def test_verify_numeric_types_compare_as_numbers(reg, state):
    # yaml often deserializes 50 as int while the belief is 50.0
    state.current_values["uniform_sampling_rate"] = 0.1
    cfg = _fake_cfg(uniform=0.1)
    state.current_values["termination_threshold.foot_pos_xyz"] = 0.2
    res = reg.verify_against_config(state, cfg, _PATHS)
    assert res.ok


def test_verify_ignores_paths_for_unknown_knobs(reg, state):
    paths = dict(_PATHS, not_a_knob="some.random.path")
    state.current_values["termination_threshold.foot_pos_xyz"] = 0.2
    state.current_values["uniform_sampling_rate"] = 0.1
    res = reg.verify_against_config(state, _fake_cfg(), paths)
    assert res.ok and "not_a_knob" not in res.checked
