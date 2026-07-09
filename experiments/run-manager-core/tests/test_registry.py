# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the engine-agnostic knob registry + decision validator.

Migrated from skills/agentic/sonic-knob-registry/test_knob_registry.py;
every SONIC-specific knob/value is replaced by the injected spec in
conftest.py (the core ships no default action space)."""

import pytest

from core.registry import (
    ConfigDriftError,
    KnobRegistry,
    RunState,
    load_registry,
)


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


# ── engine-agnostic construction (the de-SONIC-ification seam) ────────
def test_no_default_registry_path():
    with pytest.raises(TypeError):
        KnobRegistry.load()  # path is REQUIRED — no SONIC yaml default


def test_load_registry_requires_exactly_one_source(spec):
    with pytest.raises(ValueError):
        load_registry()
    with pytest.raises(ValueError):
        load_registry(path="x.yaml", spec=spec)
    assert load_registry(spec=spec).knobs


def test_load_registry_from_yaml_path(tmp_path, spec):
    import yaml
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump(spec))
    reg = load_registry(path=str(p))
    assert set(reg.knobs) == set(spec["knobs"])


def test_core_never_imports_engine_modules():
    import core, core.registry, core.digest, core.journal, core.protocols  # noqa
    import sys
    assert not any(m.startswith("gear_sonic") or "job_adapter" in m
                   for m in sys.modules)


def test_heldout_gated_families_injectable(spec, state):
    # gate moved to a DIFFERENT family: schedule knob passes without
    # held-out evidence, optimizer knob is now the gated one
    reg = KnobRegistry(spec, heldout_gated_families=("optimizer",))
    state.current_values["threshold.a"] = 0.30
    assert reg.validate_decision(decision("threshold.a", 0.25), state,
                                 digest={"eval": None})
    res = reg.validate_decision(decision("kl_target", 0.012), state,
                                digest={"eval": None})
    assert not res and any("hard rule 4" in e for e in res.errors)


# ── registry integrity ────────────────────────────────────────────────
def test_registry_loads_and_has_three_families(reg):
    families = {k["family"] for k in reg.knobs.values()}
    assert families == {"data_curriculum", "schedule", "optimizer"}


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
    res = reg.validate_decision(decision(["mix_rate", "kl_target"], 0.2), state)
    assert not res and "atomic" in res.errors[0]


def test_missing_rationale_or_tripwire_rejected(reg, state):
    res = reg.validate_decision(decision("mix_rate", 0.15, rationale=""), state)
    assert not res
    res = reg.validate_decision(decision("mix_rate", 0.15, tripwire=None), state)
    assert not res


def test_malformed_tripwire_rejected(reg, state):
    res = reg.validate_decision(
        decision("mix_rate", 0.15, tripwire={"metric": "x"}), state
    )
    assert not res and any("tripwire missing" in e for e in res.errors)


# ── bounds and steps ──────────────────────────────────────────────────
def test_hard_range_enforced(reg, state):
    assert not reg.validate_decision(decision("mix_rate", 0.9), state)
    assert not reg.validate_decision(decision("mix_rate", 0.01), state)


def test_multiplicative_step_enforced(reg, state):
    # default 0.1, factor 1.5 → 0.15 ok, 0.2 too big
    assert reg.validate_decision(decision("mix_rate", 0.15), state)
    res = reg.validate_decision(decision("mix_rate", 0.2), state)
    assert not res and any("step too large" in e for e in res.errors)


def test_multiplicative_step_down_enforced(reg, state):
    state.current_values["cap_ratio"] = 200.0
    # factor 2 → floor 100
    assert reg.validate_decision(decision("cap_ratio", 100.0), state)
    assert not reg.validate_decision(decision("cap_ratio", 50.0), state)


def test_additive_step_enforced(reg, state):
    state.current_values["threshold.a"] = 0.30
    assert reg.validate_decision(decision("threshold.a", 0.25), state)
    res = reg.validate_decision(decision("threshold.a", 0.18), state)
    assert not res and any("step too large" in e for e in res.errors)


def test_notch_step_enforced(reg, state):
    state.current_values["bin_size"] = 50
    assert reg.validate_decision(decision("bin_size", 25), state)
    assert reg.validate_decision(decision("bin_size", 100), state)
    state.current_values["bin_size"] = 25
    res = reg.validate_decision(decision("bin_size", 100), state)
    assert not res and any("skips a notch" in e for e in res.errors)


def test_non_finite_and_non_numeric_rejected(reg, state):
    assert not reg.validate_decision(decision("kl_target", float("nan")), state)
    assert not reg.validate_decision(decision("kl_target", "0.01"), state)
    assert not reg.validate_decision(decision("kl_target", True), state)


# ── cooldown ──────────────────────────────────────────────────────────
def test_cooldown_blocks_then_releases(reg, state):
    state.apply("mix_rate", 0.15)  # changed at tick 10
    state.tick = 11
    res = reg.validate_decision(decision("mix_rate", 0.2), state)
    assert not res and any("cooldown" in e for e in res.errors)
    state.tick = 12  # cooldown_ticks: 2 elapsed
    assert reg.validate_decision(decision("mix_rate", 0.12), state)


def test_cooldown_is_per_knob(reg, state):
    state.apply("mix_rate", 0.15)
    state.tick = 11
    assert reg.validate_decision(decision("kl_target", 0.012), state)


# ── design-status gating ──────────────────────────────────────────────
def test_design_knob_rejected_by_default(reg, state):
    res = reg.validate_decision(decision("push_scale", 0.9), state)
    assert not res and any("'design'" in e for e in res.errors)


def test_design_knob_allowed_in_replay(reg, state):
    assert reg.validate_decision(decision("push_scale", 0.9), state,
                                 allow_design=True)


# ── warnings ──────────────────────────────────────────────────────────
def test_noop_value_warns(reg, state):
    res = reg.validate_decision(decision("kl_target", 0.01), state)
    assert res.ok and any("prefer action 'none'" in w for w in res.warnings)


def test_restart_required_warns(reg, state):
    res = reg.validate_decision(decision("bin_size", 25), state)
    assert res.ok and any("restart" in w for w in res.warnings)


# ── registry-level pending gate ───────────────────────────────────────
def test_pending_gate_rejects_any_set(reg, state):
    state.arm_pending("mix_rate")
    res = reg.validate_decision(decision("kl_target", 0.012), state)
    assert not res and any("pending change" in e and "mix_rate" in e
                           for e in res.errors)


def test_pending_gate_allows_none(reg, state):
    state.arm_pending("mix_rate")
    assert reg.validate_decision({"action": "none"}, state)


def test_pending_gate_clears(reg, state):
    state.arm_pending("mix_rate")
    state.clear_pending()
    assert state.pending is None
    assert reg.validate_decision(decision("kl_target", 0.012), state)


def test_legacy_state_without_pending_still_validates(reg):
    class LegacyState:
        tick = 10
        current_values = {}
        last_changed_tick = {}
    assert reg.validate_decision(decision("kl_target", 0.012), LegacyState())


# ── hard rule 4: gated family needs a held-out metric ────────────────
def _digest_with_heldout(last, trend):
    return {"eval": {"heldout_success_rate": {"last": last, "trend": trend}}}


def test_family_b_rejected_without_heldout(reg, state):
    state.current_values["threshold.a"] = 0.30
    d = decision("threshold.a", 0.25)
    res = reg.validate_decision(d, state, digest={"eval": None})
    assert not res and any("hard rule 4" in e for e in res.errors)
    for bad in (_digest_with_heldout(None, "flat"),
                _digest_with_heldout(0.6, "unknown")):
        assert not reg.validate_decision(d, state, digest=bad)


def test_family_b_allowed_with_heldout(reg, state):
    state.current_values["threshold.a"] = 0.30
    d = decision("threshold.a", 0.25)
    assert reg.validate_decision(d, state,
                                 digest=_digest_with_heldout(0.42, "flat"))


def test_non_family_b_unaffected_by_missing_heldout(reg, state):
    d = decision("mix_rate", 0.15)
    assert reg.validate_decision(d, state, digest={"eval": None})


def test_no_digest_skips_heldout_check(reg, state):
    state.current_values["threshold.a"] = 0.30
    assert reg.validate_decision(decision("threshold.a", 0.25), state)


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
        KnobRegistry({"knobs": {"k": {**base, "max_step":
                                      {"kind": "multiplicative", "factor": 0.5}}}})


# ── verify against a run's resolved config ────────────────────────────
_PATHS = {
    "threshold.a": "env.terminations.a.params.threshold",
    "mix_rate": "env.sampling.mix_rate",
}


def _fake_cfg(thr=0.2, mix=0.1):
    return {"env": {
        "terminations": {"a": {"params": {"threshold": thr}}},
        "sampling": {"mix_rate": mix}}}


def test_verify_matching_beliefs_pass(reg, state):
    state.current_values["threshold.a"] = 0.2
    state.current_values["mix_rate"] = 0.1
    res = reg.verify_against_config(state, _fake_cfg(), _PATHS)
    assert res and res.ok and not res.drifts and not res.missing
    assert set(res.checked) == set(_PATHS)
    res.raise_on_drift()  # no-op when ok


def test_verify_divergence_flags_and_raises(reg, state):
    state.current_values["threshold.a"] = 0.35
    res = reg.verify_against_config(state, _fake_cfg(thr=0.2), _PATHS)
    assert not res
    drift = res.drifts["threshold.a"]
    assert drift == {"believed": 0.35, "resolved": 0.2}
    with pytest.raises(ConfigDriftError, match="amendment 8"):
        res.raise_on_drift()
    # the wrong belief is NOT silently rewritten — the caller must refuse
    assert state.current_values["threshold.a"] == 0.35


def test_verify_adopts_unseeded_beliefs_from_ground_truth(reg, state):
    res = reg.verify_against_config(state, _fake_cfg(thr=0.2), _PATHS)
    assert res.ok
    assert res.adopted["threshold.a"] == 0.2
    assert state.current_values["threshold.a"] == 0.2
    assert reg.current_of("threshold.a", state) == 0.2


def test_verify_no_adopt_compares_registry_default(reg, state):
    res = reg.verify_against_config(state, _fake_cfg(thr=0.2), _PATHS,
                                    adopt_unseeded=False)
    assert not res
    assert res.drifts["threshold.a"]["believed"] == 0.35
    assert "threshold.a" not in state.current_values


def test_verify_missing_path_flagged_not_fatal(reg, state):
    state.current_values["threshold.a"] = 0.2
    cfg = _fake_cfg()
    del cfg["env"]["sampling"]  # mix_rate path gone
    res = reg.verify_against_config(state, cfg, _PATHS)
    assert res.ok  # missing is a wiring gap, not drift
    assert res.missing == ["mix_rate"]
    res.raise_on_drift()


def test_verify_numeric_types_compare_as_numbers(reg, state):
    state.current_values["mix_rate"] = 0.1
    state.current_values["threshold.a"] = 0.2
    res = reg.verify_against_config(state, _fake_cfg(), _PATHS)
    assert res.ok


def test_verify_ignores_paths_for_unknown_knobs(reg, state):
    paths = dict(_PATHS, not_a_knob="some.random.path")
    state.current_values["threshold.a"] = 0.2
    state.current_values["mix_rate"] = 0.1
    res = reg.verify_against_config(state, _fake_cfg(), paths)
    assert res.ok and "not_a_knob" not in res.checked
