# SPDX-License-Identifier: Apache-2.0
"""Phase-0 acceptance tests for the Curriculum-Manager decision loop
(design doc 08 Phase 0): hysteresis holds in thrash, tighten fires on
sustained success, sampler knob (not threshold) moves on a concentrated
plateau, tripwire rolls back a regression, and the journal is auditable.
"""

import json

import pytest

from replay_harness import ArmedTripwire, BandStepperPolicy, ReplayHarness
from scenarios import SCENARIOS, healthy, plateau, regression, thrash


def make_harness(**policy_kw):
    return ReplayHarness(BandStepperPolicy(**policy_kw))


# ── healthy: mostly nothing, then a tighten on sustained success ─────
def test_healthy_tightens_after_sustained_high():
    h = make_harness()
    h.run(healthy(n_ticks=12))
    s = h.summary()
    applied = [e for e in h.journal if e.get("applied")]
    assert applied, "expected at least one tighten in a healthy run"
    first = applied[0]
    assert first["knob"] == "termination_threshold.anchor_pos"
    assert first["decision"]["value"] < first["prev_value"], "must tighten (decrease)"
    assert s["rollbacks"] == 0
    # sustained-evidence rule: healthy() crosses t_high=0.82 (heldout) around
    # tick 8; with sustain=3 nothing may fire before tick 3 regardless
    assert first["tick"] >= 3


def test_healthy_never_rejected_by_validator():
    h = make_harness()
    h.run(healthy(n_ticks=12))
    assert h.summary()["rejected"] == 0, (
        "the playbook policy must only propose registry-legal steps"
    )


# ── thrash: hysteresis + sustained-trend rules → zero decisions ──────
def test_thrash_produces_no_decisions():
    h = make_harness()
    h.run(thrash(n_ticks=12))
    s = h.summary()
    assert s["decisions_applied"] == 0, (
        f"thrash scenario must yield no interventions, got {s}"
    )
    assert s["rollbacks"] == 0


# ── plateau: sampler knob moves, threshold does not ──────────────────
def test_plateau_moves_sampler_floor_not_threshold():
    h = make_harness()
    h.run(plateau(n_ticks=12))
    applied = [e for e in h.journal if e.get("applied")]
    assert applied, "expected a sampler-health intervention on the plateau"
    knobs = {e["knob"] for e in applied}
    assert "uniform_sampling_rate" in knobs
    assert "termination_threshold.anchor_pos" not in knobs, (
        "a stuck run must never tighten thresholds"
    )
    # bounded step: first move is exactly one multiplicative step (x1.5)
    first = next(e for e in applied if e["knob"] == "uniform_sampling_rate")
    assert first["decision"]["value"] == pytest.approx(first["prev_value"] * 1.5, rel=1e-6)


def test_plateau_respects_cooldown_between_sampler_moves():
    h = make_harness()
    h.run(plateau(n_ticks=12))
    ticks = [e["tick"] for e in h.journal
             if e.get("applied") and e["knob"] == "uniform_sampling_rate"]
    assert all(b - a >= 2 for a, b in zip(ticks, ticks[1:])), (
        f"cooldown_ticks=2 violated: {ticks}"
    )


# ── regression: tripwire fires and rolls back ────────────────────────
def test_regression_tripwire_rolls_back():
    # Policy tightens during the healthy prefix (sustain=2 to get a change
    # in before the collapse at tick 5), then held-out collapses.
    h = make_harness(t_high=0.72, sustain=2)
    h.run(regression(n_ticks=12, collapse_after_tick=5))
    s = h.summary()
    assert s["decisions_applied"] >= 1, "need an applied change for the tripwire to guard"
    assert s["rollbacks"] >= 1, f"tripwire must fire on the collapse: {s}"
    rb = h.rollbacks[0]
    # the originating decision (the one the tripwire guarded — the LAST
    # change to that knob) is marked failed and its pre-change value restored
    origin = next(
        e for e in h.journal
        if e.get("applied") and e.get("outcome") == "failed_rolled_back"
    )
    assert origin["knob"] == rb["knob"]
    assert rb["restored_value"] == origin["prev_value"]


def test_tripwire_requires_sustained_breach():
    tw = ArmedTripwire(knob="k", prev_value=1, baseline=0.8, drop_pct=5, evals=3)
    assert not tw.check(0.5)   # breach 1
    assert not tw.check(0.5)   # breach 2
    assert tw.check(0.5)       # breach 3 → fire
    tw2 = ArmedTripwire(knob="k", prev_value=1, baseline=0.8, drop_pct=5, evals=3)
    assert not tw2.check(0.5)
    assert not tw2.check(0.79)  # recovers above 0.76 → counter resets
    assert not tw2.check(0.5)
    assert not tw2.check(0.5)


# ── protected-metric rule: no held-out signal → no action ────────────
def test_no_heldout_metric_means_no_action():
    h = make_harness()
    for records in healthy(n_ticks=8):
        for r in records.eval:
            r.pop("heldout_success_rate", None)
        h.tick(records)
    assert h.summary()["decisions_applied"] == 0


# ── the harness distrusts the policy: illegal proposals are rejected ─
class RoguePolicy:
    """Proposes an out-of-registry knob, then an oversized step."""

    def __init__(self):
        self.calls = 0

    def propose(self, digest, state, registry):
        self.calls += 1
        tw = {"metric": "heldout_success_rate", "drop_pct": 5, "evals": 3}
        if self.calls % 2:
            return {"action": "set", "knob": "eval_thresholds", "value": 0.5,
                    "rationale": "r", "expected_effect": "e", "tripwire": tw}
        return {"action": "set", "knob": "uniform_sampling_rate", "value": 0.5,
                "rationale": "r", "expected_effect": "e", "tripwire": tw}


def test_rogue_policy_is_fully_rejected():
    h = ReplayHarness(RoguePolicy())
    h.run(healthy(n_ticks=6))
    s = h.summary()
    assert s["decisions_applied"] == 0
    assert s["rejected"] == 6
    # every rejection carries auditable errors
    for e in h.journal:
        if e.get("validation"):
            assert e["validation"]["errors"]


# ── journal auditability + determinism ───────────────────────────────
def test_journal_is_json_serializable_and_complete():
    h = make_harness()
    h.run(healthy(n_ticks=12))
    text = json.dumps(h.journal)
    assert len(h.journal) == 12
    for e in h.journal:
        assert "tick" in e and ("decision" in e or e.get("action") == "rollback")
    assert "rationale" in text


def test_scenarios_are_deterministic():
    a = [r.eval[0]["success_rate"] for r in healthy(n_ticks=6)]
    b = [r.eval[0]["success_rate"] for r in healthy(n_ticks=6)]
    assert a == b


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_all_scenarios_run_clean(name):
    h = make_harness()
    h.run(SCENARIOS[name](n_ticks=10))
    json.dumps(h.summary())
