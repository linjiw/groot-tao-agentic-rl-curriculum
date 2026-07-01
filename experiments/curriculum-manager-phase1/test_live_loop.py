# SPDX-License-Identifier: Apache-2.0
"""Phase-1 tests: the toy run is knob-responsive and the live loop keeps the
Phase-0 guardrails. The LLM arm is exercised manually (see RESULTS.md), not
in CI — these tests cover everything around it, including the output parser.
"""

import json

import pytest

from live_loop import LLMPolicy, LiveLoop, NullPolicy, BandStepperPolicy
from toy_tracking_run import ToyConfig, ToyTrackingRun


# ── toy run: determinism + record shapes ─────────────────────────────
def test_toy_run_deterministic():
    a = ToyTrackingRun(ToyConfig(seed=7)).advance(500, {})
    b = ToyTrackingRun(ToyConfig(seed=7)).advance(500, {})
    assert a["eval"] == b["eval"]
    assert a["sampler"][0]["failure_rate"] == b["sampler"][0]["failure_rate"]


def test_toy_run_record_shapes_match_digest_schema():
    out = ToyTrackingRun(ToyConfig(seed=0)).advance(250, {})
    ev = out["eval"][0]
    assert {"it", "success_rate", "heldout_success_rate", "failed_keys"} <= set(ev)
    assert out["train"] and "policy/approxkl_avg" in out["train"][0]
    assert out["sampler"] and isinstance(out["sampler"][0]["failure_rate"], list)


def test_heldout_motions_never_sampled():
    run = ToyTrackingRun(ToyConfig(seed=0))
    p = run._sampling_prob(0.1, 50.0)
    assert (p[run.heldout_mask] == 0).all()
    assert p.sum() == pytest.approx(1.0)


# ── toy run: knob responsiveness (the point of Phase 1) ──────────────
def _run_ticks(knobs, ticks=10, seed=0):
    run = ToyTrackingRun(ToyConfig(seed=seed))
    series = []
    for _ in range(ticks):
        out = run.advance(250, knobs)
        series.append(out["eval"][0]["heldout_success_rate"])
    return run, series


def test_uniform_floor_raises_sampler_entropy():
    import sys, os, importlib.util
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    spec = importlib.util.spec_from_file_location(
        "db", os.path.join(repo, "skills/agentic/sonic-run-digest/digest_builder.py"))
    db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db)

    def entropy_at(uniform):
        run = ToyTrackingRun(ToyConfig(seed=3))
        out = run.advance(2000, {"uniform_sampling_rate": uniform})
        p = run._sampling_prob(uniform, 50.0)
        train = p[~run.heldout_mask]
        return db.normalized_entropy(train.tolist())

    assert entropy_at(0.5) > entropy_at(0.05)


def test_lower_threshold_trains_harder():
    """Tighter termination (more pressure) → faster skill growth."""
    tight_run, _ = _run_ticks({"termination_threshold.anchor_pos": 0.15}, ticks=12)
    loose_run, _ = _run_ticks({"termination_threshold.anchor_pos": 0.50}, ticks=12)
    assert tight_run.skill.mean() > loose_run.skill.mean()


def test_heldout_improves_via_spillover_only():
    run, series = _run_ticks({}, ticks=14)
    assert series[-1] > series[0]           # generalization spillover works
    train_gain = run.skill[~run.heldout_mask].mean()
    held_gain = run.skill[run.heldout_mask].mean()
    assert train_gain > held_gain            # but held-out lags training


# ── live loop keeps the guardrails ───────────────────────────────────
def test_null_policy_never_acts():
    loop = LiveLoop(NullPolicy(), ToyTrackingRun(ToyConfig(seed=0)))
    s = loop.run_ticks(8)
    assert s["decisions_applied"] == 0 and s["rejected"] == 0


def test_band_policy_closed_loop_applies_and_scores():
    loop = LiveLoop(BandStepperPolicy(t_high=0.45, sustain=2),
                    ToyTrackingRun(ToyConfig(seed=0)))
    s = loop.run_ticks(12)
    assert s["decisions_applied"] >= 1
    assert s["rejected"] == 0
    outcomes = {a["outcome"] for a in s["applied_knobs"]}
    assert outcomes <= {"met", "regressed", "pending", "failed_rolled_back"}


def test_live_loop_rejects_rogue_policy():
    class Rogue:
        def propose(self, digest, state, registry):
            return {"action": "set", "knob": "eval_thresholds", "value": 0,
                    "rationale": "r", "expected_effect": "e",
                    "tripwire": {"metric": "m", "drop_pct": 5, "evals": 3}}

    loop = LiveLoop(Rogue(), ToyTrackingRun(ToyConfig(seed=0)))
    s = loop.run_ticks(4)
    assert s["decisions_applied"] == 0 and s["rejected"] == 4


def test_journal_serializable():
    loop = LiveLoop(BandStepperPolicy(), ToyTrackingRun(ToyConfig(seed=0)))
    loop.run_ticks(6)
    json.dumps(loop.journal)


# ── LLM output parser (fail-safe) ────────────────────────────────────
def test_parse_fenced_json():
    d = LLMPolicy._parse('text\n```json\n{"action": "none", "reason": "x"}\n```\n')
    assert d == {"action": "none", "reason": "x"}


def test_parse_bare_json():
    d = LLMPolicy._parse('{"action": "set", "knob": "desired_kl", "value": 0.012}')
    assert d["knob"] == "desired_kl"


def test_parse_garbage_degrades_to_none():
    d = LLMPolicy._parse("I think we should probably tighten the threshold")
    assert d["action"] == "none" and "unparseable" in d["reason"]


def test_parse_json_without_action_degrades_to_none():
    d = LLMPolicy._parse('{"knob": "desired_kl"}')
    assert d["action"] == "none"
