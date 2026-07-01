# SPDX-License-Identifier: Apache-2.0
"""Smoke-driver tests with a fake adapter (no docker): composition wiring,
band policy behavior, tripwire rollback, control-arm inertness.
"""

import json

import pytest

from smoke_driver import SmokeDriver, TrainSideBandPolicy, job_adapter


class FakeAdapter:
    """Scripted segments: each entry is (len_mean, rew_mean)."""

    def __init__(self, script):
        self.script = list(script)
        self.launched = []  # (name, knobs, checkpoint_in)
        self.i = 0

    def launch_segment(self, name, iterations, knobs, checkpoint_in=None):
        self.launched.append((name, dict(knobs), checkpoint_in))
        return job_adapter.Segment(name=name, iterations=iterations,
                                   knobs=dict(knobs), checkpoint_in=checkpoint_in,
                                   status="running")

    def wait(self, seg, poll_s=0, timeout_s=0):
        seg.status = "done"
        seg.snapshot = f"/fake/{seg.name}/snapshot.pt"
        return seg

    def parse_segment(self, seg):
        ln, rew = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        train = [{"it": k + 1, "Episode/len_mean": ln, "Episode/rew_mean": rew,
                  "loss/entropy_avg": -45.0,
                  # foot_pos_xyz is the binding termination axis (like the real runs)
                  "Episode_Termination/foot_pos_xyz": 0.56,
                  "Episode_Termination/ee_body_pos": 0.42,
                  "Episode_Termination/anchor_pos": 0.0} for k in range(3)]
        sampler = [{"it": 2, "failure_rate_mean": 2.0, "effective_num_bins": 60.0}]
        return job_adapter.ParsedLog(train=train, sampler=sampler)


def run_driver(script, arm="manager", segments=None, **policy_kw):
    fake = FakeAdapter(script)
    driver = SmokeDriver(TrainSideBandPolicy(**policy_kw), adapter=fake, arm=arm)
    summary = driver.run(segments or len(script))
    return driver, fake, summary


# ── control arm never acts ───────────────────────────────────────────
def test_control_arm_inert():
    driver, fake, s = run_driver([(10, 1.0)] * 4, arm="control")
    assert s["decisions_applied"] == 0 and s["rollbacks"] == 0
    # all launches carry identical (empty) knobs
    assert all(k == {} for _, k, _ in fake.launched)


# ── band policy: loosen on sustained short episodes ──────────────────
def test_loosens_on_sustained_short_episodes():
    driver, fake, s = run_driver([(10, 1.0)] * 4, len_low=20, sustain=2)
    assert s["decisions_applied"] == 1
    applied = next(e for e in driver.journal if e.get("applied"))
    d = applied["decision"]
    assert d["knob"] == "termination_threshold.foot_pos_xyz"
    assert d["value"] == pytest.approx(0.40)  # foot default 0.35 + notch, loosen
    # next launch carries the new knob
    later_knobs = [k for _, k, _ in fake.launched[applied["tick"]:]]
    assert all(k.get("termination_threshold.foot_pos_xyz") == 0.40 for k in later_knobs)


def test_sustain_prevents_single_segment_action():
    driver, _, s = run_driver([(10, 1.0), (30, 1.0), (10, 1.0), (30, 1.0)],
                              len_low=20, sustain=2)
    assert s["decisions_applied"] == 0  # alternating: never 2 consecutive lows


def test_no_action_inside_band():
    driver, _, s = run_driver([(50, 1.0)] * 4, len_low=20, sustain=2)
    assert s["decisions_applied"] == 0
    reasons = [e["decision"]["reason"] for e in driver.journal]
    assert all("band" in r or "sustain" in r for r in reasons)


# ── tripwire: reward collapse after an applied change rolls back ─────
def test_tripwire_rolls_back_on_reward_collapse():
    # 2 short-episode segments -> loosen applied after seg2;
    # then reward collapses for 2 consecutive segments -> rollback
    script = [(10, 1.0), (10, 1.0), (10, 0.1), (10, 0.1), (10, 1.0)]
    driver, fake, s = run_driver(script, len_low=20, sustain=2, segments=5)
    assert s["decisions_applied"] == 1
    assert s["rollbacks"] == 1
    rb = next(e for e in driver.journal if e.get("event") == "rollback")
    assert rb["restored"] == {"termination_threshold.foot_pos_xyz": 0.35}
    # the segment AFTER rollback relaunches from the pre-change snapshot
    idx = driver.journal.index(rb)
    next_launch = fake.launched[idx + 1] if idx + 1 < len(fake.launched) else None
    if next_launch:
        assert next_launch[1].get("termination_threshold.foot_pos_xyz", 0.35) == 0.35


def test_reward_recovery_resets_tripwire():
    # collapse for 1 segment then recovery: no rollback
    script = [(10, 1.0), (10, 1.0), (10, 0.1), (10, 1.0), (10, 1.0)]
    driver, _, s = run_driver(script, len_low=20, sustain=2, segments=5)
    assert s["rollbacks"] == 0


# ── registry still guards the policy ─────────────────────────────────
def test_hard_range_ceiling_stops_loosening():
    # policy at ceiling: driver must not push beyond hard range
    driver = SmokeDriver(TrainSideBandPolicy(len_low=20, sustain=2),
                         adapter=FakeAdapter([(10, 1.0)] * 8), arm="manager")
    driver.state.apply("termination_threshold.foot_pos_xyz", 0.50)  # at hard max
    driver.knobs["termination_threshold.foot_pos_xyz"] = 0.50
    s = driver.run(4)
    assert s["decisions_applied"] == 0  # policy checks range; nothing applied


def test_journal_serializable_and_labeled():
    driver, _, _ = run_driver([(10, 1.0)] * 3)
    text = json.dumps(driver.journal)
    assert "training-side" in text  # honest labeling travels in rationales


# ── pending-decision gate (review findings 1-2) ──────────────────────
def test_pending_gate_blocks_second_change():
    """While a change is under tripwire watch, no new change is applied —
    even if the policy would propose one (sustained short episodes)."""
    driver, fake, s = run_driver([(10, 1.0)] * 6, len_low=20, sustain=2, segments=6)
    assert s["decisions_applied"] >= 1
    applied = [e for e in driver.journal if e.get("applied")]
    ticks = [e["tick"] for e in applied]
    # tripwire watch is evals=2 clean segments: consecutive applied ticks
    # must be separated by at least that window
    assert all(b - a >= 3 for a, b in zip(ticks, ticks[1:])), ticks
    # journal shows explicit pending-gate holds
    reasons = [e["decision"].get("reason", "") for e in driver.journal]
    assert any("under tripwire watch" in r for r in reasons)


def test_survived_change_scored_survived():
    driver, _, s = run_driver([(10, 1.0)] * 6, len_low=20, sustain=2, segments=6)
    applied = [e for e in driver.journal if e.get("applied")]
    assert applied[0]["outcome"] == "survived"  # 2 clean segments


def test_rolled_back_decision_marked_failed():
    script = [(10, 1.0), (10, 1.0), (10, 0.1), (10, 0.1), (10, 1.0)]
    driver, _, s = run_driver(script, len_low=20, sustain=2, segments=5)
    origin = next(e for e in driver.journal if e.get("applied"))
    assert origin["outcome"] == "failed_rolled_back"


def test_no_reward_baseline_refuses_change():
    class NoRewAdapter(FakeAdapter):
        def parse_segment(self, seg):
            p = super().parse_segment(seg)
            for r in p.train:
                r.pop("Episode/rew_mean", None)
            return p

    fake = NoRewAdapter([(10, 1.0)] * 4)
    driver = SmokeDriver(TrainSideBandPolicy(len_low=20, sustain=2),
                         adapter=fake, arm="manager")
    s = driver.run(4)
    assert s["decisions_applied"] == 0
    errs = [e["validation"]["errors"] for e in driver.journal if e.get("validation")]
    assert any("no reward baseline" in str(e) for e in errs)


def test_binding_fraction_in_rationale():
    driver, _, _ = run_driver([(10, 1.0)] * 4, len_low=20, sustain=2)
    applied = next(e for e in driver.journal if e.get("applied"))
    assert "0.56" in applied["decision"]["rationale"]  # windowed mean cited
