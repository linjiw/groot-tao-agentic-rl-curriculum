# SPDX-License-Identifier: Apache-2.0
"""Smoke-driver tests with a fake adapter (no docker): composition wiring,
band policy behavior, tripwire rollback, control-arm inertness.
"""

import json

import pytest

from smoke_driver import (DiskSpaceError, ScriptedPolicy, SmokeDriver,
                          TrainSideBandPolicy, V4_MANAGER_LADDER,
                          job_adapter, knob_registry,
                          purge_intermediate_checkpoints)


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
    # 2 clean segments -> survived; lengths stayed at 10 < len_low, so the
    # stated effect ("episodes lengthen") was NOT observed
    assert applied[0]["outcome"] == "survived_effect_not_observed"
    assert applied[0]["effect"]["observed"] == 10


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
    assert any("baseline available" in str(e) for e in errs)


def test_binding_fraction_in_rationale():
    driver, _, _ = run_driver([(10, 1.0)] * 4, len_low=20, sustain=2)
    applied = next(e for e in driver.journal if e.get("applied"))
    assert "0.56" in applied["decision"]["rationale"]  # windowed mean cited


# ── v3: per-segment eval + eval-side tripwire ────────────────────────
class FakeEvalAdapter(FakeAdapter):
    """FakeAdapter + scripted eval_segment: eval_script[i] = progress_rate
    for the i-th eval pass (None = eval failure)."""

    def __init__(self, script, eval_script):
        super().__init__(script)
        self.eval_script = list(eval_script)
        self.evals_run = []
        self.j = 0

    def eval_segment(self, seg, it, num_envs=64, poll_s=0, timeout_s=0):
        pr = self.eval_script[min(self.j, len(self.eval_script) - 1)]
        self.j += 1
        self.evals_run.append((seg.name, it))
        if pr is None:
            raise RuntimeError("scripted eval failure")
        return {"it": it, "success_rate": 0.0, "progress_rate": pr,
                "mpjpe_all_mean": 60.0, "mpjpe_pa_all_mean": 20.0,
                "failed_keys": ["m1", "m2"]}


def run_eval_driver(script, eval_script, arm="manager", segments=None, **policy_kw):
    fake = FakeEvalAdapter(script, eval_script)
    driver = SmokeDriver(TrainSideBandPolicy(**policy_kw), adapter=fake, arm=arm)
    summary = driver.run(segments or len(script))
    return driver, fake, summary


def test_eval_runs_every_segment_both_arms():
    for arm in ("control", "manager"):
        driver, fake, _ = run_eval_driver([(50, 1.0)] * 3, [0.04] * 3, arm=arm)
        assert len(fake.evals_run) == 3
        assert all(e.get("eval", {}).get("progress_rate") == 0.04
                   for e in driver.journal)
    # eval records carry the driver's cumulative iteration counter
    assert [it for _, it in fake.evals_run] == [3, 6, 9]  # 3 fake iters/segment


def test_tripwire_is_eval_side_when_eval_present():
    driver, _, _ = run_eval_driver([(10, 1.0)] * 4, [0.04] * 4,
                                   len_low=20, sustain=2)
    applied = next(e for e in driver.journal if e.get("applied"))
    assert applied["decision"]["tripwire"]["metric"] == "eval/progress_rate"
    assert "eval-side fixed-threshold guard" in applied["decision"]["rationale"]


def test_eval_tripwire_rolls_back_on_progress_collapse():
    # change applied after seg2 (baseline progress 0.04); progress then
    # collapses to 0.01 for 2 consecutive segments -> rollback
    driver, _, s = run_eval_driver(
        [(10, 1.0)] * 5, [0.04, 0.04, 0.01, 0.01, 0.04],
        len_low=20, sustain=2, segments=5)
    assert s["decisions_applied"] == 1
    assert s["rollbacks"] == 1
    origin = next(e for e in driver.journal if e.get("applied"))
    assert origin["outcome"] == "failed_rolled_back"


def test_eval_tripwire_ignores_train_reward_collapse():
    """The v2 caveat retired: training-side reward collapse alone does NOT
    trip an eval-side tripwire."""
    script = [(10, 1.0), (10, 1.0), (10, 0.1), (10, 0.1), (10, 1.0)]
    driver, _, s = run_eval_driver(script, [0.04] * 5,
                                   len_low=20, sustain=2, segments=5)
    assert s["rollbacks"] == 0
    applied = next(e for e in driver.journal if e.get("applied"))
    assert applied["outcome"].startswith("survived")


def test_eval_absolute_floor_guard():
    """At noise-level baselines a big RELATIVE drop is not a breach unless
    the absolute drop clears EVAL_ABS_MIN_DROP."""
    driver, _, s = run_eval_driver(
        [(10, 1.0)] * 5, [0.003, 0.003, 0.001, 0.001, 0.003],
        len_low=20, sustain=2, segments=5)
    assert s["rollbacks"] == 0  # 0.003->0.001 is 66% relative but 0.002 abs = not > floor


def test_eval_failure_does_not_kill_run_or_watch():
    # eval fails on segment 3 (while a change is under watch): the run
    # continues; the watch neither breaches nor clears that segment
    driver, _, s = run_eval_driver(
        [(10, 1.0)] * 6, [0.04, 0.04, None, 0.04, 0.04, 0.04],
        len_low=20, sustain=2, segments=6)
    assert any("eval_error" in e for e in driver.journal)
    assert s["rollbacks"] == 0
    applied = [e for e in driver.journal if e.get("applied")]
    assert applied and applied[0]["outcome"].startswith("survived")
    # the failed-eval segment is explicitly journaled as watch-unchanged
    assert any("watch unchanged" in e.get("tripwire_note", "")
               for e in driver.journal)


def test_consecutive_eval_failures_do_not_score_survived():
    """Review M3: with the watch armed, segments whose eval FAILED must not
    count as clean — previously the stale pre-change record (== baseline)
    was re-read and could never breach, so 2 failed evals scored the change
    `survived` on zero post-change evidence."""
    driver, _, s = run_eval_driver(
        [(10, 1.0)] * 5, [0.04, 0.04, None, None, None],
        len_low=20, sustain=2, segments=5)
    applied = [e for e in driver.journal if e.get("applied")]
    assert applied
    # no post-change eval evidence ever arrived: still pending, NOT survived
    assert applied[0]["outcome"] == "pending"
    assert s["rollbacks"] == 0
    notes = [e.get("tripwire_note", "") for e in driver.journal]
    assert sum("watch unchanged" in n for n in notes) == 3


def test_stale_eval_cannot_arm_baseline():
    """A decision proposed in a segment whose eval failed must be refused —
    arming against the previous segment's record would watch pre-change
    state."""
    driver, _, s = run_eval_driver(
        [(10, 1.0)] * 4, [0.04, None, 0.04, 0.04],
        len_low=20, sustain=2, segments=4)
    # decision proposed after segment 2 (sustain met) but its eval failed
    rejected = [e for e in driver.journal
                if e.get("validation") and not e["validation"]["ok"]]
    assert any("baseline available" in str(e["validation"]["errors"])
               for e in rejected)


def test_observe_during_gated_ticks_no_history_holes():
    """v2 residual 5: gated segments still feed the sustain history, so the
    SECOND change (after the first watch clears) fires as soon as the gate
    lifts rather than needing to rebuild sustain from scratch."""
    driver, _, s = run_eval_driver([(10, 1.0)] * 6, [0.04] * 6,
                                   len_low=20, sustain=2, segments=6)
    applied = [e for e in driver.journal if e.get("applied")]
    # seg2: first change; watch clears after 2 clean evals (seg3, seg4);
    # with an unbroken history the second change lands at seg5 (with holes
    # it would need seg5+seg6 to rebuild sustain and land at seg6)
    assert [e["tick"] for e in applied] == [2, 5]


def test_applied_entries_carry_provenance():
    driver, _, _ = run_eval_driver([(10, 1.0)] * 4, [0.04] * 4,
                                   len_low=20, sustain=2)
    applied = next(e for e in driver.journal if e.get("applied"))
    assert len(applied["digest_hash"]) == 12
    assert applied["applied_at_iter"] == 6  # applied after segment 2 (3 iters each)


def test_summary_carries_eval_series():
    _, _, s = run_eval_driver([(50, 1.0)] * 3, [0.03, 0.04, 0.05])
    assert s["eval_progress_series"] == [0.03, 0.04, 0.05]
    assert s["eval_mpjpe_series"] == [60.0, 60.0, 60.0]


def test_no_eval_adapter_falls_back_to_train_side():
    """FakeAdapter has no eval_segment: run_eval self-disables and the
    tripwire is training-side (the whole v2 suite above covers behavior)."""
    driver, _, _ = run_driver([(10, 1.0)] * 4, len_low=20, sustain=2)
    assert driver.run_eval is False
    applied = next(e for e in driver.journal if e.get("applied"))
    assert applied["decision"]["tripwire"]["metric"] == "Episode/rew_mean"


# ── registry-level pending gate (defense in depth) ───────────────────
def test_registry_rejects_set_while_driver_watch_armed():
    """The validator ITSELF (not just the driver's gate) rejects a new 'set'
    while a change is under tripwire watch: state.pending is armed on apply
    and any decision validated against that state fails."""
    driver, _, s = run_driver([(10, 1.0)] * 3, len_low=20, sustain=2, segments=3)
    assert s["decisions_applied"] == 1
    assert driver.armed is not None  # watch still open after 3 segments
    assert driver.state.pending == "termination_threshold.foot_pos_xyz"
    d = {"action": "set", "knob": "desired_kl", "value": 0.012,
         "rationale": "r", "expected_effect": "e",
         "tripwire": {"metric": "Episode/rew_mean", "drop_pct": 20, "evals": 2}}
    res = driver.registry.validate_decision(d, driver.state)
    assert not res.ok and any("pending change" in e for e in res.errors)


def test_pending_cleared_after_watch_scores():
    driver, _, _ = run_eval_driver([(10, 1.0)] * 5, [0.04] * 5,
                                   len_low=20, sustain=2, segments=5)
    # first change (tick 2) scored after 2 clean evals; before the second
    # change at tick 5 the pending slot must have been cleared and re-armed
    applied = [e for e in driver.journal if e.get("applied")]
    assert len(applied) == 2  # gate cleared, second change landed


def test_pending_cleared_after_rollback():
    script = [(10, 1.0), (10, 1.0), (10, 0.1), (10, 0.1), (10, 1.0)]
    driver, _, s = run_driver(script, len_low=20, sustain=2, segments=5)
    assert s["rollbacks"] == 1
    # after the rollback tick, the state must not still hold the rolled-back
    # knob as pending (unless a NEW change re-armed it later)
    rb_idx = next(i for i, e in enumerate(driver.journal)
                  if e.get("event") == "rollback")
    re_armed = any(e.get("applied") for e in driver.journal[rb_idx + 1:])
    if not re_armed:
        assert driver.state.pending is None


# ── expected_effect scoring (doc 08 §11 amendment 4 payoff) ──────────
class EffectAdapter(FakeAdapter):
    """Like FakeAdapter but episode length RESPONDS to the loosened knob:
    once any termination_threshold.* override is in the launch knobs,
    subsequent segments report long episodes."""

    def parse_segment(self, seg):
        p = super().parse_segment(seg)
        loosened = any(k.startswith("termination_threshold.")
                       for k in self.launched[-1][1])
        if loosened:
            for r in p.train:
                r["Episode/len_mean"] = 80.0
        return p


def test_effect_confirmed_when_lengths_enter_band():
    fake = EffectAdapter([(10, 1.0)] * 5)
    driver = SmokeDriver(TrainSideBandPolicy(len_low=20, sustain=2),
                         adapter=fake, arm="manager")
    driver.run(5)
    applied = next(e for e in driver.journal if e.get("applied"))
    assert applied["outcome"] == "survived_effect_confirmed"
    assert applied["effect"]["observed"] == 80.0
    assert applied["effect"]["confirmed"] is True


def test_effect_not_observed_when_lengths_stay_low():
    driver, _, _ = run_driver([(10, 1.0)] * 5, len_low=20, sustain=2, segments=5)
    applied = next(e for e in driver.journal if e.get("applied"))
    assert applied["outcome"] == "survived_effect_not_observed"
    assert applied["effect"]["check"] == {
        "metric": "Episode/len_mean", "op": ">=", "value": 20}


def test_rolled_back_outcome_unchanged_by_effect_scoring():
    """Rollback wins: a tripped change is failed_rolled_back, never
    effect-scored."""
    script = [(10, 1.0), (10, 1.0), (10, 0.1), (10, 0.1), (10, 1.0)]
    driver, _, _ = run_driver(script, len_low=20, sustain=2, segments=5)
    origin = next(e for e in driver.journal if e.get("applied"))
    assert origin["outcome"] == "failed_rolled_back"
    assert "effect" not in origin


def test_decision_without_effect_check_scores_plain_survived():
    """A decision lacking expected_effect_check still scores 'survived' —
    the effect claim is honestly unscored, not silently confirmed."""

    class NoCheckPolicy(TrainSideBandPolicy):
        def propose(self, digest, state, registry):
            d = super().propose(digest, state, registry)
            d.pop("expected_effect_check", None)
            return d

    fake = FakeAdapter([(10, 1.0)] * 5)
    driver = SmokeDriver(NoCheckPolicy(len_low=20, sustain=2),
                         adapter=fake, arm="manager")
    driver.run(5)
    applied = next(e for e in driver.journal if e.get("applied"))
    assert applied["outcome"] == "survived"
    assert "effect" not in applied


# ── believed-vs-resolved config verification (doc 08 §11 amendment 8) ─
def _cfg_yaml(foot=0.2):
    return (
        "manager_env:\n"
        "  terminations:\n"
        "    foot_pos_xyz:\n"
        "      params:\n"
        f"        threshold: {foot}\n"
    )


class ConfigFakeAdapter(FakeAdapter):
    """FakeAdapter that also exposes the resolved-config seam."""

    def __init__(self, script, config_texts):
        super().__init__(script)
        self.config_texts = list(config_texts)  # one per segment; None = absent
        self.j = 0

    def wait(self, seg, poll_s=0, timeout_s=0):
        seg = super().wait(seg, poll_s, timeout_s)
        seg.experiment_dir = f"/fake/{seg.name}"
        return seg

    def resolved_config_text(self, seg):
        text = self.config_texts[min(self.j, len(self.config_texts) - 1)]
        self.j += 1
        return text


def test_config_verify_adopts_ground_truth_then_passes():
    """First segment seeds beliefs from the resolved config (structural
    replacement for --base-knobs hand-seeding); later segments verify."""
    fake = ConfigFakeAdapter([(50, 1.0)] * 3, [_cfg_yaml(0.2)] * 3)
    driver = SmokeDriver(TrainSideBandPolicy(len_low=20, sustain=2),
                         adapter=fake, arm="manager")
    driver.run(3)
    assert driver.state.current_values["termination_threshold.foot_pos_xyz"] == 0.2
    first = driver.journal[0]["config_verify"]
    assert first["status"] == "ok"
    assert first["adopted"]["termination_threshold.foot_pos_xyz"] == 0.2
    assert all(e["config_verify"]["status"] == "ok" for e in driver.journal)


def test_config_verify_refuses_on_drift():
    """The v2 defect shape: belief says 0.35 (registry default via
    --base-knobs) but the run really launched at strict 0.2 -> refuse."""
    fake = ConfigFakeAdapter([(50, 1.0)] * 2, [_cfg_yaml(0.2)] * 2)
    driver = SmokeDriver(
        TrainSideBandPolicy(len_low=20, sustain=2), adapter=fake,
        arm="manager", base_knobs={"termination_threshold.foot_pos_xyz": 0.35})
    with pytest.raises(knob_registry.ConfigDriftError, match="amendment 8"):
        driver.run(2)
    # nothing journaled for the failing segment: the raise happens before
    # the entry is appended — the exception itself (match above) is the flag


def test_config_verify_missing_config_yaml_flagged_not_fatal():
    fake = ConfigFakeAdapter([(50, 1.0)] * 2, [None, None])
    driver = SmokeDriver(TrainSideBandPolicy(len_low=20, sustain=2),
                         adapter=fake, arm="manager")
    s = driver.run(2)
    assert all(e["config_verify"]["status"] == "no_config_yaml"
               for e in driver.journal)
    assert len(driver.journal) == 2  # both segments ran; missing file is not fatal


def test_config_verify_skipped_for_adapters_without_seam():
    """v2 fakes (plain FakeAdapter) have no resolved_config_text: the
    driver must not invent one."""
    driver, _, _ = run_driver([(50, 1.0)] * 2, len_low=20, sustain=2)
    assert all("config_verify" not in e for e in driver.journal)


# ---------------------------------------------------------------------------
# motion_file pin (v4 post-mortem 2026-07-07): the standard eval pass must
# carry an EXPLICIT motion_file override whenever the run pins training to a
# curriculum motion set, or eval inherits the checkpoint config's 116,924-
# motion curriculum directory (28+ h eval, driver timeout, orphaned GPU job).


class PinCaptureAdapter(FakeAdapter):
    """FakeAdapter + eval_segment that records extra_overrides per call."""

    def __init__(self, script, n_evals):
        super().__init__(script)
        self.eval_calls = []  # (out_suffix, extra_overrides)
        self.n_evals = n_evals

    def eval_segment(self, seg, it, num_envs=64, poll_s=0, timeout_s=0,
                     extra_overrides=None, out_suffix="_eval", raw=False):
        self.eval_calls.append((out_suffix, list(extra_overrides or [])))
        rec = {"it": it, "success_rate": 0.5, "progress_rate": 0.04,
               "mpjpe_all_mean": 60.0, "mpjpe_pa_all_mean": 20.0,
               "failed_keys": []}
        if raw:
            return {"eval/success/success_rate": 0.5, "failed_keys": []}
        return rec


def _write_manifest(tmp_path, n=40, fraction=0.3, salt="pin-test"):
    from smoke_driver import holdout
    keys = [f"m{i:03d}" for i in range(n)]
    path = str(tmp_path / "manifest.json")
    holdout.write_manifest(path, keys, fraction, salt)
    return path


def test_heldout_manifest_without_eval_pin_refused(tmp_path):
    """Structural guard: the config that caused the v4 incident (heldout
    wiring on, standard eval unpinned) must not be constructible."""
    manifest = _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="eval_motion_file"):
        SmokeDriver(TrainSideBandPolicy(),
                    adapter=PinCaptureAdapter([(50, 1.0)], 1), arm="manager",
                    heldout_manifest=manifest,
                    heldout_eval_motion_file="data/x/robot_heldout_eval64",
                    curriculum_motion_file="data/x/robot_curriculum")


def test_standard_eval_pass_pins_motion_file(tmp_path):
    manifest = _write_manifest(tmp_path)
    fake = PinCaptureAdapter([(50, 1.0)] * 2, 4)
    driver = SmokeDriver(TrainSideBandPolicy(len_low=20, sustain=2),
                         adapter=fake, arm="manager",
                         heldout_manifest=manifest,
                         heldout_eval_motion_file="data/x/robot_heldout_eval64",
                         curriculum_motion_file="data/x/robot_curriculum",
                         eval_motion_file="data/x/robot_curriculum_eval64")
    driver.run(2)
    std = [(sfx, ov) for sfx, ov in fake.eval_calls if sfx == "_eval"]
    held = [(sfx, ov) for sfx, ov in fake.eval_calls if sfx == "_heldout_eval"]
    assert len(std) == 2 and len(held) == 2
    # every STANDARD eval pass carries the fixed-subset pin
    for _, ov in std:
        assert any(o.endswith("motion_file=data/x/robot_curriculum_eval64")
                   for o in ov), ov
    # the held-out pass keeps its own pin — never the standard one
    for _, ov in held:
        assert any(o.endswith("motion_file=data/x/robot_heldout_eval64")
                   for o in ov), ov
        assert not any("robot_curriculum_eval64" in o for o in ov)


def test_eval_pin_absent_without_heldout_wiring():
    """No heldout manifest, no eval_motion_file: legacy behavior — the
    adapter is called without any motion_file override (checkpoint config
    rules), matching the v2/v3 runs whose artifacts we keep."""
    fake = PinCaptureAdapter([(50, 1.0)] * 2, 2)
    driver = SmokeDriver(TrainSideBandPolicy(len_low=20, sustain=2),
                         adapter=fake, arm="manager")
    driver.run(2)
    assert all(ov == [] for _, ov in fake.eval_calls)


# ---------------------------------------------------------------------------
# E0: disk hygiene — per-segment checkpoint purge + pre-launch free-space gate
# (per-segment checkpoints are ~3.6 GB on a 41-GB-free volume; a 20-segment
# comparison without purging would write ~72 GB).


def _make_run_dir(tmp_path):
    """Run dir shaped like a real segment's experiment_dir."""
    run = tmp_path / "manager_s1-20260707_120000"
    run.mkdir()
    (run / "last.pt").write_bytes(b"x" * 1000)
    (run / "model_step_000005.pt").write_bytes(b"y" * 2000)
    (run / "model_step_000010.pt").write_bytes(b"z" * 3000)
    (run / "snapshot_manager_s1.pt").write_bytes(b"s" * 4000)
    (run / "config.yaml").write_text("cfg: 1\n")
    ev = run / "eval_out"
    ev.mkdir()
    (ev / "metrics_eval.json").write_text("{}")
    return run


def test_purge_deletes_intermediates_keeps_snapshots(tmp_path):
    run = _make_run_dir(tmp_path)
    deleted, freed = purge_intermediate_checkpoints(str(run))
    assert sorted(deleted) == ["last.pt", "model_step_000005.pt",
                               "model_step_000010.pt"]
    assert freed == 1000 + 2000 + 3000
    survivors = sorted(p.name for p in run.iterdir())
    assert "snapshot_manager_s1.pt" in survivors      # rollback point kept
    assert "eval_out" in survivors                    # eval dir kept
    assert "config.yaml" in survivors
    assert not any(n.startswith("model_step") or n == "last.pt"
                   for n in survivors)
    # idempotent: a second purge deletes nothing
    assert purge_intermediate_checkpoints(str(run)) == ([], 0)


class PurgeFakeAdapter(FakeAdapter):
    """FakeAdapter whose wait() exposes a REAL host run dir (the seam the
    driver's purge reads: the training volume is bind-mounted at the same
    path in and out of the container)."""

    def __init__(self, script, run_dirs):
        super().__init__(script)
        self.run_dirs = list(run_dirs)

    def wait(self, seg, poll_s=0, timeout_s=0):
        seg = super().wait(seg, poll_s, timeout_s)
        seg.experiment_dir = self.run_dirs.pop(0) if self.run_dirs else None
        return seg


def test_purge_event_journaled_after_verified_segment(tmp_path):
    run = _make_run_dir(tmp_path)
    fake = PurgeFakeAdapter([(50, 1.0)], [str(run)])
    driver = SmokeDriver(TrainSideBandPolicy(len_low=20, sustain=2),
                         adapter=fake, arm="manager")
    driver.run(1)
    purge = driver.journal[0]["checkpoint_purge"]
    assert purge["run_dir"] == str(run)
    assert sorted(purge["deleted"]) == ["last.pt", "model_step_000005.pt",
                                        "model_step_000010.pt"]
    assert purge["bytes_freed"] == 6000
    assert (run / "snapshot_manager_s1.pt").exists()
    # journal stays JSON-serializable with the new event
    json.dumps(driver.journal)


def test_purge_noop_without_run_dir():
    """v2-style fakes (no experiment_dir): no purge event, no crash."""
    driver, _, _ = run_driver([(50, 1.0)] * 2, len_low=20, sustain=2)
    assert all("checkpoint_purge" not in e for e in driver.journal)


def test_failed_segment_not_purged(tmp_path):
    """A segment that did not finish keeps ALL its checkpoints — last.pt
    may be the only recovery point."""
    run = _make_run_dir(tmp_path)

    class FailingAdapter(PurgeFakeAdapter):
        def wait(self, seg, poll_s=0, timeout_s=0):
            seg = super().wait(seg, poll_s, timeout_s)
            seg.status = "failed"
            return seg

    fake = FailingAdapter([(50, 1.0)], [str(run)])
    driver = SmokeDriver(TrainSideBandPolicy(), adapter=fake, arm="manager")
    driver.run(1)
    assert (run / "last.pt").exists()
    assert driver.journal[0]["event"] == "segment_failed"


def _fake_usage(free):
    import collections
    U = collections.namedtuple("usage", "total used free")
    return U(total=100 * 1024**3, used=100 * 1024**3 - free, free=free)


def test_disk_gate_blocks_launch_below_8gb(tmp_path, monkeypatch):
    import smoke_driver as sd
    monkeypatch.setattr(sd.shutil, "disk_usage",
                        lambda p: _fake_usage(4 * 1024**3))
    fake = FakeAdapter([(50, 1.0)] * 2)
    driver = SmokeDriver(TrainSideBandPolicy(), adapter=fake, arm="manager",
                         disk_gate_path=str(tmp_path))
    with pytest.raises(DiskSpaceError, match="refusing to launch"):
        driver.run(2)
    assert fake.launched == []  # fail-fast: NO training launch happened
    ev = driver.journal[-1]
    assert ev["event"] == "disk_gate_failed"
    assert ev["free_bytes"] == 4 * 1024**3
    assert ev["required_bytes"] == 8 * 1024**3
    assert ev["path"] == str(tmp_path)


def test_disk_gate_passes_with_headroom(tmp_path, monkeypatch):
    import smoke_driver as sd
    monkeypatch.setattr(sd.shutil, "disk_usage",
                        lambda p: _fake_usage(20 * 1024**3))
    fake = FakeAdapter([(50, 1.0)] * 2)
    driver = SmokeDriver(TrainSideBandPolicy(), adapter=fake, arm="manager",
                         disk_gate_path=str(tmp_path))
    driver.run(2)
    assert len(fake.launched) == 2
    assert all(e.get("event") != "disk_gate_failed" for e in driver.journal)


def test_disk_gate_off_for_injected_adapters_without_path():
    """Unit-test fakes never hit the host's real disk state."""
    fake = FakeAdapter([(50, 1.0)])
    driver = SmokeDriver(TrainSideBandPolicy(), adapter=fake, arm="manager")
    assert driver.disk_gate_path is None


# ---------------------------------------------------------------------------
# E1: scripted-decision ablation arm — open-loop replay of the exact knob
# ladder both v4 manager seeds walked (verified via knobs_in in the v4
# journals). No digest reads for the CHOICE; eval-side tripwire stays armed.

# the v4 runs' actual stock starting values (BASE_KNOBS in
# run_comparison_multiseed.sh) — the ladder's notch arithmetic starts here
V4_BASE_KNOBS = {"termination_threshold.anchor_pos": 0.15,
                 "termination_threshold.ee_body_pos": 0.15,
                 "termination_threshold.foot_pos_xyz": 0.2}


def run_scripted(script, eval_script, segments=None):
    fake = FakeEvalAdapter(script, eval_script)
    driver = SmokeDriver(ScriptedPolicy(), adapter=fake, arm="scripted",
                         base_knobs=dict(V4_BASE_KNOBS))
    summary = driver.run(segments or len(script))
    return driver, fake, summary


def test_scripted_replays_exact_v4_ladder():
    driver, fake, s = run_scripted([(50, 1.0)] * 10, [0.04] * 10)
    applied = [e for e in driver.journal if e.get("applied")]
    assert [(e["tick"], e["decision"]["knob"], e["decision"]["value"])
            for e in applied] == [
        (2, "termination_threshold.foot_pos_xyz", 0.25),
        (4, "termination_threshold.ee_body_pos", 0.20),
        (6, "termination_threshold.foot_pos_xyz", 0.30),
        (8, "termination_threshold.ee_body_pos", 0.25),
        (10, "termination_threshold.foot_pos_xyz", 0.35),
    ]
    assert s["decisions_applied"] == 5 and s["rollbacks"] == 0
    # final knobs match the v4 manager arms' endpoint
    assert s["final_knobs"]["termination_threshold.foot_pos_xyz"] == 0.35
    assert s["final_knobs"]["termination_threshold.ee_body_pos"] == 0.25


def test_scripted_no_decision_at_off_ladder_ticks():
    driver, _, _ = run_scripted([(50, 1.0)] * 10, [0.04] * 10)
    off = [e for e in driver.journal if e["tick"] not in V4_MANAGER_LADDER]
    assert off and all(e["decision"]["action"] == "none" for e in off)
    assert all("scripted ladder" in e["decision"]["reason"] for e in off)


def test_scripted_choice_ignores_digest_and_eval_state():
    """The CHOICE reads only state.tick: wildly different digests (including
    None) at the same tick yield the identical knob/value; digests that
    would drive TrainSideBandPolicy differently change nothing."""
    registry = knob_registry.load_registry()
    policy = ScriptedPolicy()
    digests = [
        None,
        {},
        {"train": {"Episode/len_mean": {"last": 1.0},
                   "termination_terms_mean_recent": {"ee_body_pos": 0.99}},
         "eval": {"progress_rate": {"last": 0.0001}}},
        {"train": {"Episode/len_mean": {"last": 500.0}},
         "eval": {"progress_rate": {"last": 0.9}}},
    ]
    for tick, (knob, value) in V4_MANAGER_LADDER.items():
        for dg in digests:
            state = knob_registry.RunState(tick=tick)
            d = policy.propose(dg, state, registry)
            assert (d["action"], d["knob"], d["value"]) == ("set", knob, value), (tick, dg)
    # off-ladder tick: none, regardless of digest
    for dg in digests:
        state = knob_registry.RunState(tick=3)
        assert policy.propose(dg, state, registry)["action"] == "none"


def test_scripted_policy_keeps_no_history():
    """No observe(), no internal run-state: the driver's observe() hook is
    simply absent for this policy (open-loop by construction)."""
    assert not hasattr(ScriptedPolicy(), "observe")


def test_scripted_ladder_not_gated_by_watch_window():
    """Even when a prior rung's tripwire watch is still open at the next
    rung's tick, the ladder fires on schedule (no watch-window gating);
    the open watch is retired unscored, journaled explicitly."""
    # eval fails on segment 3: the tick-2 rung's watch cannot clear by
    # tick 4, yet the tick-4 rung must still land
    driver, _, s = run_scripted([(50, 1.0)] * 5,
                                [0.04, 0.04, None, 0.04, 0.04], segments=5)
    applied = [e for e in driver.journal if e.get("applied")]
    assert [e["tick"] for e in applied] == [2, 4]
    assert any("retired unscored" in e.get("tripwire_note", "")
               for e in driver.journal)


def test_scripted_arm_tripwire_safety_stays_armed():
    """Decisions are fixed but the eval-side tripwire is NOT disabled:
    every applied rung arms an eval/progress_rate watch, and a genuine
    progress collapse still rolls back."""
    driver, _, _ = run_scripted([(50, 1.0)] * 4, [0.04] * 4, segments=4)
    applied = [e for e in driver.journal if e.get("applied")]
    assert all(e["decision"]["tripwire"]["metric"] == "eval/progress_rate"
               for e in applied)
    # collapse after the tick-2 rung (0.04 -> 0.005: >30% rel and >abs floor)
    driver, _, s = run_scripted([(50, 1.0)] * 4,
                                [0.04, 0.04, 0.005, 0.005], segments=4)
    assert s["rollbacks"] == 1
    origin = next(e for e in driver.journal if e.get("applied"))
    assert origin["outcome"] == "failed_rolled_back"


def test_scripted_control_and_manager_arms_unaffected():
    """Arm selection is additive: existing arms behave exactly as before
    with the ladder module present (spot check)."""
    driver, fake, s = run_driver([(10, 1.0)] * 4, arm="control")
    assert s["decisions_applied"] == 0
    assert all(k == {} for _, k, _ in fake.launched)
    driver, _, s = run_driver([(10, 1.0)] * 4, len_low=20, sustain=2)
    assert s["decisions_applied"] == 1
