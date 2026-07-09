# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure-python logic of score_comparison_multiseed.py:
seed parsing, aggregation math, prefix-identity, and report assembly
from fixture journals/evals (no docker, no container reads)."""

import json
import subprocess

import pytest

import score_comparison_multiseed as m


# ── seed parsing ──────────────────────────────────────────────────────

def test_parse_seeds_ints_and_strings():
    assert m.parse_seeds([42, 1337]) == [42, 1337]
    assert m.parse_seeds(["42", "1337"]) == [42, 1337]


def test_parse_seeds_comma_and_space_separated():
    assert m.parse_seeds(["42,1337"]) == [42, 1337]
    assert m.parse_seeds(["42 1337 7"]) == [42, 1337, 7]


def test_parse_seeds_rejects_empty_and_duplicates():
    with pytest.raises(ValueError):
        m.parse_seeds([])
    with pytest.raises(ValueError):
        m.parse_seeds(["42", "42"])
    with pytest.raises(ValueError):
        m.parse_seeds(["not_a_seed"])


# ── aggregation math ──────────────────────────────────────────────────

def test_mean_range_basic():
    agg = m.mean_range([0.004, 0.002])
    assert agg == {"mean": 0.003, "min": 0.002, "max": 0.004,
                   "n": 2, "n_missing": 0}


def test_mean_range_skips_none_and_nonfinite_but_counts_them():
    agg = m.mean_range([1.0, None, float("nan"), 3.0])
    assert agg["mean"] == 2.0 and agg["n"] == 2 and agg["n_missing"] == 2


def test_mean_range_all_missing_is_none():
    assert m.mean_range([None, None]) is None
    assert m.mean_range([]) is None


def test_aggregate_final_uses_last_finite_value_per_seed():
    r = m.aggregate_final({42: [0.001, 0.004, None], 1337: [0.002, 0.002]})
    assert r["per_seed_final"] == {42: 0.004, 1337: 0.002}
    assert r["cross_seed"]["mean"] == 0.003
    assert r["cross_seed"]["min"] == 0.002 and r["cross_seed"]["max"] == 0.004


def test_aggregate_final_all_none_seed():
    r = m.aggregate_final({42: [None, None]})
    assert r["per_seed_final"] == {42: None}
    assert r["cross_seed"] is None


def test_aggregate_per_segment_ragged():
    out = m.aggregate_per_segment({42: [1.0, 2.0, 3.0], 1337: [3.0, 4.0]})
    assert out[0]["mean"] == 2.0 and out[0]["n"] == 2
    assert out[1]["mean"] == 3.0
    # segment 3 exists only for seed 42: n=1, one visibly missing
    assert out[2]["n"] == 1 and out[2]["n_missing"] == 1


# ── prefix identity ───────────────────────────────────────────────────

def _entry(tick, ln, **kw):
    return {"tick": tick, "len_mean_last": ln, **kw}


def test_prefix_identity_holds_before_first_change():
    ctrl = [_entry(1, 18.02), _entry(2, 15.37), _entry(3, 20.47)]
    mgr = [_entry(1, 18.02), _entry(2, 15.37),
           _entry(3, 20.43, applied=True,
                  decision={"knob": "k", "value": 0.25})]
    # applied at tick 3 -> ticks 1-2 must match
    mgr[2]["tick"] = 3
    r = m.prefix_identity(ctrl, mgr)
    assert r == {"first_change_tick": 3, "prefix_identical": True,
                 "n_compared": 2}


def test_prefix_identity_detects_divergence():
    ctrl = [_entry(1, 18.02), _entry(2, 15.37)]
    mgr = [_entry(1, 18.02), _entry(2, 99.0),
           _entry(3, 1.0, applied=True, decision={})]
    r = m.prefix_identity(ctrl, mgr)
    assert r["prefix_identical"] is False and r["first_change_tick"] == 3


def test_prefix_identity_no_applied_change_compares_full_overlap():
    ctrl = [_entry(1, 10.0), _entry(2, 11.0)]
    mgr = [_entry(1, 10.0), _entry(2, 11.0)]
    r = m.prefix_identity(ctrl, mgr)
    assert r["first_change_tick"] is None
    assert r["prefix_identical"] is True and r["n_compared"] == 2


def test_prefix_identity_change_on_first_tick_yields_none():
    ctrl = [_entry(1, 10.0)]
    mgr = [_entry(1, 10.0, applied=True, decision={})]
    r = m.prefix_identity(ctrl, mgr)
    assert r["prefix_identical"] is None and r["n_compared"] == 0


# ── report assembly from fixtures ─────────────────────────────────────

def _fixture_journal(arm, n=3, applied_tick=None):
    out = []
    for i in range(1, n + 1):
        e = _entry(i, 15.0 + i, rew_mean_last=1.0,
                   segment=f"{arm}_s{i}", knobs_in={})
        if arm == "manager" and applied_tick == i:
            e["applied"] = True
            e["decision"] = {"knob": "termination_threshold.foot_pos_xyz",
                             "value": 0.25}
            e["digest_hash"] = "abc123"
            e["applied_at_iter"] = i * 50
            e["outcome"] = "survived"
        out.append(e)
    return out


def _fixture_evals(n=3, prog0=0.003, source="_eval"):
    return [{"eval/success/progress_rate": prog0 + 0.00025 * i,
             "eval/all/mpjpe_l": 28.0 - 0.1 * i,
             "eval/all/mpjpe_g": 60.0,
             "_eval_source": source} for i in range(n)]


def _fixture_inputs(seeds=(42, 1337), n=3):
    journals = {s: {"control": _fixture_journal("control", n),
                    "manager": _fixture_journal("manager", n, applied_tick=2)}
                for s in seeds}
    evals = {s: {"control": _fixture_evals(n),
                 "manager": _fixture_evals(n, prog0=0.00325)}
             for s in seeds}
    return journals, evals


def test_build_report_structure_and_caveats():
    journals, evals = _fixture_inputs()
    r = m.build_report(journals, evals)
    assert r["seeds"] == [42, 1337]
    # honest-reporting caveats are always present
    assert "quanta" in r["caveats"]["quantization"]
    assert "survivor" in r["caveats"]["mpjpe_g"]
    assert set(r["eval_progress_rate_PRIMARY"]) == {
        "per_seed_per_segment", "final_cross_seed", "per_segment_cross_seed"}
    # JSON-serializable end to end
    json.dumps(r)


def test_build_report_primary_aggregation_math():
    journals, evals = _fixture_inputs()
    r = m.build_report(journals, evals)
    prim = r["eval_progress_rate_PRIMARY"]
    # both seeds identical fixtures -> range collapses, n=2
    ctrl_final = prim["final_cross_seed"]["control"]["cross_seed"]
    assert ctrl_final["n"] == 2 and ctrl_final["min"] == ctrl_final["max"]
    # manager fixture is one quantum ahead of control at every segment
    # (fmt rounds tables to 4 decimals, so allow rounding slack)
    mgr_final = prim["final_cross_seed"]["manager"]["cross_seed"]
    assert mgr_final["mean"] == pytest.approx(
        ctrl_final["mean"] + 0.00025, abs=1e-4)
    assert mgr_final["mean"] > ctrl_final["mean"]


def test_build_report_prefix_identity_per_seed():
    journals, evals = _fixture_inputs()
    # identical fixtures -> prefix holds for both seeds
    r = m.build_report(journals, evals)
    assert r["prefix_identity_all_seeds"] is True
    assert all(v["prefix_identical"] for v in
               r["prefix_identity_per_seed"].values())
    # break seed 1337's prefix -> flagged per-seed AND globally
    journals[1337]["manager"][0]["len_mean_last"] = 999.0
    r2 = m.build_report(journals, evals)
    assert r2["prefix_identity_per_seed"][1337]["prefix_identical"] is False
    assert r2["prefix_identity_per_seed"][42]["prefix_identical"] is True
    assert r2["prefix_identity_all_seeds"] is False


def test_build_report_decisions_rollbacks_errors_per_seed():
    journals, evals = _fixture_inputs()
    journals[42]["manager"].append(
        {"tick": 4, "event": "rollback",
         "restored": {"termination_threshold.foot_pos_xyz": 0.2}})
    journals[1337]["control"][1]["eval_error"] = "GPU busy"
    r = m.build_report(journals, evals)
    assert [d["knob"] for d in r["manager_decisions"][42]] == \
        ["termination_threshold.foot_pos_xyz"]
    assert r["rollbacks"][42]["manager"] == 1
    assert r["rollbacks"][1337]["manager"] == 0
    assert r["eval_errors"][1337]["control"] == ["GPU busy"]


def test_build_report_missing_evals_stay_visible():
    journals, evals = _fixture_inputs()
    evals[42]["manager"][2] = None  # a failed eval pass
    r = m.build_report(journals, evals)
    prim = r["eval_progress_rate_PRIMARY"]
    assert prim["per_seed_per_segment"]["manager"][42][2] is None
    seg3 = prim["per_segment_cross_seed"]["manager"][2]
    assert seg3["n"] == 1 and seg3["n_missing"] == 1
    assert r["eval_sources"][42]["manager"][2] is None


# ── container reader fallback (fake runner, no docker) ────────────────

def test_container_eval_prefers_pinned_and_tolerates_failures():
    calls = []

    def fake_runner(cmd, capture_output, text, timeout):
        path = cmd[-1]
        calls.append(path)

        import types
        r = types.SimpleNamespace()
        # s1: pinned exists; s2: only unpinned; s3: nothing
        if "s1_eval_pinned" in path:
            r.returncode, r.stdout = 0, json.dumps({"k": 1})
        elif "s2_eval_pinned" in path or "s3" in path:
            r.returncode, r.stdout = 1, ""
        elif "s2_eval" in path:
            r.returncode, r.stdout = 0, json.dumps({"k": 2})
        else:
            r.returncode, r.stdout = 1, ""
        return r

    out = m._container_eval("cmp_manager_seed42", "manager", 3,
                            runner=fake_runner)
    assert out[0] == {"k": 1, "_eval_source": "_eval_pinned"}
    assert out[1] == {"k": 2, "_eval_source": "_eval"}
    assert out[2] is None
    assert calls[0].endswith(
        "cmp_manager_seed42_manager_s1_eval_pinned/metrics_eval.json")


def test_container_eval_survives_subprocess_errors():
    def boom(*a, **k):
        raise subprocess.SubprocessError("docker gone")
    out = m._container_eval("p", "control", 2, runner=boom)
    assert out == [None, None]


# ── per-motion decomposition (unit fixtures) ──────────────────────────

def _pm_entry(tick, pm):
    return {"tick": tick, "len_mean_last": 15.0,
            "eval": {"per_motion_progress": pm}}


def test_paired_final_delta_counts_median_and_top5():
    mgr = {"a": 0.3, "b": 0.1, "c": 0.2, "d": 0.05, "e": 0.0, "f": 0.5}
    ctl = {"a": 0.1, "b": 0.2, "c": 0.2, "d": 0.05, "e": 0.0, "f": 0.1}
    r = m.paired_final_delta(mgr, ctl)
    assert (r["wins"], r["losses"], r["ties"]) == (2, 1, 3)
    assert r["n_motions"] == 6
    assert r["median_delta"] == 0.0
    tops = [t["motion"] for t in r["top5_by_abs_delta"]]
    assert tops[0] == "f" and tops[1] == "a" and len(tops) == 5
    assert r["top5_by_abs_delta"][0]["delta"] == pytest.approx(0.4)


def test_paired_final_delta_shared_motions_only():
    r = m.paired_final_delta({"a": 1.0, "x": 9.0}, {"a": 0.5, "y": 9.0})
    assert r["n_motions"] == 1 and r["wins"] == 1
    assert m.paired_final_delta({"x": 1.0}, {"y": 1.0}) is None


def test_largest_jump_decomposition_finds_single_motion_artifact():
    # aggregate jumps: seg1->2 driven ~91% by 'spike'
    j = [_pm_entry(1, {"spike": 0.0, "a": 0.1, "b": 0.1}),
         _pm_entry(2, {"spike": 0.5, "a": 0.12, "b": 0.13}),
         _pm_entry(3, {"spike": 0.5, "a": 0.12, "b": 0.13})]
    r = m.largest_jump_decomposition(j)
    assert r["from_segment"] == 1 and r["to_segment"] == 2
    assert r["top_motion"] == "spike"
    assert r["jump"] == pytest.approx(0.55 / 3, abs=1e-6)
    assert r["top_motion_fraction_of_jump"] == pytest.approx(
        0.5 / 0.55, abs=1e-3)


def test_largest_jump_decomposition_needs_two_segments():
    assert m.largest_jump_decomposition([_pm_entry(1, {"a": 1.0})]) is None
    assert m.largest_jump_decomposition(
        [{"tick": 1}, {"tick": 2}]) is None  # no per-motion data at all


def test_leave_one_out_min_max_and_named_exclusion():
    pm = {"lo": 0.0, "mid": 0.2, "hi": 1.0}
    r = m.leave_one_out(pm, exclude_motion="hi")
    assert r["full_mean"] == pytest.approx(0.4)
    # excluding the highest motion gives the LOWEST leave-one-out mean
    assert r["min"]["excluded_motion"] == "hi"
    assert r["min"]["value"] == pytest.approx(0.1)
    assert r["max"]["excluded_motion"] == "lo"
    assert r["max"]["value"] == pytest.approx(0.6)
    assert r["excluding_top_contributor"]["motion"] == "hi"
    assert r["excluding_top_contributor"]["value"] == pytest.approx(0.1)


def test_leave_one_out_degenerate():
    assert m.leave_one_out(None) is None
    assert m.leave_one_out({"only": 1.0}) is None


def test_build_report_backward_compatible_without_per_motion_data():
    # fixtures WITHOUT eval.per_motion_progress: old fields unchanged,
    # new per_motion section present but empty-per-seed (None subfields)
    journals, evals = _fixture_inputs()
    r = m.build_report(journals, evals)
    assert "per_motion" in r
    for seed in (42, 1337):
        sec = r["per_motion"][seed]
        assert sec["final_paired"] is None
        assert sec["largest_jump"] == {"control": None, "manager": None}
        assert sec["leave_one_out_final"] == {"control": None,
                                              "manager": None}
    json.dumps(r)


def test_build_report_per_motion_from_fixture_journals():
    journals, evals = _fixture_inputs(seeds=(42,))
    for i, e in enumerate(journals[42]["control"]):
        e["eval"] = {"per_motion_progress": {"a": 0.1, "b": 0.1 + 0.01 * i}}
    for i, e in enumerate(journals[42]["manager"]):
        e["eval"] = {"per_motion_progress": {"a": 0.1 + 0.3 * (i == 2),
                                             "b": 0.1}}
    r = m.build_report(journals, evals)
    sec = r["per_motion"][42]
    p = sec["final_paired"]
    assert (p["wins"], p["losses"], p["ties"]) == (1, 1, 0)
    assert sec["largest_jump"]["manager"]["top_motion"] == "a"
    assert sec["largest_jump"]["manager"]["top_motion_fraction_of_jump"] \
        == pytest.approx(1.0)
    loo = sec["leave_one_out_final"]["manager"]
    assert loo["excluding_top_contributor"]["motion"] == "a"
    assert loo["excluding_top_contributor"]["value"] == pytest.approx(0.1)
    json.dumps(r)


# ── integration: real v4 journals (parent-verified reference values) ──

import os

V4_DIR = os.path.dirname(os.path.abspath(__file__))
_V4_JOURNALS = [os.path.join(V4_DIR, f"{arm}_journal_v4_seed{s}.json")
                for arm in ("control", "manager") for s in (42, 1337)]

POSTMORTEM = "postmortem_convulsions_stomach_loop_R_001__A471_M"


@pytest.mark.skipif(not all(os.path.exists(p) for p in _V4_JOURNALS),
                    reason="v4 journals not present")
def test_per_motion_integration_v4_reference_values():
    journals = {s: {arm: json.load(open(os.path.join(
        V4_DIR, f"{arm}_journal_v4_seed{s}.json")))
        for arm in ("control", "manager")} for s in (42, 1337)}
    sec = m.per_motion_decomposition(journals)

    # (a) paired win/loss/tie + median — parent-verified
    p42 = sec[42]["final_paired"]
    assert (p42["wins"], p42["losses"], p42["ties"]) == (22, 22, 20)
    assert abs(p42["median_delta"]) < 5e-5
    p1337 = sec[1337]["final_paired"]
    assert (p1337["wins"], p1337["losses"], p1337["ties"]) == (26, 18, 20)
    assert abs(p1337["median_delta"]) < 5e-5

    # (c) manager arms' largest jumps are driven by the postmortem motion
    for s in (42, 1337):
        mj = sec[s]["largest_jump"]["manager"]
        assert mj["top_motion"] == POSTMORTEM
        assert 0.51 <= mj["top_motion_fraction_of_jump"] <= 0.96

    # (d) leave-one-out excluding the postmortem motion — parent-verified
    def _excl(s, arm):
        loo = sec[s]["leave_one_out_final"][arm]
        # manager top contributor IS the postmortem motion; for control
        # read the same motion's exclusion from the min/max scan by
        # recomputing directly
        pm = journals[s][arm][-1]["eval"]["per_motion_progress"]
        return (sum(v for k, v in pm.items() if k != POSTMORTEM)
                / (len(pm) - 1))

    assert abs(_excl(42, "manager") - 0.0899) < 5e-5
    assert abs(_excl(1337, "manager") - 0.0914) < 5e-5
    assert abs(_excl(42, "control") - 0.0996) < 5e-5
    assert abs(_excl(1337, "control") - 0.0922) < 5e-5
    # and the report's own excluding_top_contributor for manager matches
    for s, ref in ((42, 0.0899), (1337, 0.0914)):
        etc = sec[s]["leave_one_out_final"]["manager"][
            "excluding_top_contributor"]
        assert etc["motion"] == POSTMORTEM
        assert abs(etc["value"] - ref) < 5e-5
