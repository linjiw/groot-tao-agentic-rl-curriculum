# SPDX-License-Identifier: Apache-2.0
"""Tests for the digest builder (design doc 08 §4)."""

import json
import math

import pytest

from digest_builder import (
    build_digest,
    build_eval_section,
    cap_saturation_fraction,
    normalized_entropy,
    read_jsonl,
    summarize_series,
    top_k_share,
)


# ── stats primitives ─────────────────────────────────────────────────
def test_normalized_entropy_bounds():
    assert normalized_entropy([1, 1, 1, 1]) == pytest.approx(1.0)
    assert normalized_entropy([1, 0, 0, 0]) == pytest.approx(0.0)
    assert normalized_entropy([]) is None
    assert normalized_entropy([0.0, 0.0]) is None


def test_cap_saturation_matches_sampler_semantics():
    # mean=0.25, max_over_mean=2 → cap=0.5; one of four bins at/above
    assert cap_saturation_fraction([0.1, 0.1, 0.1, 0.7], 2.0) == pytest.approx(0.25)
    # huge cap → nothing saturated
    assert cap_saturation_fraction([0.1, 0.2], 100.0) == 0.0
    assert cap_saturation_fraction([], 2.0) is None
    assert cap_saturation_fraction([0.0, 0.0], 2.0) == 0.0


def test_top_k_share():
    assert top_k_share([1, 0, 0, 0], k=1) == pytest.approx(1.0)
    assert top_k_share([1, 1, 1, 1], k=2) == pytest.approx(0.5)
    assert top_k_share([], k=3) is None


def test_summarize_series_trends():
    rising = summarize_series([0.1, 0.2, 0.3, 0.4, 0.5], window=5)
    assert rising["trend"] == "rising" and rising["last"] == 0.5
    falling = summarize_series([0.5, 0.4, 0.3], window=5)
    assert falling["trend"] == "falling"
    flat = summarize_series([0.500, 0.501, 0.499, 0.500], window=5)
    assert flat["trend"] == "flat"
    assert summarize_series([0.7], window=5)["trend"] == "unknown"  # 1 point
    assert summarize_series([], window=5)["last"] is None


def test_summarize_series_respects_window():
    # falling overall, rising in the window
    s = summarize_series([0.9, 0.5, 0.1, 0.2, 0.3], window=3)
    assert s["trend"] == "rising" and s["n_points"] == 3


# ── eval section ─────────────────────────────────────────────────────
def _eval_rec(it, sr, failed, heldout=None):
    r = {"it": it, "success_rate": sr, "failed_keys": failed}
    if heldout is not None:
        r["heldout_success_rate"] = heldout
    return r


def test_failed_keys_diff_and_persistence():
    records = [
        _eval_rec(100, 0.5, ["a", "b", "c"]),
        _eval_rec(200, 0.6, ["b", "c", "d"]),
        _eval_rec(300, 0.7, ["c", "d"]),
    ]
    sec = build_eval_section(records, window=3)
    fk = sec["failed_keys"]
    assert fk["count"] == 2
    assert fk["newly_failing"] == []
    assert fk["newly_recovered"] == ["b"]
    assert fk["persistent"] == ["c"]  # failing in all 3 evals


def test_heldout_tracked_separately():
    records = [_eval_rec(100, 0.8, [], heldout=0.5), _eval_rec(200, 0.9, [], heldout=0.4)]
    sec = build_eval_section(records, window=5)
    assert sec["success_rate"]["last"] == 0.9
    assert sec["heldout_success_rate"]["last"] == 0.4  # decoupling visible


def test_missing_heldout_is_none_not_crash():
    sec = build_eval_section([_eval_rec(100, 0.8, [])], window=5)
    assert sec["heldout_success_rate"] is None


# ── full digest ──────────────────────────────────────────────────────
def _train_rec(it, kl=0.01, ent=1.0):
    return {
        "it": it,
        "policy/approxkl_avg": kl,
        "loss/entropy_avg": ent,
        "loss/value_avg": 0.5,
        "lr": 2e-5,
        "Episode/tracking_anchor_pos": 0.8,
        "Episode/feet_acc": -0.01,
        "scheduled_params/entropy_coef": 0.01,
        "irrelevant_key": "ignored",
    }


def _sampler_rec(it, rates):
    return {"it": it, "failure_rate": rates}


def test_full_digest_shape_and_serializability():
    digest = build_digest(
        train_records=[_train_rec(i) for i in (100, 200, 300)],
        eval_records=[_eval_rec(250, 0.6, ["m1"], heldout=0.55)],
        sampler_records=[_sampler_rec(300, [0.1, 0.1, 0.8, 0.0])],
        knob_state={"uniform_sampling_rate": {"value": 0.1, "ticks_since_change": 7}},
        decision_history=[{"tick": 3, "decision": {"action": "none"}, "outcome": "n/a"}],
        max_over_mean=2.0,
        window=5,
    )
    json.dumps(digest)  # must be JSON-serializable
    assert digest["last_iteration"] == 300
    assert digest["eval"]["success_rate"]["last"] == 0.6
    assert digest["sampler"]["num_bins"] == 4
    assert digest["sampler"]["cap_saturation_fraction"] == pytest.approx(0.25)
    assert digest["train"]["episode_terms_last"] == {
        "feet_acc": -0.01,
        "tracking_anchor_pos": 0.8,
    }
    assert digest["train"]["scheduled_params_last"] == {"entropy_coef": 0.01}
    assert digest["knobs"]["uniform_sampling_rate"]["value"] == 0.1
    assert digest["decision_history"][0]["tick"] == 3


def test_empty_inputs_produce_valid_digest():
    digest = build_digest()
    json.dumps(digest)
    assert digest["eval"] is None
    assert digest["sampler"] is None
    assert digest["train"] is None
    assert digest["last_iteration"] is None


def test_sampler_section_without_failure_rate_is_none():
    digest = build_digest(sampler_records=[{"it": 100}])
    assert digest["sampler"] is None


# ── jsonl reader ─────────────────────────────────────────────────────
def test_read_jsonl_roundtrip_and_errors(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"it": 1}\n\n{"it": 2}\n')
    assert [r["it"] for r in read_jsonl(str(p))] == [1, 2]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"it": 1}\nnot json\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_jsonl(str(bad))


def test_cli_end_to_end(tmp_path):
    from digest_builder import main

    (tmp_path / "eval.jsonl").write_text(
        json.dumps(_eval_rec(100, 0.7, ["k1"])) + "\n"
    )
    out = tmp_path / "digest.json"
    rc = main(["--eval", str(tmp_path / "eval.jsonl"), "--out", str(out)])
    assert rc == 0
    digest = json.loads(out.read_text())
    assert digest["eval"]["success_rate"]["last"] == 0.7
