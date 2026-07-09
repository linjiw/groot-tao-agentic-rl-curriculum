# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the engine-agnostic digest builder (migrated from the SONIC
digest_builder; TRAIN_SCALAR_KEYS is now injected, not hardcoded)."""

import math

from core.digest import (
    DEFAULT_TRAIN_SCALAR_KEYS,
    build_digest,
    build_train_section,
    cap_saturation_fraction,
    normalized_entropy,
    read_jsonl,
    summarize_series,
    top_k_share,
)


# ── stats helpers ─────────────────────────────────────────────────────
def test_summarize_series_trend_rising():
    s = summarize_series([1.0, 2.0, 3.0, 4.0], window=4)
    assert s["last"] == 4.0 and s["trend"] == "rising"
    assert s["n_points"] == 4 and math.isclose(s["slope_recent"], 1.0)


def test_summarize_series_empty():
    s = summarize_series([], window=5)
    assert s == {"last": None, "mean_recent": None, "slope_recent": None,
                 "trend": "unknown", "n_points": 0}


def test_normalized_entropy_bounds():
    assert normalized_entropy([1.0, 1.0, 1.0, 1.0]) == 1.0
    assert normalized_entropy([1.0, 0.0, 0.0]) == 0.0
    assert normalized_entropy([]) is None
    assert normalized_entropy([0.0, 0.0]) is None


def test_cap_saturation_fraction():
    # mean = 1.0, max_over_mean = 2 -> cap = 2.0; one of four bins >= cap
    assert cap_saturation_fraction([0.0, 1.0, 1.0, 2.0], 2.0) == 0.25
    assert cap_saturation_fraction([], 2.0) is None
    assert cap_saturation_fraction([0.0, 0.0], 2.0) == 0.0


def test_top_k_share():
    assert top_k_share([1.0, 1.0, 2.0], k=1) == 0.5
    assert top_k_share([], k=3) is None


def test_read_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"it": 1}\n\n{"it": 2}\n')
    assert read_jsonl(str(p)) == [{"it": 1}, {"it": 2}]


# ── injection seam (the de-SONIC-ification) ───────────────────────────
def test_default_train_scalar_keys_are_engine_neutral():
    # no SONIC trainer keys (policy/approxkl_avg etc.) baked in
    assert set(DEFAULT_TRAIN_SCALAR_KEYS) == {"Episode/rew_mean",
                                              "Episode/len_mean"}


def test_train_scalar_keys_injected():
    recs = [{"it": i, "tao/loss": 1.0 / (i + 1), "Episode/rew_mean": 0.1 * i}
            for i in range(5)]
    section = build_train_section(recs, window=5,
                                  train_scalar_keys=("tao/loss",))
    assert "tao/loss" in section
    assert "Episode/rew_mean" not in section  # only injected keys summarized


def test_custom_prefixes_injected():
    recs = [{"it": 1, "Rew/track": 0.5, "Term/fall": 0.1, "Sched/lr": 3e-4}]
    section = build_train_section(
        recs, window=5, train_scalar_keys=(),
        episode_prefix="Rew/", termination_prefix="Term/",
        scheduled_prefix="Sched/")
    assert section["episode_terms_last"] == {"track": 0.5}
    assert section["termination_terms_last"] == {"fall": 0.1}
    assert section["scheduled_params_last"] == {"lr": 3e-4}


# ── full digest ───────────────────────────────────────────────────────
def _train(n=6):
    return [{"it": i + 1, "Episode/rew_mean": 0.1 * i,
             "Episode/len_mean": 10.0 + i,
             "Episode_Termination/fall": 0.2}
            for i in range(n)]


def _evals():
    return [{"it": 3, "success_rate": 0.0, "progress_rate": 0.08,
             "failed_keys": ["a", "b"]},
            {"it": 6, "success_rate": 0.0, "progress_rate": 0.09,
             "failed_keys": ["b", "c"], "heldout_success_rate": 0.0}]


def test_build_digest_shape():
    d = build_digest(train_records=_train(), eval_records=_evals(),
                     sampler_records=[{"it": 6, "failure_rate": [0.1, 0.9]}],
                     knob_state={"mix_rate": {"value": 0.1,
                                              "ticks_since_change": None}},
                     decision_history=[{"tick": 1, "decision": {"action": "none"}}],
                     max_over_mean=2.0, window=5)
    assert d["schema_version"] == "0.1.0"
    assert d["last_iteration"] == 6
    assert d["train"]["Episode/rew_mean"]["trend"] == "rising"
    assert d["train"]["termination_terms_last"] == {"fall": 0.2}
    assert d["eval"]["n_evals"] == 2
    assert d["eval"]["failed_keys"]["newly_failing"] == ["c"]
    assert d["eval"]["failed_keys"]["newly_recovered"] == ["a"]
    assert d["sampler"]["num_bins"] == 2
    assert d["knobs"]["mix_rate"]["value"] == 0.1
    assert len(d["decision_history"]) == 1


def test_build_digest_empty_streams():
    d = build_digest()
    assert d["eval"] is None and d["sampler"] is None and d["train"] is None
    assert d["last_iteration"] is None
    assert d["decision_history"] == []


def test_sampler_section_requires_failure_rate_vectors():
    d = build_digest(sampler_records=[{"it": 1, "mean": 0.2}])
    assert d["sampler"] is None
