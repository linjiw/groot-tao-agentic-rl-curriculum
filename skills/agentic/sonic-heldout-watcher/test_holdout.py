# SPDX-License-Identifier: Apache-2.0
"""Tests for the held-out watcher core (design doc 08 axiom 5)."""

import json

import pytest

from holdout import (
    curriculum_keys,
    heldout_record_from_metrics_eval,
    load_manifest,
    select_holdout,
    write_manifest,
)

KEYS = [f"motion_{i:04d}" for i in range(500)]


# ── split determinism & stability ────────────────────────────────────
def test_split_is_deterministic():
    a = select_holdout(KEYS, 0.1, salt="s1")
    b = select_holdout(KEYS, 0.1, salt="s1")
    assert a == b


def test_split_depends_on_salt():
    a = select_holdout(KEYS, 0.1, salt="s1")
    b = select_holdout(KEYS, 0.1, salt="s2")
    assert a["heldout"] != b["heldout"]


def test_split_fraction_approximate():
    split = select_holdout(KEYS, 0.1, salt="s1")
    assert 25 <= len(split["heldout"]) <= 80  # ~50 of 500, hash variance


def test_membership_stable_under_library_growth():
    """A key keeps its side when other keys are added/removed."""
    small = select_holdout(KEYS[:100], 0.2, salt="s1")
    big = select_holdout(KEYS, 0.2, salt="s1")
    assert set(small["heldout"]) == set(big["heldout"]) & set(KEYS[:100])


def test_degenerate_and_invalid_inputs():
    with pytest.raises(ValueError, match="fraction"):
        select_holdout(KEYS, 0.0, salt="s")
    with pytest.raises(ValueError, match="salt"):
        select_holdout(KEYS, 0.1, salt="")
    with pytest.raises(ValueError, match="duplicate"):
        select_holdout(["a", "a"], 0.5, salt="s")
    with pytest.raises(ValueError, match="degenerate"):
        select_holdout(["a"], 0.5, salt="s")  # one key can't split


# ── manifest ─────────────────────────────────────────────────────────
def test_manifest_roundtrip_and_integrity(tmp_path):
    p = tmp_path / "manifest.json"
    write_manifest(str(p), KEYS, 0.1, salt="s1")
    m = load_manifest(str(p))
    assert m["heldout_keys"] == select_holdout(KEYS, 0.1, "s1")["heldout"]

    # tampering (e.g. the manager removing a hard motion) is detected
    m["heldout_keys"] = m["heldout_keys"][:-1]
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="integrity"):
        load_manifest(str(p))


def test_curriculum_keys_excludes_heldout_and_new_heldout_side(tmp_path):
    p = tmp_path / "m.json"
    manifest = write_manifest(str(p), KEYS[:400], 0.1, salt="s1")
    grown = KEYS  # 100 new keys appear later
    cur = curriculum_keys(manifest, grown)
    assert not set(cur) & set(manifest["heldout_keys"])
    # new keys that hash held-out-side are ALSO excluded from curriculum
    full_split = select_holdout(grown, 0.1, salt="s1")
    assert not set(cur) & set(full_split["heldout"])
    # and curriculum + full-heldout covers everything
    assert set(cur) | set(full_split["heldout"]) == set(grown)


# ── protected-record production ──────────────────────────────────────
@pytest.fixture()
def manifest(tmp_path):
    return write_manifest(str(tmp_path / "m.json"), KEYS, 0.1, salt="s1")


def test_record_from_success_scalar(manifest):
    rec = heldout_record_from_metrics_eval(
        {"eval/success/success_rate": 0.83}, manifest, it=5000
    )
    assert rec["heldout_success_rate"] == 0.83
    assert rec["it"] == 5000
    assert rec["heldout_manifest_integrity"] == manifest["integrity"]


def test_record_derived_from_failed_keys(manifest):
    failed = manifest["heldout_keys"][:5]
    rec = heldout_record_from_metrics_eval({"failed_keys": failed}, manifest, it=1)
    n = len(manifest["heldout_keys"])
    assert rec["heldout_success_rate"] == pytest.approx(1 - 5 / n)
    assert rec["heldout_failed_count"] == 5


def test_record_refuses_foreign_failed_keys(manifest):
    """Eval ran on the wrong subset → refuse, never emit a wrong metric."""
    with pytest.raises(ValueError, match="outside the held-out subset"):
        heldout_record_from_metrics_eval(
            {"eval/success/success_rate": 0.9,
             "failed_keys": ["not_a_heldout_motion"]},
            manifest, it=1,
        )


def test_record_refuses_empty_and_out_of_range(manifest):
    with pytest.raises(ValueError, match="neither"):
        heldout_record_from_metrics_eval({}, manifest, it=1)
    with pytest.raises(ValueError, match="out of range"):
        heldout_record_from_metrics_eval(
            {"eval/success/success_rate": 1.7}, manifest, it=1
        )


def test_record_surfaces_resolving_progress_metric(manifest):
    """doc 10 I2.1: success_rate is 0.0-everywhere at scale; the per-motion
    PROGRESS aggregate + spread is the RESOLVING held-out metric. It must be
    surfaced from eval/all_metrics_dict.progress + eval/success/progress_rate."""
    hk = manifest["heldout_keys"]
    prog = [0.0, 0.1, 0.32, 0.05, 0.0]          # continuous, non-degenerate
    metrics = {
        "eval/success/success_rate": 0.0,        # the useless metric
        "eval/success/progress_rate": 0.094,     # the resolving scalar
        "eval/all_metrics_dict": {
            "motion_keys": hk[:5],
            "progress": prog,
        },
    }
    rec = heldout_record_from_metrics_eval(metrics, manifest, it=100)
    assert rec["heldout_success_rate"] == 0.0
    assert rec["heldout_progress_rate"] == 0.094
    pm = rec["heldout_progress_per_motion"]
    assert pm["n"] == 5
    assert pm["max"] == 0.32
    assert pm["nonzero"] == 3
    assert pm["mean"] == pytest.approx(sum(prog) / 5, abs=1e-6)


def test_record_refuses_foreign_per_motion_keys(manifest):
    """Per-motion progress keyed outside the held-out subset must refuse,
    same integrity guard as failed_keys — a wrong motion set can't feed a
    wrong resolving metric either."""
    with pytest.raises(ValueError, match="outside the subset"):
        heldout_record_from_metrics_eval(
            {"eval/success/success_rate": 0.0,
             "eval/all_metrics_dict": {
                 "motion_keys": ["not_a_heldout_motion", "also_foreign"],
                 "progress": [0.1, 0.2]}},
            manifest, it=1)


def test_record_progress_metric_optional(manifest):
    """When no per-motion progress is present, the record still emits (the
    metric is additive, not required) — back-compat with old eval files."""
    rec = heldout_record_from_metrics_eval(
        {"eval/success/success_rate": 0.5}, manifest, it=1)
    assert "heldout_progress_per_motion" not in rec
    assert "heldout_progress_rate" not in rec


def test_record_feeds_digest_builder(manifest, tmp_path):
    """End-to-end: watcher record → digest builder sees the protected metric."""
    import importlib.util, os, sys
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    spec = importlib.util.spec_from_file_location(
        "digest_builder",
        os.path.join(repo, "skills/agentic/sonic-run-digest/digest_builder.py"),
    )
    db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db)

    recs = [
        heldout_record_from_metrics_eval(
            {"eval/success/success_rate": 0.6 + 0.05 * i}, manifest, it=250 * (i + 1)
        )
        for i in range(4)
    ]
    digest = db.build_digest(eval_records=recs)
    hs = digest["eval"]["heldout_success_rate"]
    assert hs["last"] == pytest.approx(0.75)
    assert hs["trend"] == "rising"


# ── CLI ──────────────────────────────────────────────────────────────
def test_cli_end_to_end(tmp_path):
    from holdout import main

    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("\n".join(KEYS))
    manifest_path = tmp_path / "manifest.json"
    assert main(["make-manifest", "--keys-file", str(keys_file),
                 "--fraction", "0.1", "--salt", "s1",
                 "--out", str(manifest_path)]) == 0

    me = tmp_path / "metrics_eval.json"
    me.write_text(json.dumps({"eval/success/success_rate": 0.77}))
    out = tmp_path / "eval.jsonl"
    assert main(["record", "--metrics-eval", str(me),
                 "--manifest", str(manifest_path), "--it", "750",
                 "--append-to", str(out)]) == 0
    rec = json.loads(out.read_text().strip())
    assert rec["heldout_success_rate"] == 0.77 and rec["it"] == 750
