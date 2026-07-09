# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for journal entry construction / append / save / load.

Byte-compatibility oracle: the real Phase-2 journal
experiments/curriculum-manager-phase2/control_journal_v4_seed42.json —
field names AND top-level key order per entry must match what
SmokeDriver.run() + json.dump(indent=2) produced."""

import json
import os

import pytest

from core.journal import (
    JOURNAL_ENTRY_FIELD_ORDER,
    append_entry,
    build_event_entry,
    build_segment_entry,
    load_journal,
    save_journal,
)

REAL_JOURNAL = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "curriculum-manager-phase2", "control_journal_v4_seed42.json")


def test_base_entry_fields_and_order():
    e = build_segment_entry(tick=1, segment="control_s1",
                            knobs_in={"threshold.a": 0.15},
                            rew_mean_last=0.89844, len_mean_last=13.18)
    assert list(e) == ["tick", "segment", "knobs_in",
                       "rew_mean_last", "len_mean_last"]
    assert e["tick"] == 1 and e["segment"] == "control_s1"


def test_knobs_in_is_copied():
    knobs = {"threshold.a": 0.15}
    e = build_segment_entry(1, "s1", knobs)
    knobs["threshold.a"] = 0.2  # driver mutates its live dict later
    assert e["knobs_in"]["threshold.a"] == 0.15


def test_extra_fields_follow_canonical_order():
    # kwargs given out of order still serialize in the driver's order
    e = build_segment_entry(1, "s1", {}, decision={"action": "none"},
                            heldout={"heldout_success_rate": 0.0},
                            config_verify={"status": "ok"},
                            eval={"success_rate": 0.0})
    tail = [k for k in e if k not in
            ("tick", "segment", "knobs_in", "rew_mean_last", "len_mean_last")]
    assert tail == ["config_verify", "eval", "heldout", "decision"]


def test_unknown_extra_fields_appended_after_known():
    e = build_segment_entry(1, "s1", {}, decision={"action": "none"},
                            custom_field=42)
    keys = list(e)
    assert keys.index("decision") < keys.index("custom_field")


def test_event_entry_shape():
    e = build_event_entry(3, "manager_s3", "segment_failed", tracebacks=2)
    assert e == {"tick": 3, "segment": "manager_s3",
                 "event": "segment_failed", "tracebacks": 2}


def test_append_entry_in_place():
    j = []
    out = append_entry(j, {"tick": 1})
    assert out is j and j == [{"tick": 1}]


def test_save_load_roundtrip(tmp_path):
    j = [build_segment_entry(1, "s1", {"k": 1.0}, 0.5, 10.0,
                             decision={"action": "none", "reason": "control arm"})]
    p = tmp_path / "journal.json"
    save_journal(j, str(p))
    assert load_journal(str(p)) == j
    # exact serialization contract of the original driver
    assert p.read_text() == json.dumps(j, indent=2)


def test_load_rejects_non_list(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"tick": 1}')
    with pytest.raises(ValueError):
        load_journal(str(p))


# ── byte-compatibility against the REAL v4 journal ────────────────────
@pytest.mark.skipif(not os.path.exists(REAL_JOURNAL),
                    reason="real Phase-2 journal not present")
def test_real_journal_loads_and_field_names_match():
    entries = load_journal(REAL_JOURNAL)
    assert entries, "journal empty"
    for e in entries:
        # every key the real driver emitted is a known canonical field
        assert set(e) <= set(JOURNAL_ENTRY_FIELD_ORDER), sorted(
            set(e) - set(JOURNAL_ENTRY_FIELD_ORDER))
        for base in ("tick", "segment", "knobs_in",
                     "rew_mean_last", "len_mean_last", "decision"):
            assert base in e


@pytest.mark.skipif(not os.path.exists(REAL_JOURNAL),
                    reason="real Phase-2 journal not present")
def test_rebuilt_entry_serializes_byte_identical_to_real_first_entry():
    entries = load_journal(REAL_JOURNAL)
    real = entries[0]
    rebuilt = build_segment_entry(
        tick=real["tick"], segment=real["segment"],
        knobs_in=real["knobs_in"],
        rew_mean_last=real["rew_mean_last"],
        len_mean_last=real["len_mean_last"],
        **{k: v for k, v in real.items()
           if k not in ("tick", "segment", "knobs_in",
                        "rew_mean_last", "len_mean_last")})
    assert list(rebuilt) == list(real)
    assert (json.dumps(rebuilt, indent=2)
            == json.dumps(real, indent=2))


@pytest.mark.skipif(not os.path.exists(REAL_JOURNAL),
                    reason="real Phase-2 journal not present")
def test_full_real_journal_roundtrips_byte_identical(tmp_path):
    entries = load_journal(REAL_JOURNAL)
    p = tmp_path / "roundtrip.json"
    save_journal(entries, str(p))
    with open(REAL_JOURNAL) as f:
        original = f.read()
    assert p.read_text() == original
