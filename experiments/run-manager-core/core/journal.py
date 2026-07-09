# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Journal entry construction / append / load for the run-manager core.

Field names and ordering are byte-compatible with the existing Phase-2
journals (e.g. experiments/curriculum-manager-phase2/
control_journal_v4_seed42.json): each per-segment entry carries, in
insertion order,

    tick, segment, knobs_in, rew_mean_last, len_mean_last,
    [config_verify], [eval], [eval_error], [heldout], [heldout_error],
    [checkpoint_purge], [tripwire_note], [event], [restored],
    [validation], [applied], [outcome], [digest_hash], [applied_at_iter],
    decision

matching how SmokeDriver.run() assembles entries
(smoke_driver.py:603-608 base fields, :514-518 config_verify,
:635-642 eval, :663-666 heldout, :744/:756/:764 decision) and how the
file is written (`json.dump(driver.journal, f, indent=2)`,
smoke_driver.py:891-892 — Python dicts preserve insertion order, so key
order in the file follows insertion order here).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# canonical ordering of the optional per-segment fields, as produced by the
# original driver's control flow (base fields first, decision last)
JOURNAL_ENTRY_FIELD_ORDER = (
    "tick", "segment", "knobs_in", "rew_mean_last", "len_mean_last",
    "config_verify", "eval", "eval_error", "heldout", "heldout_error",
    "checkpoint_purge", "tripwire_note", "event", "restored",
    "validation", "applied", "outcome", "digest_hash", "applied_at_iter",
    "decision",
)


def build_segment_entry(
    tick: int,
    segment: str,
    knobs_in: Dict[str, Any],
    rew_mean_last: Optional[float] = None,
    len_mean_last: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """One per-segment journal entry with the byte-compatible base fields
    (smoke_driver.py:603-608). `knobs_in` is copied — the caller keeps
    mutating its live knob dict after the entry is built.

    `extra` (config_verify / eval / heldout / decision / ...) is inserted
    in the canonical field order regardless of kwargs order, so serialized
    output is stable; unknown keys land after the known ones, preserving
    their given order.
    """
    entry: Dict[str, Any] = {
        "tick": tick,
        "segment": segment,
        "knobs_in": dict(knobs_in),
        "rew_mean_last": rew_mean_last,
        "len_mean_last": len_mean_last,
    }
    remaining = dict(extra)
    for key in JOURNAL_ENTRY_FIELD_ORDER:
        if key in remaining:
            entry[key] = remaining.pop(key)
    entry.update(remaining)
    return entry


def build_event_entry(tick: int, segment: str, event: str,
                      **fields: Any) -> Dict[str, Any]:
    """A non-segment lifecycle event (e.g. `segment_failed`
    smoke_driver.py:582-584, `disk_gate_failed` :532-538)."""
    return {"tick": tick, "segment": segment, "event": event, **fields}


def append_entry(journal: List[Dict[str, Any]],
                 entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Append in place and return the journal (mirrors the driver's
    `self.journal.append(entry)`)."""
    journal.append(entry)
    return journal


def save_journal(journal: List[Dict[str, Any]], path: str) -> None:
    """Write exactly as the original driver does
    (smoke_driver.py:891-892): `json.dump(journal, f, indent=2)`."""
    with open(path, "w") as f:
        json.dump(journal, f, indent=2)


def load_journal(path: str) -> List[Dict[str, Any]]:
    """Load a journal file; validates the top-level shape (a list of
    dict entries) but never rewrites field names or ordering."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        raise ValueError(f"{path}: journal must be a JSON list of entry objects")
    return data
