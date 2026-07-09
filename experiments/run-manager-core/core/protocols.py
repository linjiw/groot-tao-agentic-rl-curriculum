# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Engine-agnostic protocols for the run-manager core.

The `EngineAdapter` call surface is NOT invented: every method mirrors an
actual call the Phase-2 SmokeDriver makes on its adapter
(experiments/curriculum-manager-phase2/smoke_driver.py), with signatures
taken from the SONIC JobAdapter implementation
(skills/agentic/sonic-job-adapter/job_adapter.py). Call-site provenance:

- launch_segment      smoke_driver.py:577-578
- wait                smoke_driver.py:579
- parse_segment       smoke_driver.py:580
- eval_segment        smoke_driver.py:626-633 (standard), 655-660 (held-out,
                      with out_suffix/raw/extra_overrides)
- resolved_config_text smoke_driver.py:501-504 (optional seam: the driver
                      probes it with getattr and skips config verification
                      when absent)
- knob_to_config_path smoke_driver.py:513 — the ONLY hard coupling in the
                      old driver (module-level job_adapter.KNOB_TO_CONFIG_PATH);
                      lifted onto the adapter surface so the core never
                      imports an engine module.

The `Policy` surface mirrors smoke_driver.py:688-689 (observe, optional)
and smoke_driver.py:763 (propose).
"""

from __future__ import annotations

import dataclasses
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)


# ── dataclasses ───────────────────────────────────────────────────────
@dataclasses.dataclass
class Segment:
    """One knob-constant stretch of training (the manager's unit of change).

    Field set matches job_adapter.Segment — the driver reads .status
    (smoke_driver.py:581), .snapshot (:599) and .experiment_dir (:554).
    """

    name: str
    iterations: int
    knobs: Dict[str, Any]
    checkpoint_in: Optional[str] = None   # None = fresh start
    log_path: str = ""
    experiment_dir: Optional[str] = None
    snapshot: Optional[str] = None        # rollback point for this segment
    status: str = "pending"               # pending|running|done|failed


@dataclasses.dataclass
class ParsedSegment:
    """Parsed metric streams of one finished segment.

    Field set matches job_adapter.ParsedLog — the driver reads .train,
    .sampler (smoke_driver.py:591-598) and .tracebacks (:584).
    """

    train: List[dict] = dataclasses.field(default_factory=list)
    sampler: List[dict] = dataclasses.field(default_factory=list)
    checkpoint_loaded_step: Optional[int] = None
    experiment_dir: Optional[str] = None
    tracebacks: int = 0


@dataclasses.dataclass
class Tripwire:
    """Machine-readable rollback condition attached to a 'set' decision
    (shape from knob_registry.REQUIRED_TRIPWIRE_FIELDS and the driver's
    armed-watch dict, smoke_driver.py:694-714)."""

    metric: str
    drop_pct: float
    evals: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Decision:
    """One per-tick manager decision.

    The journal/validator consume the plain-dict shape
    ({"action": "none"} or {"action": "set", "knob": ..., "value": ...,
    "rationale": ..., "expected_effect": ..., "tripwire": {...}});
    `to_dict()` produces exactly that, dropping None-valued optionals so
    an {"action": "none"} decision serializes byte-compatibly with the
    existing journals.
    """

    action: str                              # "none" | "set"
    knob: Optional[str] = None
    value: Optional[Any] = None
    rationale: Optional[str] = None
    expected_effect: Optional[str] = None
    tripwire: Optional[Tripwire] = None
    reason: Optional[str] = None             # used by action == "none"
    expected_effect_check: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"action": self.action}
        for field in ("knob", "value", "rationale", "expected_effect",
                      "reason", "expected_effect_check"):
            v = getattr(self, field)
            if v is not None:
                out[field] = v
        if self.tripwire is not None:
            out["tripwire"] = self.tripwire.to_dict()
        return out


# ── protocols ─────────────────────────────────────────────────────────
@runtime_checkable
class EngineAdapter(Protocol):
    """The engine seam: everything the run-manager loop needs from a
    training backend. Six methods — the exact call surface of
    SmokeDriver.run() plus the one module-level constant it reached for
    (KNOB_TO_CONFIG_PATH), lifted onto the instance."""

    def launch_segment(self, name: str, iterations: int,
                       knobs: Dict[str, Any],
                       checkpoint_in: Optional[str] = None) -> Segment:
        """Start one training segment. (smoke_driver.py:577-578)"""
        ...

    def wait(self, seg: Segment, poll_s: int = 20,
             timeout_s: int = 3600) -> Segment:
        """Block until `seg` finishes; sets status/experiment_dir/snapshot.
        (smoke_driver.py:579)"""
        ...

    def parse_segment(self, seg: Segment) -> ParsedSegment:
        """Turn the segment's log into train/sampler records.
        (smoke_driver.py:580)"""
        ...

    def eval_segment(self, seg: Segment, it: int,
                     num_envs: int = 64,
                     extra_overrides: Optional[List[str]] = None,
                     out_suffix: str = "_eval",
                     raw: bool = False) -> Dict[str, Any]:
        """Run an eval-only pass on the segment's snapshot; one eval-stream
        record (or the raw metrics dict when raw=True).
        (smoke_driver.py:626-633, 655-660)"""
        ...

    def resolved_config_text(self, seg: Segment) -> Optional[str]:
        """Raw text of the segment's resolved config, or None when absent.
        (smoke_driver.py:501-504)"""
        ...

    def knob_to_config_path(self) -> Dict[str, str]:
        """Knob name -> dotted path into the resolved config — what
        KnobRegistry.verify_against_config walks. Replaces the old
        module-level job_adapter.KNOB_TO_CONFIG_PATH coupling
        (smoke_driver.py:513)."""
        ...


@runtime_checkable
class Policy(Protocol):
    """The decision-maker seam (smoke_driver.py:688-689, :763)."""

    def propose(self, digest: Dict[str, Any], state: Any,
                registry: Any) -> Dict[str, Any]:
        """One decision dict per tick ({"action": "none"} is the default)."""
        ...


@runtime_checkable
class ObservingPolicy(Policy, Protocol):
    """Optional extension: policies that keep per-tick history. The driver
    probes for `observe` with hasattr (smoke_driver.py:688)."""

    def observe(self, digest: Dict[str, Any]) -> None:
        ...
