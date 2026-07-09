# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the protocols: dataclass shapes, Protocol structural checks
(a fake TAO-ish adapter and the old SONIC JobAdapter surface must both
satisfy EngineAdapter), Decision serialization."""

from typing import Any, Dict, List, Optional

from core.protocols import (
    Decision,
    EngineAdapter,
    ObservingPolicy,
    ParsedSegment,
    Policy,
    Segment,
    Tripwire,
)


class FakeAdapter:
    """Engine-neutral fake with the exact SmokeDriver call surface."""

    def __init__(self):
        self.launched: List[Segment] = []

    def launch_segment(self, name: str, iterations: int,
                       knobs: Dict[str, Any],
                       checkpoint_in: Optional[str] = None) -> Segment:
        seg = Segment(name=name, iterations=iterations, knobs=dict(knobs),
                      checkpoint_in=checkpoint_in, status="running")
        self.launched.append(seg)
        return seg

    def wait(self, seg: Segment, poll_s: int = 20,
             timeout_s: int = 3600) -> Segment:
        seg.status = "done"
        seg.snapshot = f"/ckpt/{seg.name}.pt"
        return seg

    def parse_segment(self, seg: Segment) -> ParsedSegment:
        return ParsedSegment(train=[{"it": 1, "Episode/rew_mean": 0.5,
                                     "Episode/len_mean": 12.0}])

    def eval_segment(self, seg: Segment, it: int, num_envs: int = 64,
                     extra_overrides: Optional[List[str]] = None,
                     out_suffix: str = "_eval",
                     raw: bool = False) -> Dict[str, Any]:
        return {"it": it, "success_rate": 0.0, "progress_rate": 0.1}

    def resolved_config_text(self, seg: Segment) -> Optional[str]:
        return "env:\n  sampling:\n    mix_rate: 0.1\n"

    def knob_to_config_path(self) -> Dict[str, str]:
        return {"mix_rate": "env.sampling.mix_rate"}


class FakePolicy:
    def propose(self, digest, state, registry):
        return {"action": "none"}


class FakeObservingPolicy(FakePolicy):
    def __init__(self):
        self.seen = []

    def observe(self, digest):
        self.seen.append(digest)


# ── protocol conformance ─────────────────────────────────────────────
def test_fake_adapter_satisfies_engine_adapter():
    assert isinstance(FakeAdapter(), EngineAdapter)


def test_incomplete_adapter_rejected():
    class Partial:
        def launch_segment(self, *a, **k): ...
    assert not isinstance(Partial(), EngineAdapter)


def test_policy_protocols():
    assert isinstance(FakePolicy(), Policy)
    assert not isinstance(FakePolicy(), ObservingPolicy)
    assert isinstance(FakeObservingPolicy(), ObservingPolicy)


def test_adapter_lifecycle_flow():
    """The exact per-segment sequence SmokeDriver.run() performs
    (smoke_driver.py:577-580, 626, 501-504, 513)."""
    a = FakeAdapter()
    seg = a.launch_segment("arm_s1", 10, {"mix_rate": 0.1},
                           checkpoint_in=None)
    a.wait(seg, poll_s=10, timeout_s=3600)
    parsed = a.parse_segment(seg)
    assert seg.status == "done"
    assert parsed.train[0]["Episode/rew_mean"] == 0.5
    ev = a.eval_segment(seg, it=10, num_envs=64)
    assert ev["it"] == 10
    assert a.resolved_config_text(seg)
    assert "mix_rate" in a.knob_to_config_path()


# ── dataclasses ──────────────────────────────────────────────────────
def test_segment_defaults():
    seg = Segment(name="s1", iterations=10, knobs={})
    assert seg.status == "pending"
    assert seg.checkpoint_in is None and seg.snapshot is None


def test_parsed_segment_defaults():
    p = ParsedSegment()
    assert p.train == [] and p.sampler == [] and p.tracebacks == 0


def test_decision_none_serializes_like_journal():
    d = Decision(action="none", reason="control arm")
    assert d.to_dict() == {"action": "none", "reason": "control arm"}


def test_decision_set_serializes_full_shape():
    d = Decision(action="set", knob="mix_rate", value=0.15,
                 rationale="r", expected_effect="e",
                 tripwire=Tripwire(metric="eval/progress_rate",
                                   drop_pct=5, evals=3))
    out = d.to_dict()
    assert out["action"] == "set" and out["knob"] == "mix_rate"
    assert out["tripwire"] == {"metric": "eval/progress_rate",
                               "drop_pct": 5, "evals": 3}
    assert "reason" not in out  # None optionals are dropped
