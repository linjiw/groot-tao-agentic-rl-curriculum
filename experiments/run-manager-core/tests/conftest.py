# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Shared fixtures: put the package root on sys.path and provide a fully
engine-neutral in-memory registry spec (the core ships no default action
space, so tests inject their own)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from core.registry import KnobRegistry, RunState  # noqa: E402


@pytest.fixture()
def spec():
    """Engine-neutral action space covering every max_step kind, a
    'design' knob and a held-out-gated family."""
    return {
        "meta": {"global_rules": {"max_changes_per_tick": 1,
                                  "default_action": "none"}},
        "knobs": {
            "mix_rate": {
                "family": "data_curriculum", "type": "float",
                "default": 0.1, "hard_range": [0.05, 0.5],
                "max_step": {"kind": "multiplicative", "factor": 1.5},
                "cooldown_ticks": 2, "status": "available",
                "verified_source": "test",
            },
            "cap_ratio": {
                "family": "data_curriculum", "type": "float",
                "default": 50.0, "hard_range": [10.0, 500.0],
                "max_step": {"kind": "multiplicative", "factor": 2},
                "cooldown_ticks": 1, "status": "available",
                "verified_source": "test",
            },
            "threshold.a": {
                "family": "schedule", "type": "float",
                "default": 0.35, "hard_range": [0.05, 0.5],
                "max_step": {"kind": "additive", "step": 0.05},
                "cooldown_ticks": 1, "status": "available",
                "verified_source": "test",
            },
            "kl_target": {
                "family": "optimizer", "type": "float",
                "default": 0.01, "hard_range": [0.001, 0.05],
                "max_step": {"kind": "multiplicative", "factor": 1.5},
                "cooldown_ticks": 1, "status": "available",
                "verified_source": "test",
            },
            "bin_size": {
                "family": "data_curriculum", "type": "choice",
                "default": 50, "choices": [10, 25, 50, 100],
                "max_step": {"kind": "notch"},
                "cooldown_ticks": 1, "status": "available",
                "restart_required": True,
                "verified_source": "test",
            },
            "push_scale": {
                "family": "optimizer", "type": "float",
                "default": 1.0, "hard_range": [0.5, 1.5],
                "max_step": {"kind": "additive", "step": 0.1},
                "cooldown_ticks": 1, "status": "design",
                "verified_source": "test",
            },
        },
    }


@pytest.fixture()
def reg(spec):
    return KnobRegistry(spec)


@pytest.fixture()
def state():
    return RunState(tick=10)
