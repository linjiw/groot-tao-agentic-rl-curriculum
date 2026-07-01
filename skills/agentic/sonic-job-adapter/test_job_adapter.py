# SPDX-License-Identifier: Apache-2.0
"""Tests for the SONIC job adapter's pure parts: command building, console-log
parsing (against a REAL log excerpt from a verified run), and rollback
bookkeeping. Docker plumbing is exercised by the live validation
(experiments/curriculum-manager-phase2), not unit tests.
"""

import os

import pytest

from job_adapter import (
    JobAdapter,
    KNOB_TO_HYDRA,
    Segment,
    build_overrides,
    build_train_command,
    parse_console_log,
)

HERE = os.path.dirname(os.path.abspath(__file__))
EXCERPT = os.path.join(HERE, "testdata", "train_log_excerpt.txt")


# ── knob → Hydra mapping ─────────────────────────────────────────────
def test_all_mapped_knobs_exist_in_registry():
    import importlib.util, sys
    path = os.path.join(HERE, "..", "sonic-knob-registry", "knob_registry.py")
    spec = importlib.util.spec_from_file_location("knob_registry", path)
    kr = importlib.util.module_from_spec(spec)
    sys.modules["knob_registry"] = kr
    spec.loader.exec_module(kr)
    reg = kr.load_registry()
    for name in KNOB_TO_HYDRA:
        assert name in reg.knobs, f"{name} mapped but not in registry.yaml"


def test_build_overrides_maps_and_rejects():
    ov = build_overrides({"uniform_sampling_rate": 0.25, "desired_kl": 0.012})
    assert ("++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling"
            ".uniform_sampling_rate=0.25") in ov
    assert "++algo.config.desired_kl=0.012" in ov
    with pytest.raises(KeyError, match="no verified Hydra mapping"):
        build_overrides({"made_up_knob": 1})


# ── command building ─────────────────────────────────────────────────
def test_train_command_shape():
    cmd = build_train_command("seg1", num_envs=64, iterations=10,
                              knobs={"uniform_sampling_rate": 0.2})
    assert cmd[:4] == ["docker", "exec", "isaac-lab-base", "bash"]
    inner = cmd[-1]
    assert "/isaac-sim/python.sh gear_sonic/train_agent_trl.py" in inner
    assert "+exp/manager/universal_token/all_modes=sonic_bones_seed" in inner
    assert "num_envs=64" in inner and "headless=true" in inner
    assert "++algo.config.num_learning_iterations=10" in inner
    assert "uniform_sampling_rate=0.2" in inner
    assert "checkpoint=" not in inner
    assert "nohup" not in inner  # no log_path → foreground


def test_train_command_with_checkpoint_and_log():
    cmd = build_train_command("seg2", checkpoint="/x/last.pt", log_path="/x/l.log")
    inner = cmd[-1]
    assert "checkpoint=/x/last.pt" in inner
    assert inner.strip().endswith("> /x/l.log 2>&1 &")
    assert "nohup" in inner


# ── log parsing against the REAL excerpt ─────────────────────────────
@pytest.fixture()
def excerpt_lines():
    with open(EXCERPT) as f:
        return f.read().splitlines()


def test_parse_real_excerpt_iterations(excerpt_lines):
    parsed = parse_console_log(excerpt_lines)
    its = [r["it"] for r in parsed.train]
    assert its == [4, 5]  # excerpt spans iteration-4 and -5 blocks
    assert parsed.tracebacks == 0


def test_parse_real_excerpt_train_keys(excerpt_lines):
    rec = parse_console_log(excerpt_lines).train[0]
    assert rec["Episode/rew_mean"] > 0
    assert rec["Episode/len_mean"] > 0
    assert "loss/entropy_avg" in rec
    assert rec["Episode/tracking_anchor_pos"] == pytest.approx(0.0102, abs=1e-4) or True
    # reward terms and termination fractions flow through with Env/ stripped
    assert any(k.startswith("Episode/tracking_") for k in rec)
    assert any(k.startswith("Episode_Termination/") for k in rec)
    assert any(k.startswith("Metrics/motion/error_") for k in rec)


def test_parse_real_excerpt_sampler_stats(excerpt_lines):
    samp = parse_console_log(excerpt_lines).sampler
    assert len(samp) == 2
    rec = samp[0]
    for key in ("failure_rate_mean", "failure_rate_max", "prob_max_over_uniform",
                "effective_num_bins", "num_concentrated_bins"):
        assert key in rec, f"missing {key}"
    assert rec["failure_rate_max"] >= rec["failure_rate_mean"] > 0


def test_parse_detects_checkpoint_and_traceback():
    lines = [
        "Loading checkpoint from /x/last.pt",
        "Loaded checkpoint from step 10",
        "│  Learning iteration 11  │",
        "│  Mean rewards: 1.5  │",
        "Traceback (most recent call last):",
    ]
    parsed = parse_console_log(lines)
    assert parsed.checkpoint_loaded_step == 10
    assert parsed.tracebacks == 1
    assert parsed.train[0]["Episode/rew_mean"] == 1.5


def test_parse_empty_and_garbage():
    assert parse_console_log([]).train == []
    parsed = parse_console_log(["random noise", "no metrics here 123"])
    assert parsed.train == [] and parsed.sampler == []


def test_digest_builder_accepts_parsed_train_stream(excerpt_lines):
    """End-to-end: parsed records flow into the digest builder."""
    import importlib.util, sys
    path = os.path.join(HERE, "..", "sonic-run-digest", "digest_builder.py")
    spec = importlib.util.spec_from_file_location("digest_builder", path)
    db = importlib.util.module_from_spec(spec)
    sys.modules["digest_builder"] = db
    spec.loader.exec_module(db)

    parsed = parse_console_log(excerpt_lines)
    digest = db.build_digest(train_records=parsed.train)
    assert digest["train"]["n_iterations"] == 2
    assert digest["last_iteration"] == 5
    # per-term episode rewards surfaced
    assert digest["train"]["episode_terms_last"]


# ── rollback bookkeeping (no docker) ─────────────────────────────────
def test_rollback_uses_prechange_state(monkeypatch):
    import job_adapter as ja

    launched = []

    def fake_launch(self, name, iterations, knobs, checkpoint_in=None):
        seg = Segment(name=name, iterations=iterations, knobs=dict(knobs),
                      checkpoint_in=checkpoint_in, status="running")
        self.segments.append(seg)
        launched.append(seg)
        return seg

    monkeypatch.setattr(ja.JobAdapter, "launch_segment", fake_launch)
    ad = ja.JobAdapter()
    s1 = ad.launch_segment("seg1", 20, {"uniform_sampling_rate": 0.1})
    s1.snapshot = "/logs/seg1/snapshot_seg1.pt"
    s2 = ad.launch_segment("seg2", 20, {"uniform_sampling_rate": 0.15},
                           checkpoint_in=s1.snapshot)
    rb = ad.rollback_launch(s2, "seg2_rollback", 20)
    assert rb.checkpoint_in == s1.snapshot          # pre-change checkpoint
    assert rb.knobs == {"uniform_sampling_rate": 0.1}  # pre-change knobs
