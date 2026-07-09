# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Mock EngineAdapter / Policy implementations of the core protocols.

Promoted from the ad-hoc fake家族 in
experiments/curriculum-manager-phase2/test_smoke_driver.py (FakeAdapter
:16-45, FakeEvalAdapter :186-204, EffectAdapter :385-397,
ConfigFakeAdapter :459-475, PinCaptureAdapter :529-545, PurgeFakeAdapter
:640-652) — these fakes were already the de-facto adapter protocol; here
they implement core.protocols.EngineAdapter formally (Segment /
ParsedSegment instead of job_adapter.Segment / ParsedLog).

Also carries the two reference policies (TrainSideBandPolicy,
ScriptedPolicy + V4_MANAGER_LADDER), copied VERBATIM from
smoke_driver.py:152-316 — they are pure digest/state readers with no
engine imports, and the loop tests / journal-equivalence harness need
them byte-identical to the Phase-2 originals.

And `MOCK_REGISTRY_SPEC`: the subset of the SONIC registry.yaml action
space the old test suite exercised (same defaults / ranges / steps /
cooldowns), injected because the core ships no default action space.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import os
from typing import Any, Dict, List, Optional

from core.protocols import ParsedSegment, Segment

# ── registry spec (mirrors skills/agentic/sonic-knob-registry/
#    registry.yaml for the knobs the loop tests touch) ─────────────────
MOCK_REGISTRY_SPEC: Dict[str, Any] = {
    "meta": {"global_rules": {"max_changes_per_tick": 1,
                              "default_action": "none"}},
    "knobs": {
        "termination_threshold.anchor_pos": {
            "family": "schedule", "type": "float", "default": 0.30,
            "hard_range": [0.15, 0.5],
            "max_step": {"kind": "additive", "step": 0.05},
            "cooldown_ticks": 3, "status": "patch",
        },
        "termination_threshold.ee_body_pos": {
            "family": "schedule", "type": "float", "default": 0.30,
            "hard_range": [0.15, 0.5],
            "max_step": {"kind": "additive", "step": 0.05},
            "cooldown_ticks": 3, "status": "patch",
        },
        "termination_threshold.foot_pos_xyz": {
            "family": "schedule", "type": "float", "default": 0.35,
            "hard_range": [0.2, 0.5],
            "max_step": {"kind": "additive", "step": 0.05},
            "cooldown_ticks": 3, "status": "patch",
        },
        "desired_kl": {
            "family": "optimizer", "type": "float", "default": 0.01,
            "hard_range": [0.005, 0.02],
            "max_step": {"kind": "multiplicative", "factor": 1.5},
            "cooldown_ticks": 4, "status": "available",
        },
        "adp_samp_failure_rate_max_over_mean": {
            "family": "data_curriculum", "type": "float", "default": 50.0,
            "hard_range": [10.0, 500.0],
            "max_step": {"kind": "multiplicative", "factor": 2},
            "cooldown_ticks": 2, "status": "available",
        },
    },
}

# knob -> dotted resolved-config path (mirrors job_adapter.KNOB_TO_CONFIG_PATH
# for the knobs above; injected/adapter-provided, never imported by core)
MOCK_KNOB_CONFIG_PATHS: Dict[str, str] = {
    "termination_threshold.anchor_pos":
        "manager_env.terminations.anchor_pos.params.threshold",
    "termination_threshold.ee_body_pos":
        "manager_env.terminations.ee_body_pos.params.threshold",
    "termination_threshold.foot_pos_xyz":
        "manager_env.terminations.foot_pos_xyz.params.threshold",
    "desired_kl": "algo.config.desired_kl",
}


# ── adapters ──────────────────────────────────────────────────────────
class FakeAdapter:
    """Scripted segments: each entry is (len_mean, rew_mean).
    (test_smoke_driver.py:16-45)"""

    def __init__(self, script):
        self.script = list(script)
        self.launched = []  # (name, knobs, checkpoint_in)
        self.i = 0

    def launch_segment(self, name, iterations, knobs, checkpoint_in=None):
        self.launched.append((name, dict(knobs), checkpoint_in))
        return Segment(name=name, iterations=iterations,
                       knobs=dict(knobs), checkpoint_in=checkpoint_in,
                       status="running")

    def wait(self, seg, poll_s=0, timeout_s=0):
        seg.status = "done"
        seg.snapshot = f"/fake/{seg.name}/snapshot.pt"
        return seg

    def parse_segment(self, seg):
        ln, rew = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        train = [{"it": k + 1, "Episode/len_mean": ln, "Episode/rew_mean": rew,
                  "loss/entropy_avg": -45.0,
                  # foot_pos_xyz is the binding termination axis (like the real runs)
                  "Episode_Termination/foot_pos_xyz": 0.56,
                  "Episode_Termination/ee_body_pos": 0.42,
                  "Episode_Termination/anchor_pos": 0.0} for k in range(3)]
        sampler = [{"it": 2, "failure_rate_mean": 2.0, "effective_num_bins": 60.0}]
        return ParsedSegment(train=train, sampler=sampler)


class NoRewAdapter(FakeAdapter):
    """FakeAdapter minus the reward stream — the no-baseline refusal path.
    (test_smoke_driver.py:163-168, promoted from a test-local class)"""

    def parse_segment(self, seg):
        p = super().parse_segment(seg)
        for r in p.train:
            r.pop("Episode/rew_mean", None)
        return p


class FakeEvalAdapter(FakeAdapter):
    """FakeAdapter + scripted eval_segment: eval_script[i] = progress_rate
    for the i-th eval pass (None = eval failure).
    (test_smoke_driver.py:186-204)"""

    def __init__(self, script, eval_script):
        super().__init__(script)
        self.eval_script = list(eval_script)
        self.evals_run = []
        self.j = 0

    def eval_segment(self, seg, it, num_envs=64, poll_s=0, timeout_s=0):
        pr = self.eval_script[min(self.j, len(self.eval_script) - 1)]
        self.j += 1
        self.evals_run.append((seg.name, it))
        if pr is None:
            raise RuntimeError("scripted eval failure")
        return {"it": it, "success_rate": 0.0, "progress_rate": pr,
                "mpjpe_all_mean": 60.0, "mpjpe_pa_all_mean": 20.0,
                "failed_keys": ["m1", "m2"]}


class EffectAdapter(FakeAdapter):
    """Like FakeAdapter but episode length RESPONDS to the loosened knob:
    once any termination_threshold.* override is in the launch knobs,
    subsequent segments report long episodes.
    (test_smoke_driver.py:385-397)"""

    def parse_segment(self, seg):
        p = super().parse_segment(seg)
        loosened = any(k.startswith("termination_threshold.")
                       for k in self.launched[-1][1])
        if loosened:
            for r in p.train:
                r["Episode/len_mean"] = 80.0
        return p


class ConfigFakeAdapter(FakeAdapter):
    """FakeAdapter that also exposes the resolved-config seam.
    (test_smoke_driver.py:459-475; knob_to_config_path added — the seam
    that replaced the old module-level KNOB_TO_CONFIG_PATH coupling)"""

    def __init__(self, script, config_texts):
        super().__init__(script)
        self.config_texts = list(config_texts)  # one per segment; None = absent
        self.j = 0

    def wait(self, seg, poll_s=0, timeout_s=0):
        seg = super().wait(seg, poll_s, timeout_s)
        seg.experiment_dir = f"/fake/{seg.name}"
        return seg

    def resolved_config_text(self, seg):
        text = self.config_texts[min(self.j, len(self.config_texts) - 1)]
        self.j += 1
        return text

    def knob_to_config_path(self):
        return dict(MOCK_KNOB_CONFIG_PATHS)


class PinCaptureAdapter(FakeAdapter):
    """FakeAdapter + eval_segment that records extra_overrides per call.
    (test_smoke_driver.py:529-545)"""

    def __init__(self, script, n_evals):
        super().__init__(script)
        self.eval_calls = []  # (out_suffix, extra_overrides)
        self.n_evals = n_evals

    def eval_segment(self, seg, it, num_envs=64, poll_s=0, timeout_s=0,
                     extra_overrides=None, out_suffix="_eval", raw=False):
        self.eval_calls.append((out_suffix, list(extra_overrides or [])))
        rec = {"it": it, "success_rate": 0.5, "progress_rate": 0.04,
               "mpjpe_all_mean": 60.0, "mpjpe_pa_all_mean": 20.0,
               "failed_keys": []}
        if raw:
            return {"eval/success/success_rate": 0.5, "failed_keys": []}
        return rec


class PurgeFakeAdapter(FakeAdapter):
    """FakeAdapter whose wait() exposes a REAL host run dir (the seam the
    loop's purge hook reads). (test_smoke_driver.py:640-652)"""

    def __init__(self, script, run_dirs):
        super().__init__(script)
        self.run_dirs = list(run_dirs)

    def wait(self, seg, poll_s=0, timeout_s=0):
        seg = super().wait(seg, poll_s, timeout_s)
        seg.experiment_dir = self.run_dirs.pop(0) if self.run_dirs else None
        return seg


class FailingAdapter(PurgeFakeAdapter):
    """Segment never finishes (status 'failed').
    (test_smoke_driver.py:682-686, promoted from a test-local class)"""

    def wait(self, seg, poll_s=0, timeout_s=0):
        seg = super().wait(seg, poll_s, timeout_s)
        seg.status = "failed"
        return seg


# ── purge hook (engine-neutral file purge; the docker-exec fallback of
#    smoke_driver.purge_intermediate_checkpoints stays SONIC-side) ─────
CHECKPOINT_PURGE_PATTERNS = ("last.pt", "model_step_*.pt")


def simple_purge(run_dir: str) -> tuple:
    """Delete plain files matching CHECKPOINT_PURGE_PATTERNS directly
    inside `run_dir`; keep snapshot_*.pt and every subdirectory. Returns
    (deleted_names, bytes_freed). Mirrors the host-side half of
    smoke_driver.purge_intermediate_checkpoints (no container fallback)."""
    deleted: List[str] = []
    freed = 0
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue  # never touch directories (eval outputs)
        if not any(fnmatch.fnmatch(name, pat)
                   for pat in CHECKPOINT_PURGE_PATTERNS):
            continue
        size = os.path.getsize(path)
        os.remove(path)
        freed += size
        deleted.append(name)
    return deleted, freed


# ── reference policies (verbatim from smoke_driver.py:152-316) ────────
@dataclasses.dataclass
class TrainSideBandPolicy:
    """Episode-length band stepper (training-side decision signal).

    Targets the BINDING termination axis: the threshold knob whose term
    has the highest termination fraction in the digest. Tripwire:
    eval-side `eval/progress_rate` when the digest carries an eval
    section, else training-side `Episode/rew_mean`.
    (smoke_driver.py:152-260, byte-identical decision text)"""

    len_low: float = 20.0     # mean episode length below this = terminating too early
    len_high: float = 200.0   # above this = competent enough to tighten
    sustain: int = 2          # consecutive segments required
    notch: float = 0.05

    # digest termination-term name -> registry knob
    TERM_TO_KNOB = {
        "anchor_pos": "termination_threshold.anchor_pos",
        "ee_body_pos": "termination_threshold.ee_body_pos",
        "foot_pos_xyz": "termination_threshold.foot_pos_xyz",
    }

    def __post_init__(self):
        self._len_history: List[float] = []

    def observe(self, digest: Dict[str, Any]) -> None:
        """Update sustain history from a digest WITHOUT proposing."""
        train = digest.get("train") or {}
        len_stats = train.get("Episode/len_mean")
        if len_stats and len_stats.get("last") is not None:
            self._len_history.append(len_stats["last"])

    def _binding_knob(self, digest) -> tuple:
        """(knob, fraction) for the axis with highest WINDOWED termination
        mean."""
        train = digest.get("train") or {}
        terms = (train.get("termination_terms_mean_recent")
                 or train.get("termination_terms_last") or {})
        candidates = {t: f for t, f in terms.items()
                      if t in self.TERM_TO_KNOB and isinstance(f, (int, float))}
        if not candidates or max(candidates.values()) <= 0:
            return None, None
        top = max(candidates, key=candidates.get)
        return self.TERM_TO_KNOB[top], candidates[top]

    def _tripwire(self, digest: Dict[str, Any]) -> Dict[str, Any]:
        if digest.get("eval"):
            return {"metric": "eval/progress_rate", "drop_pct": 30, "evals": 2}
        return {"metric": "Episode/rew_mean", "drop_pct": 20, "evals": 2}

    def propose(self, digest: Dict[str, Any], state, registry) -> Dict[str, Any]:
        # history is maintained by observe(); propose() only reads it
        if not self._len_history:
            return {"action": "none", "reason": "no episode-length signal"}
        recent = self._len_history[-self.sustain:]
        full = len(recent) == self.sustain

        knob, frac = self._binding_knob(digest)
        if knob is None:
            return {"action": "none",
                    "reason": "no binding threshold-termination term in digest"}
        cur = float(registry.current_of(knob, state))
        lo, hi = (float(x) for x in registry.knobs[knob]["hard_range"])
        tripwire = self._tripwire(digest)
        guard = ("eval-side fixed-threshold guard" if tripwire["metric"].startswith("eval/")
                 else "training-side guard (no eval stream)")

        if full and all(v < self.len_low for v in recent) and cur + self.notch <= hi:
            return {
                "action": "set", "knob": knob,
                "value": round(cur + self.notch, 4),
                "rationale": (f"mean episode length {[round(v,1) for v in recent]} < "
                              f"{self.len_low} for {self.sustain} segments; {knob} is the "
                              f"binding termination axis (windowed mean {frac:.2f} of "
                              f"episodes; training-side signal; {guard}; no held-out here)"),
                "expected_effect": "episodes lengthen; eval progress_rate does not fall",
                "expected_effect_check": {"metric": "Episode/len_mean",
                                          "op": ">=", "value": self.len_low},
                "tripwire": tripwire,
            }
        if full and all(v > self.len_high for v in recent) and cur - self.notch >= lo:
            return {
                "action": "set", "knob": knob,
                "value": round(cur - self.notch, 4),
                "rationale": (f"mean episode length {[round(v,1) for v in recent]} > "
                              f"{self.len_high} for {self.sustain} segments "
                              f"(training-side signal; {guard})"),
                "expected_effect": "tracking precision demand rises; length dips then recovers",
                "expected_effect_check": {"metric": "Episode/len_mean",
                                          "op": "<=", "value": self.len_high},
                "tripwire": tripwire,
            }
        return {"action": "none",
                "reason": f"len {[round(v,1) for v in recent]} inside "
                          f"[{self.len_low}, {self.len_high}] band or sustain unmet"}


# the EXACT decision stream both v4 manager seeds walked: tick -> (knob, value)
V4_MANAGER_LADDER = {
    2: ("termination_threshold.foot_pos_xyz", 0.25),
    4: ("termination_threshold.ee_body_pos", 0.20),
    6: ("termination_threshold.foot_pos_xyz", 0.30),
    8: ("termination_threshold.ee_body_pos", 0.25),
    10: ("termination_threshold.foot_pos_xyz", 0.35),
}


@dataclasses.dataclass
class ScriptedPolicy:
    """Open-loop ladder replay (E1 ablation arm). Replays a fixed
    tick-indexed knob ladder UNCONDITIONALLY; the digest is read ONLY to
    pick the tripwire metric. Deliberately NO observe().
    (smoke_driver.py:277-316, byte-identical decision text)"""

    ladder: Dict[int, tuple] = dataclasses.field(
        default_factory=lambda: dict(V4_MANAGER_LADDER))

    def propose(self, digest: Optional[Dict[str, Any]], state, registry) -> Dict[str, Any]:
        rung = self.ladder.get(state.tick)
        if rung is None:
            return {"action": "none",
                    "reason": f"scripted ladder: no rung scheduled at tick {state.tick}"}
        knob, value = rung
        if (digest or {}).get("eval"):
            tripwire = {"metric": "eval/progress_rate", "drop_pct": 30, "evals": 2}
        else:
            tripwire = {"metric": "Episode/rew_mean", "drop_pct": 20, "evals": 2}
        return {
            "action": "set", "knob": knob, "value": value,
            "rationale": (f"scripted open-loop replay (E1 ablation): fixed v4 "
                          f"manager ladder rung at tick {state.tick} — no "
                          "digest/eval input to the choice"),
            "expected_effect": ("reproduces the v4 manager arm's knob "
                                "trajectory without closed-loop decisions"),
            "tripwire": tripwire,
        }
