# SPDX-License-Identifier: Apache-2.0
"""Manager ON-vs-OFF smoke driver: the first composition of every validated
piece against real SONIC training (design doc 08 §8 Phase 2).

Loop per segment (manager arm):
  launch_segment → wait → parse_segment → build_digest → policy.propose
  → registry.validate → apply (next segment's knobs) or rollback → journal

Control arm: identical segments, no knob changes ever.

**Honest scope (bones-seed gated, 2-motion library):**
- No held-out metric exists here, so the doc-08 protected-metric discipline
  CANNOT be exercised; the policy acts on TRAINING-SIDE signals only
  (mean episode length bands, PBHC-style; sampler concentration proxies).
  This is explicitly weaker than the Phase-1 policy and is labeled so.
- With 2 motions, sampler-health decisions are near-meaningless; the run
  demonstrates the MECHANISM (composition, cadence, guardrails, journal),
  not curriculum value. Curriculum-value claims require bones-seed.

Policy: TrainSideBandPolicy —
  - loosen termination_threshold.anchor_pos one notch when mean episode
    length is BELOW len_low for `sustain` consecutive segments (relief for
    a run terminating too early to learn: ASAP/PBHC loose-first rationale);
  - tighten one notch when above len_high sustained (competence);
  - else none. Tripwire on Episode/rew_mean (training-side; drop_pct vs
    value at apply time) — weaker than held-out, labeled.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


knob_registry = _load("knob_registry", "skills/agentic/sonic-knob-registry/knob_registry.py")
digest_builder = _load("digest_builder", "skills/agentic/sonic-run-digest/digest_builder.py")
job_adapter = _load("job_adapter", "skills/agentic/sonic-job-adapter/job_adapter.py")


@dataclasses.dataclass
class TrainSideBandPolicy:
    """Episode-length band stepper (training-side only — see module header).

    Targets the BINDING termination axis: the threshold knob whose term has
    the highest termination fraction in the digest (loosening a term that
    never fires provably changes nothing — measured in the first smoke run,
    where anchor_pos sat at 0.0000 while foot_pos_xyz terminated 56% of
    episodes and the "managed" arm's trajectory was identical to control).
    """

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

    def _binding_knob(self, digest) -> tuple:
        """(knob, fraction) for the axis with highest WINDOWED termination
        mean (single-iteration fractions are too noisy — review finding 6)."""
        train = digest.get("train") or {}
        terms = (train.get("termination_terms_mean_recent")
                 or train.get("termination_terms_last") or {})
        candidates = {t: f for t, f in terms.items()
                      if t in self.TERM_TO_KNOB and isinstance(f, (int, float))}
        if not candidates or max(candidates.values()) <= 0:
            return None, None
        top = max(candidates, key=candidates.get)
        return self.TERM_TO_KNOB[top], candidates[top]

    def propose(self, digest: Dict[str, Any], state, registry) -> Dict[str, Any]:
        train = digest.get("train") or {}
        len_stats = train.get("Episode/len_mean")
        if not len_stats or len_stats.get("last") is None:
            return {"action": "none", "reason": "no episode-length signal"}
        self._len_history.append(len_stats["last"])
        recent = self._len_history[-self.sustain:]
        full = len(recent) == self.sustain

        knob, frac = self._binding_knob(digest)
        if knob is None:
            return {"action": "none",
                    "reason": "no binding threshold-termination term in digest"}
        cur = float(registry.current_of(knob, state))
        lo, hi = (float(x) for x in registry.knobs[knob]["hard_range"])
        tripwire = {"metric": "Episode/rew_mean", "drop_pct": 20, "evals": 2}

        if full and all(v < self.len_low for v in recent) and cur + self.notch <= hi:
            return {
                "action": "set", "knob": knob,
                "value": round(cur + self.notch, 4),
                "rationale": (f"mean episode length {[round(v,1) for v in recent]} < "
                              f"{self.len_low} for {self.sustain} segments; {knob} is the "
                              f"binding termination axis (windowed mean {frac:.2f} of "
                              "episodes; training-side signal; no held-out here)"),
                "expected_effect": "episodes lengthen; Episode/len_mean rises next segment",
                "tripwire": tripwire,
            }
        if full and all(v > self.len_high for v in recent) and cur - self.notch >= lo:
            return {
                "action": "set", "knob": knob,
                "value": round(cur - self.notch, 4),
                "rationale": (f"mean episode length {[round(v,1) for v in recent]} > "
                              f"{self.len_high} for {self.sustain} segments (training-side)"),
                "expected_effect": "tracking precision demand rises; length dips then recovers",
                "tripwire": tripwire,
            }
        return {"action": "none",
                "reason": f"len {[round(v,1) for v in recent]} inside "
                          f"[{self.len_low}, {self.len_high}] band or sustain unmet"}


class SmokeDriver:
    """Composes adapter + digest + policy + registry over real segments."""

    def __init__(self, policy, adapter=None, arm: str = "manager",
                 iterations_per_segment: int = 10, window: int = 5,
                 base_knobs: Optional[Dict[str, Any]] = None):
        self.policy = policy
        self.adapter = adapter or job_adapter.JobAdapter(
            project=f"smoke_{arm}", num_envs=64, save_last_frequency=5)
        self.arm = arm
        self.iters = iterations_per_segment
        self.window = window
        self.registry = knob_registry.load_registry()
        self.state = knob_registry.RunState(tick=0)
        self.knobs: Dict[str, Any] = dict(base_knobs or {})
        self.journal: List[Dict[str, Any]] = []
        self.all_train: List[dict] = []
        self.all_sampler: List[dict] = []
        self.armed: Optional[Dict[str, Any]] = None  # {knob, prev_value, prev_ckpt, baseline, breaches, tw}

    def _knob_state(self):
        return {
            name: {"value": self.registry.current_of(name, self.state),
                   "ticks_since_change": (self.state.tick - self.state.last_changed_tick[name]
                                          if name in self.state.last_changed_tick else None)}
            for name in self.knobs or {}
        }

    def run(self, n_segments: int) -> Dict[str, Any]:
        checkpoint = None
        for i in range(n_segments):
            self.state.tick += 1
            name = f"{self.arm}_s{i+1}"
            seg = self.adapter.launch_segment(name, self.iters, self.knobs,
                                              checkpoint_in=checkpoint)
            self.adapter.wait(seg, poll_s=10, timeout_s=1800)
            parsed = self.adapter.parse_segment(seg)
            if seg.status != "done":
                self.journal.append({"tick": self.state.tick, "segment": name,
                                     "event": "segment_failed",
                                     "tracebacks": parsed.tracebacks})
                break
            # offset iteration numbers so records accumulate monotonically.
            # Console iteration numbering restarts at 1 every segment even on
            # resume (verified: seg2 logs "Learning iteration 1" after
            # "Loaded checkpoint from step 5") — normalize against the first
            # parsed iteration so a numbering change upstream can't double-offset.
            base_it = self.all_train[-1]["it"] if self.all_train else 0
            first_it = parsed.train[0]["it"] if parsed.train else 1
            for r in parsed.train:
                r["it"] += base_it - (first_it - 1)
            for r in parsed.sampler:
                r["it"] += base_it - (first_it - 1)
            self.all_train.extend(parsed.train)
            self.all_sampler.extend(parsed.sampler)
            checkpoint = seg.snapshot or checkpoint

            rew = parsed.train[-1].get("Episode/rew_mean") if parsed.train else None
            entry: Dict[str, Any] = {
                "tick": self.state.tick, "segment": name,
                "knobs_in": dict(self.knobs),
                "rew_mean_last": rew,
                "len_mean_last": parsed.train[-1].get("Episode/len_mean") if parsed.train else None,
            }

            # tripwire watch (training-side rew_mean)
            if self.armed is not None and rew is not None:
                tw = self.armed
                if rew < tw["baseline"] * (1 - tw["tw"]["drop_pct"] / 100.0):
                    tw["breaches"] += 1
                    tw["clean"] = 0
                else:
                    tw["breaches"] = 0
                    tw["clean"] = tw.get("clean", 0) + 1
                if tw["breaches"] >= tw["tw"]["evals"]:
                    self.knobs[tw["knob"]] = tw["prev_value"]
                    self.state.apply(tw["knob"], tw["prev_value"])
                    checkpoint = tw["prev_ckpt"]
                    entry["event"] = "rollback"
                    entry["restored"] = {tw["knob"]: tw["prev_value"]}
                    # mark the originating decision failed (doc 08 §3 step 5)
                    for prev in reversed(self.journal):
                        if prev.get("applied") and prev["decision"]["knob"] == tw["knob"]:
                            prev["outcome"] = "failed_rolled_back"
                            break
                    self.armed = None
                    self.journal.append(entry)
                    continue
                if tw["clean"] >= tw["tw"]["evals"]:
                    # survived its watch window: score it and disarm (NOT "met" —
                    # survival of the tripwire is weaker than expected_effect
                    # satisfaction, which is unchecked here)
                    for prev in reversed(self.journal):
                        if prev.get("applied") and prev["decision"]["knob"] == tw["knob"]:
                            prev["outcome"] = "survived"
                            break
                    self.armed = None

            if self.arm == "control":
                entry["decision"] = {"action": "none", "reason": "control arm"}
                self.journal.append(entry)
                continue

            # pending-decision gate (playbook tick-procedure step 2; review
            # finding 1-2): while a change is armed/unscored, no new change —
            # overlapping changes orphan the first tripwire and destroy
            # attribution. The policy is not even consulted.
            if self.armed is not None:
                entry["decision"] = {
                    "action": "none",
                    "reason": f"pending decision on {self.armed['knob']} still "
                              "under tripwire watch (one change at a time)"}
                self.journal.append(entry)
                continue

            digest = digest_builder.build_digest(
                train_records=self.all_train, sampler_records=self.all_sampler,
                knob_state=self._knob_state(), decision_history=self.journal[-5:],
                max_over_mean=float(self.knobs.get(
                    "adp_samp_failure_rate_max_over_mean", 50.0)),
                window=self.window)
            decision = self.policy.propose(digest, self.state, self.registry)
            entry["decision"] = decision
            if decision.get("action") == "set":
                res = self.registry.validate_decision(decision, self.state)
                entry["validation"] = {"ok": res.ok, "errors": res.errors}
                if res.ok and rew is None:
                    # no baseline → the tripwire would be unarmed in practice;
                    # refuse the change rather than apply it unguarded
                    entry["validation"] = {"ok": False, "errors":
                        ["no reward baseline available to arm the tripwire"]}
                elif res.ok:
                    prev = self.registry.current_of(decision["knob"], self.state)
                    self.armed = {"knob": decision["knob"], "prev_value": prev,
                                  "prev_ckpt": checkpoint,
                                  "baseline": rew,
                                  "breaches": 0, "clean": 0,
                                  "tw": decision["tripwire"]}
                    self.knobs[decision["knob"]] = decision["value"]
                    self.state.apply(decision["knob"], decision["value"])
                    entry["applied"] = True
                    entry["outcome"] = "pending"
            self.journal.append(entry)
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        applied = [e for e in self.journal if e.get("applied")]
        return {
            "arm": self.arm,
            "segments": len([e for e in self.journal if "segment" in e]),
            "decisions_applied": len(applied),
            "rollbacks": len([e for e in self.journal if e.get("event") == "rollback"]),
            "rejected": len([e for e in self.journal
                             if e.get("validation") and not e["validation"]["ok"]]),
            "final_knobs": dict(self.knobs),
            "rew_series": [e.get("rew_mean_last") for e in self.journal],
            "len_series": [e.get("len_mean_last") for e in self.journal],
        }


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="manager ON-vs-OFF smoke")
    p.add_argument("--arm", choices=["manager", "control"], required=True)
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--len-low", type=float, default=20.0)
    p.add_argument("--sustain", type=int, default=2)
    p.add_argument("--journal-out")
    args = p.parse_args(argv)

    policy = TrainSideBandPolicy(len_low=args.len_low, sustain=args.sustain)
    driver = SmokeDriver(policy, arm=args.arm, iterations_per_segment=args.iters)
    summary = driver.run(args.segments)
    print(json.dumps(summary, indent=2))
    if args.journal_out:
        with open(args.journal_out, "w") as f:
            json.dump(driver.journal, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
