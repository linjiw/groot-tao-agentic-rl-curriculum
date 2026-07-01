# SPDX-License-Identifier: Apache-2.0
"""Phase 1 closed loop: manager policy vs a LIVE toy tracking run.

Reuses the Phase-0 harness architecture (digest → decide → validate →
apply → tripwire → journal; harness owns the guardrails, the policy is
never trusted) but the records come from a knob-responsive live optimizer
(toy_tracking_run.py), so the manager's actions have consequences it must
live with in later ticks.

Policies:
  band      — BandStepperPolicy from Phase 0 (deterministic baseline)
  llm       — LLMPolicy: shells out to `claude -p` with the
              sonic-curriculum-manager playbook + the digest; parses the
              emitted decision YAML/JSON. Requires the claude CLI.
  none      — NullPolicy: always `action: none` (control arm)

Usage:
  python3 live_loop.py --policy band --ticks 16
  python3 live_loop.py --policy llm  --ticks 8   # needs `claude` on PATH
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _load(module_name: str, rel_path: str):
    path = os.path.join(_REPO, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


knob_registry = _load("knob_registry", "skills/agentic/sonic-knob-registry/knob_registry.py")
digest_builder = _load("digest_builder", "skills/agentic/sonic-run-digest/digest_builder.py")
# replay_harness imports its sibling `scenarios` — make phase0 importable first
sys.path.insert(0, os.path.join(_REPO, "experiments", "curriculum-manager-phase0"))
_ph0 = _load("replay_harness", "experiments/curriculum-manager-phase0/replay_harness.py")
BandStepperPolicy = _ph0.BandStepperPolicy
ArmedTripwire = _ph0.ArmedTripwire

sys.path.insert(0, _HERE)
from toy_tracking_run import ToyConfig, ToyTrackingRun  # noqa: E402

ITERS_PER_TICK = 250
PLAYBOOK_PATH = os.path.join(_REPO, "skills/agentic/sonic-curriculum-manager/SKILL.md")


class NullPolicy:
    def propose(self, digest, state, registry):
        return {"action": "none", "reason": "control arm"}


class LLMPolicy:
    """Shells out to `claude -p` with the playbook + digest; parses one decision.

    Any parse/subprocess failure degrades to `action: none` (fail-safe), and
    the failure is recorded in the returned reason so it lands in the journal.
    """

    def __init__(self, model: Optional[str] = None, timeout_s: int = 180):
        self.model = model
        self.timeout_s = timeout_s
        with open(PLAYBOOK_PATH) as f:
            self.playbook = f.read()

    def _prompt(self, digest: Dict[str, Any]) -> str:
        return (
            "You are the SONIC curriculum manager. Follow this playbook "
            "EXACTLY — hard rules, tick procedure, decision table:\n\n"
            "<playbook>\n" + self.playbook + "\n</playbook>\n\n"
            "Current digest (this tick's observation):\n\n"
            "```json\n" + json.dumps(digest, indent=1) + "\n```\n\n"
            "Reply with ONLY one fenced ```json block containing the decision "
            "object (the JSON equivalent of the playbook's YAML format: "
            '{"action": "none", "reason": ...} or {"action": "set", "knob": ..., '
            '"value": ..., "rationale": ..., "expected_effect": ..., '
            '"tripwire": {"metric": ..., "drop_pct": ..., "evals": ...}}). '
            "No other text."
        )

    def propose(self, digest, state, registry):
        cmd = ["claude", "-p", "--output-format", "text"]
        if self.model:
            cmd += ["--model", self.model]
        try:
            proc = subprocess.run(
                cmd, input=self._prompt(digest), capture_output=True,
                text=True, timeout=self.timeout_s,
            )
            if proc.returncode != 0:
                return {"action": "none",
                        "reason": f"llm error (rc={proc.returncode}): {proc.stderr[:200]}"}
            return self._parse(proc.stdout)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {"action": "none", "reason": f"llm unavailable: {e}"}

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = m.group(1) if m else text.strip()
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            return {"action": "none", "reason": f"unparseable llm output: {text[:200]!r}"}
        if not isinstance(decision, dict) or "action" not in decision:
            return {"action": "none", "reason": "llm output missing 'action'"}
        return decision


class LiveLoop:
    """Same guardrail layout as Phase-0's ReplayHarness, driving a live run."""

    def __init__(self, policy, run: ToyTrackingRun,
                 registry: Optional[Any] = None, window: int = 5):
        self.policy = policy
        self.run = run
        self.registry = registry or knob_registry.load_registry()
        self.window = window
        self.state = knob_registry.RunState(tick=0)
        self.journal: List[Dict[str, Any]] = []
        self.train: List[dict] = []
        self.eval: List[dict] = []
        self.sampler: List[dict] = []
        self.armed = None

    def _knob_values(self) -> Dict[str, float]:
        return {
            name: self.registry.current_of(name, self.state)
            for name in ("uniform_sampling_rate",
                         "adp_samp_failure_rate_max_over_mean",
                         "termination_threshold.anchor_pos")
        }

    def _knob_state(self):
        return {
            name: {
                "value": self.registry.current_of(name, self.state),
                "ticks_since_change": (
                    self.state.tick - self.state.last_changed_tick[name]
                    if name in self.state.last_changed_tick else None
                ),
            }
            for name in self.registry.knobs
        }

    def _heldout_last(self):
        for r in reversed(self.eval):
            if "heldout_success_rate" in r:
                return r["heldout_success_rate"]
        return None

    def tick(self) -> Dict[str, Any]:
        # 1) the LIVE run advances under current knob values
        out = self.run.advance(ITERS_PER_TICK, self._knob_values())
        self.train.extend(out["train"])
        self.eval.extend(out["eval"])
        self.sampler.extend(out["sampler"])
        self.state.tick += 1

        # 2) tripwire watch
        heldout = self._heldout_last()
        if self.armed and self.armed.check(heldout):
            self.state.apply(self.armed.knob, self.armed.prev_value)
            entry = {"tick": self.state.tick, "action": "rollback",
                     "knob": self.armed.knob,
                     "restored_value": self.armed.prev_value}
            for e in reversed(self.journal):
                if e.get("applied") and e.get("knob") == self.armed.knob:
                    e["outcome"] = "failed_rolled_back"
                    break
            self.journal.append(entry)
            self.armed = None
            return entry

        # score previous pending decision
        for e in reversed(self.journal):
            if e.get("applied") and e.get("outcome") == "pending":
                base, now = e["heldout_at_apply"], heldout
                e["outcome"] = ("unknown" if base is None or now is None
                                else "met" if now >= base - 1e-9 else "regressed")
                break

        # 3) digest → 4) decide → 5) validate/apply
        digest = digest_builder.build_digest(
            train_records=self.train, eval_records=self.eval,
            sampler_records=self.sampler, knob_state=self._knob_state(),
            decision_history=self.journal[-5:],
            max_over_mean=float(self.registry.current_of(
                "adp_samp_failure_rate_max_over_mean", self.state)),
            window=self.window,
        )
        decision = self.policy.propose(digest, self.state, self.registry)
        entry: Dict[str, Any] = {"tick": self.state.tick, "it": self.run.it,
                                 "decision": decision, "applied": False,
                                 "outcome": "n/a",
                                 "heldout": heldout}
        if decision.get("action") == "set":
            res = self.registry.validate_decision(decision, self.state)
            entry["validation"] = {"ok": res.ok, "errors": res.errors,
                                   "warnings": res.warnings}
            if res.ok:
                entry["knob"] = decision["knob"]
                entry["prev_value"] = self.registry.current_of(decision["knob"], self.state)
                self.state.apply(decision["knob"], decision["value"])
                entry["applied"] = True
                entry["outcome"] = "pending"
                entry["heldout_at_apply"] = heldout
                tw = decision["tripwire"]
                self.armed = ArmedTripwire(
                    knob=decision["knob"], prev_value=entry["prev_value"],
                    baseline=heldout if heldout is not None else 1.0,
                    drop_pct=float(tw["drop_pct"]), evals=int(tw["evals"]))
        self.journal.append(entry)
        return entry

    def run_ticks(self, n: int) -> Dict[str, Any]:
        for _ in range(n):
            self.tick()
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        applied = [e for e in self.journal if e.get("applied")]
        heldout_series = [r["heldout_success_rate"] for r in self.eval]
        return {
            "ticks": self.state.tick,
            "decisions_applied": len(applied),
            "rollbacks": len([e for e in self.journal if e.get("action") == "rollback"]),
            "rejected": len([e for e in self.journal
                             if e.get("validation") and not e["validation"]["ok"]]),
            "heldout_first": heldout_series[0] if heldout_series else None,
            "heldout_last": heldout_series[-1] if heldout_series else None,
            "final_knob_values": dict(self.state.current_values),
            "applied_knobs": [
                {"tick": e["tick"], "knob": e["knob"],
                 "value": e["decision"]["value"], "outcome": e["outcome"]}
                for e in applied
            ],
        }


POLICIES = {
    "band": lambda: BandStepperPolicy(),
    "llm": lambda model=None: LLMPolicy(model=model),
    "none": lambda: NullPolicy(),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase-1 live closed loop")
    p.add_argument("--policy", choices=sorted(POLICIES), default="band")
    p.add_argument("--ticks", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", help="model for --policy llm")
    p.add_argument("--journal-out")
    args = p.parse_args(argv)

    policy = (LLMPolicy(model=args.model) if args.policy == "llm"
              else POLICIES[args.policy]())
    loop = LiveLoop(policy, ToyTrackingRun(ToyConfig(seed=args.seed)))
    summary = loop.run_ticks(args.ticks)
    print(json.dumps(summary, indent=2))
    if args.journal_out:
        with open(args.journal_out, "w") as f:
            json.dump(loop.journal, f, indent=2)
        print(f"journal -> {args.journal_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
