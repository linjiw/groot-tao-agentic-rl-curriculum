# SPDX-License-Identifier: Apache-2.0
"""Replay harness: run the Curriculum-Manager decision loop against recorded
or synthetic run scenarios — no GPU, no LLM, no live trainer.

Design doc 08, Phase 0: "a replay harness that feeds recorded/synthetic
training logs to the manager and checks its decisions against the registry +
playbook (does it hold in thrash scenarios? does it tighten on sustained
success? does the tripwire fire?)".

The LLM slot is filled by `BandStepperPolicy` — a deterministic
implementation of the playbook's mechanical core (doc 08 §6.1–6.2):

  * ADR-style dual-threshold stepping with hysteresis (tighten a Family-B
    threshold one notch when HELD-OUT success ≥ t_high for `sustain`
    consecutive evals; loosen when ≤ t_low for `sustain` evals),
  * sampler-health regulation (raise the uniform floor when the failure
    vector is concentrated + success is stagnant),
  * "do nothing" default everywhere else.

Any future policy (including the real LLM) plugs in via the same interface:
`propose(digest, state, registry) -> decision dict`. The harness owns what
the policy must NOT: validation (knob registry), the tripwire/rollback, and
the journal. That separation is the guardrail architecture itself.
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


def _load(module_name: str, rel_path: str):
    path = os.path.join(_REPO, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


knob_registry = _load(
    "knob_registry", "skills/agentic/sonic-knob-registry/knob_registry.py"
)
digest_builder = _load(
    "digest_builder", "skills/agentic/sonic-run-digest/digest_builder.py"
)
from scenarios import SCENARIOS, TickRecords  # noqa: E402


# ── the deterministic playbook core (LLM stand-in) ───────────────────
@dataclasses.dataclass
class BandStepperPolicy:
    """Doc 08 §6.1 success-band stepping + §6.2 sampler-health regulation."""

    t_high: float = 0.85
    t_low: float = 0.50
    sustain: int = 3  # consecutive evals required (anti-thrash)
    entropy_floor: float = 0.75  # below → concentrated
    threshold_knob: str = "termination_threshold.anchor_pos"
    notch: float = 0.05

    def __post_init__(self):
        self._heldout_history: List[float] = []

    @staticmethod
    def _tripwire() -> Dict[str, Any]:
        return {"metric": "heldout_success_rate", "drop_pct": 5, "evals": 3}

    @staticmethod
    def _off_cooldown(knob: str, state, registry) -> bool:
        last = state.last_changed_tick.get(knob)
        if last is None:
            return True
        return (state.tick - last) >= int(registry.knobs[knob]["cooldown_ticks"])

    def propose(
        self,
        digest: Dict[str, Any],
        state: "knob_registry.RunState",
        registry: "knob_registry.KnobRegistry",
    ) -> Dict[str, Any]:
        ev = digest.get("eval") or {}
        heldout = ev.get("heldout_success_rate")

        # Protected-metric rule (doc 08 §9): no held-out signal → no action.
        if not heldout or heldout.get("last") is None:
            return {"action": "none", "reason": "no held-out metric"}
        self._heldout_history.append(heldout["last"])

        recent = self._heldout_history[-self.sustain:]
        sustained_high = len(recent) == self.sustain and all(v >= self.t_high for v in recent)
        sustained_low = len(recent) == self.sustain and all(v <= self.t_low for v in recent)

        # §6.1 — Family-B threshold stepping on sustained held-out evidence.
        cur = float(registry.current_of(self.threshold_knob, state))
        lo, hi = (float(x) for x in registry.knobs[self.threshold_knob]["hard_range"])
        threshold_ready = self._off_cooldown(self.threshold_knob, state, registry)
        sustained_high = sustained_high and threshold_ready
        sustained_low = sustained_low and threshold_ready
        if sustained_high and cur - self.notch >= lo:
            return {
                "action": "set",
                "knob": self.threshold_knob,
                "value": round(cur - self.notch, 4),
                "rationale": f"held-out success >= {self.t_high} for {self.sustain} evals",
                "expected_effect": "tracking precision demand rises; held-out success dips then recovers",
                "tripwire": self._tripwire(),
            }
        if sustained_low and cur + self.notch <= hi:
            return {
                "action": "set",
                "knob": self.threshold_knob,
                "value": round(cur + self.notch, 4),
                "rationale": f"held-out success <= {self.t_low} for {self.sustain} evals",
                "expected_effect": "episodes lengthen; held-out success recovers",
                "tripwire": self._tripwire(),
            }

        # §6.2 — sampler-health: concentrated failure mass + stagnant success
        # → raise the uniform floor one bounded step.
        sampler = digest.get("sampler") or {}
        ent = (sampler.get("normalized_entropy") or {}).get("last")
        stagnant = heldout.get("trend") == "flat" and heldout["last"] < self.t_high
        if (
            ent is not None
            and ent < self.entropy_floor
            and stagnant
            and self._off_cooldown("uniform_sampling_rate", state, registry)
        ):
            cur_u = float(registry.current_of("uniform_sampling_rate", state))
            factor = float(registry.knobs["uniform_sampling_rate"]["max_step"]["factor"])
            hi_u = float(registry.knobs["uniform_sampling_rate"]["hard_range"][1])
            new_u = round(min(cur_u * factor, hi_u), 4)
            if new_u > cur_u:
                return {
                    "action": "set",
                    "knob": "uniform_sampling_rate",
                    "value": new_u,
                    "rationale": f"failure-rate entropy {ent:.2f} < {self.entropy_floor} with stagnant held-out success",
                    "expected_effect": "sampler concentration falls; entropy rises",
                    "tripwire": self._tripwire(),
                }

        return {"action": "none", "reason": "healthy / insufficient sustained evidence"}


# ── tripwire ─────────────────────────────────────────────────────────
@dataclasses.dataclass
class ArmedTripwire:
    knob: str
    prev_value: Any
    baseline: float  # protected metric at apply time
    drop_pct: float
    evals: int
    breaches: int = 0

    def check(self, heldout_last: Optional[float]) -> bool:
        """True when the tripwire fires (sustained drop vs baseline)."""
        if heldout_last is None:
            return False
        if heldout_last < self.baseline * (1 - self.drop_pct / 100.0):
            self.breaches += 1
        else:
            self.breaches = 0
        return self.breaches >= self.evals


# ── the manager loop ─────────────────────────────────────────────────
class ReplayHarness:
    def __init__(
        self,
        policy,
        registry: Optional["knob_registry.KnobRegistry"] = None,
        allow_design: bool = True,  # replay may exercise design-status knobs
        window: int = 5,
    ):
        self.policy = policy
        self.registry = registry or knob_registry.load_registry()
        self.allow_design = allow_design
        self.window = window
        self.state = knob_registry.RunState(tick=0)
        self.journal: List[Dict[str, Any]] = []
        self.train: List[dict] = []
        self.eval: List[dict] = []
        self.sampler: List[dict] = []
        self.armed: Optional[ArmedTripwire] = None
        self.rollbacks: List[Dict[str, Any]] = []

    # -- helpers ------------------------------------------------------
    def _knob_state(self) -> Dict[str, Any]:
        out = {}
        for name in self.registry.knobs:
            out[name] = {
                "value": self.registry.current_of(name, self.state),
                "ticks_since_change": (
                    self.state.tick - self.state.last_changed_tick[name]
                    if name in self.state.last_changed_tick
                    else None
                ),
            }
        return out

    def _heldout_last(self) -> Optional[float]:
        for r in reversed(self.eval):
            if "heldout_success_rate" in r:
                return r["heldout_success_rate"]
        return None

    def _score_last_decision(self) -> None:
        """Outcome attribution (doc 08 §6.5): annotate the previous applied
        decision with whether the protected metric held."""
        for entry in reversed(self.journal):
            if entry.get("applied") and entry.get("outcome") == "pending":
                baseline = entry["heldout_at_apply"]
                now = self._heldout_last()
                if baseline is None or now is None:
                    entry["outcome"] = "unknown"
                elif now >= baseline - 1e-9:
                    entry["outcome"] = "met"
                else:
                    entry["outcome"] = "regressed"
                break

    # -- one tick -----------------------------------------------------
    def tick(self, records: TickRecords) -> Dict[str, Any]:
        self.train.extend(records.train)
        self.eval.extend(records.eval)
        self.sampler.extend(records.sampler)
        self.state.tick += 1

        # 0) tripwire watch BEFORE anything else (doc 08 §3 step 5)
        heldout = self._heldout_last()
        if self.armed and self.armed.check(heldout):
            self.state.apply(self.armed.knob, self.armed.prev_value)
            rollback = {
                "tick": self.state.tick,
                "action": "rollback",
                "knob": self.armed.knob,
                "restored_value": self.armed.prev_value,
                "reason": f"tripwire: held-out dropped >{self.armed.drop_pct}% "
                          f"for {self.armed.evals} evals",
            }
            self.rollbacks.append(rollback)
            # mark the originating decision failed so the policy sees it
            for entry in reversed(self.journal):
                if entry.get("applied") and entry.get("knob") == self.armed.knob:
                    entry["outcome"] = "failed_rolled_back"
                    break
            self.journal.append(rollback)
            self.armed = None
            return rollback

        self._score_last_decision()

        # 1) digest
        max_over_mean = float(
            self.registry.current_of("adp_samp_failure_rate_max_over_mean", self.state)
        )
        digest = digest_builder.build_digest(
            train_records=self.train,
            eval_records=self.eval,
            sampler_records=self.sampler,
            knob_state=self._knob_state(),
            decision_history=self.journal[-5:],
            max_over_mean=max_over_mean,
            window=self.window,
        )

        # 2) decide
        decision = self.policy.propose(digest, self.state, self.registry)

        # 3) validate (the static gate — the policy is not trusted)
        entry: Dict[str, Any] = {
            "tick": self.state.tick,
            "it": records.it,
            "decision": decision,
            "applied": False,
            "outcome": "n/a",
        }
        if decision.get("action") == "set":
            result = self.registry.validate_decision(
                decision, self.state, allow_design=self.allow_design
            )
            entry["validation"] = {"ok": result.ok, "errors": result.errors,
                                   "warnings": result.warnings}
            if result.ok:
                # 4) apply + arm the tripwire
                entry["knob"] = decision["knob"]
                entry["prev_value"] = self.registry.current_of(decision["knob"], self.state)
                self.state.apply(decision["knob"], decision["value"])
                entry["applied"] = True
                entry["outcome"] = "pending"
                entry["heldout_at_apply"] = heldout
                tw = decision["tripwire"]
                self.armed = ArmedTripwire(
                    knob=decision["knob"],
                    prev_value=entry["prev_value"],
                    baseline=heldout if heldout is not None else 1.0,
                    drop_pct=float(tw["drop_pct"]),
                    evals=int(tw["evals"]),
                )
        self.journal.append(entry)
        return entry

    def run(self, ticks) -> List[Dict[str, Any]]:
        for records in ticks:
            self.tick(records)
        return self.journal

    def summary(self) -> Dict[str, Any]:
        applied = [e for e in self.journal if e.get("applied")]
        return {
            "ticks": self.state.tick,
            "decisions_applied": len(applied),
            "rollbacks": len(self.rollbacks),
            "rejected": len(
                [e for e in self.journal
                 if e.get("validation") and not e["validation"]["ok"]]
            ),
            "final_knob_values": {
                k: self.registry.current_of(k, self.state)
                for k in self.state.current_values
            },
            "outcomes": {e.get("knob"): e.get("outcome") for e in applied},
        }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Replay the manager loop on a scenario")
    p.add_argument("scenario", choices=sorted(SCENARIOS))
    p.add_argument("--ticks", type=int, default=12)
    p.add_argument("--journal-out")
    args = p.parse_args(argv)

    harness = ReplayHarness(BandStepperPolicy())
    harness.run(SCENARIOS[args.scenario](n_ticks=args.ticks))
    print(json.dumps(harness.summary(), indent=2))
    if args.journal_out:
        with open(args.journal_out, "w") as f:
            json.dump(harness.journal, f, indent=2)
        print(f"journal -> {args.journal_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
