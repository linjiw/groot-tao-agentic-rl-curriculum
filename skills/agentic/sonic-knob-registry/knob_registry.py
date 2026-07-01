# SPDX-License-Identifier: Apache-2.0
"""Typed knob registry + decision-delta validator for the Curriculum-Manager Agent.

Design doc: docs/design/08-curriculum-manager-agent.md §5 (action space) and §3
(decision loop, step 3 "Validate"). Pure Python + PyYAML; no torch/IsaacLab.

A *decision* is what the manager LLM emits each tick:

    {"action": "none"}                                    # the default
    {"action": "set", "knob": "<name>", "value": <v>,
     "rationale": "...", "expected_effect": "...",
     "tripwire": {"metric": "...", "drop_pct": 5, "evals": 3}}

`validate_decision` is the static gate that runs BEFORE anything is applied:
knob whitelisted, value typed and in hard range, |delta| within max step,
cooldown elapsed, one atomic change per tick. It never mutates state; the
caller journals accepted decisions and advances tick state via `RunState`.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.yaml")

REQUIRED_DECISION_FIELDS = ("rationale", "expected_effect", "tripwire")
REQUIRED_TRIPWIRE_FIELDS = ("metric", "drop_pct", "evals")


@dataclasses.dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


@dataclasses.dataclass
class RunState:
    """Per-run mutable state the validator checks against (caller-owned)."""

    tick: int = 0
    # knob name -> current value (falls back to registry default when absent)
    current_values: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # knob name -> tick at which it was last changed
    last_changed_tick: Dict[str, int] = dataclasses.field(default_factory=dict)

    def apply(self, knob: str, value: Any) -> None:
        self.current_values[knob] = value
        self.last_changed_tick[knob] = self.tick


class KnobRegistry:
    def __init__(self, spec: Dict[str, Any]):
        if "knobs" not in spec or not isinstance(spec["knobs"], dict):
            raise ValueError("registry spec missing 'knobs' mapping")
        self.meta: Dict[str, Any] = spec.get("meta", {})
        self.knobs: Dict[str, Dict[str, Any]] = spec["knobs"]
        for name, k in self.knobs.items():
            self._check_knob_spec(name, k)

    @classmethod
    def load(cls, path: str = DEFAULT_REGISTRY_PATH) -> "KnobRegistry":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    @staticmethod
    def _check_knob_spec(name: str, k: Dict[str, Any]) -> None:
        for field in ("family", "type", "max_step", "cooldown_ticks", "status"):
            if field not in k:
                raise ValueError(f"knob {name!r}: missing {field!r}")
        if k["type"] == "choice":
            if "choices" not in k or len(k["choices"]) < 2:
                raise ValueError(f"knob {name!r}: choice type needs >=2 choices")
        elif k["type"] == "float":
            rng = k.get("hard_range")
            if not (isinstance(rng, list) and len(rng) == 2 and rng[0] < rng[1]):
                raise ValueError(f"knob {name!r}: bad hard_range {rng!r}")
        else:
            raise ValueError(f"knob {name!r}: unknown type {k['type']!r}")
        kind = k["max_step"].get("kind")
        if kind not in ("multiplicative", "additive", "notch"):
            raise ValueError(f"knob {name!r}: unknown max_step kind {kind!r}")
        if kind == "multiplicative" and k["max_step"].get("factor", 0) <= 1:
            raise ValueError(f"knob {name!r}: multiplicative factor must be > 1")
        if kind == "additive" and k["max_step"].get("step", 0) <= 0:
            raise ValueError(f"knob {name!r}: additive step must be > 0")

    def default_of(self, name: str) -> Any:
        return self.knobs[name].get("default")

    def current_of(self, name: str, state: RunState) -> Any:
        if name in state.current_values:
            return state.current_values[name]
        return self.default_of(name)

    # ── the static gate ──────────────────────────────────────────────
    def validate_decision(
        self,
        decision: Dict[str, Any],
        state: RunState,
        allow_design: bool = False,
    ) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        action = decision.get("action")
        if action == "none":
            return ValidationResult(ok=True)
        if action != "set":
            return ValidationResult(ok=False, errors=[f"unknown action {action!r}"])

        # one atomic change per tick: a single decision dict IS one change;
        # reject list-shaped or multi-knob payloads explicitly.
        if isinstance(decision.get("knob"), (list, tuple)):
            return ValidationResult(
                ok=False, errors=["one atomic change per tick: 'knob' must be a single name"]
            )

        name = decision.get("knob")
        if name not in self.knobs:
            return ValidationResult(ok=False, errors=[f"knob {name!r} not in registry (outside action space)"])
        k = self.knobs[name]

        for field in REQUIRED_DECISION_FIELDS:
            if not decision.get(field):
                errors.append(f"missing required field {field!r}")
        tripwire = decision.get("tripwire")
        if isinstance(tripwire, dict):
            for field in REQUIRED_TRIPWIRE_FIELDS:
                if field not in tripwire:
                    errors.append(f"tripwire missing {field!r}")
        elif tripwire is not None:
            errors.append("tripwire must be a mapping")

        if k["status"] == "design" and not allow_design:
            errors.append(
                f"knob {name!r} has status 'design' (mechanism not built); "
                "pass allow_design=True only in replay/simulation"
            )

        # cooldown
        last = state.last_changed_tick.get(name)
        cooldown = int(k["cooldown_ticks"])
        if last is not None and (state.tick - last) < cooldown:
            errors.append(
                f"cooldown: {name!r} changed at tick {last}, "
                f"{cooldown - (state.tick - last)} tick(s) remaining"
            )

        # value: type, hard range, max step
        value = decision.get("value")
        current = self.current_of(name, state)
        if k["type"] == "choice":
            choices = list(k["choices"])
            if value not in choices:
                errors.append(f"value {value!r} not in choices {choices}")
            elif current in choices and abs(choices.index(value) - choices.index(current)) > 1:
                errors.append(f"notch step: {current!r} -> {value!r} skips a notch in {choices}")
        else:  # float
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                errors.append(f"value {value!r} is not a finite number")
            else:
                value = float(value)
                lo, hi = (float(x) for x in k["hard_range"])
                if not (lo <= value <= hi):
                    errors.append(f"value {value} outside hard range [{lo}, {hi}]")
                step = k["max_step"]
                if current is not None:
                    current = float(current)
                    if value == current:
                        warnings.append(f"value equals current ({current}); prefer action 'none'")
                    if step["kind"] == "multiplicative":
                        f = float(step["factor"])
                        tol = 1e-9  # float slack so an exactly-x-factor step is legal
                        if current > 0 and not (
                            current / f * (1 - tol) <= value <= current * f * (1 + tol)
                        ):
                            errors.append(
                                f"step too large: {current} -> {value} exceeds x{f} / /{f}"
                            )
                    elif step["kind"] == "additive":
                        s = float(step["step"])
                        if abs(value - current) > s + 1e-12:
                            errors.append(f"step too large: |{value} - {current}| > {s}")

        if k.get("restart_required"):
            warnings.append(f"knob {name!r} requires a run restart to take effect")

        return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def load_registry(path: str = DEFAULT_REGISTRY_PATH) -> KnobRegistry:
    return KnobRegistry.load(path)
