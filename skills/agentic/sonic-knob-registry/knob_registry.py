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

# sentinel: a dotted path was absent from the resolved config (None is a
# legitimate config value, so it can't double as the miss marker)
_MISSING = object()


class ConfigDriftError(RuntimeError):
    """Believed knob values diverge from a run's resolved config.yaml
    (design doc 08 §11 amendment 8). Raised by callers that must REFUSE to
    keep driving a run whose real config they misdescribe."""


def resolve_config_value(cfg: Dict[str, Any], dotted_path: str) -> Any:
    """Walk `cfg` along a dotted path (a Hydra override path without the
    leading '+'/'++' append markers). Returns the `_MISSING` sentinel when
    any path segment is absent — never guesses."""
    node: Any = cfg
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _values_equal(a: Any, b: Any) -> bool:
    """Exact match; numbers compare as floats (yaml 0.2 vs Python 0.2 must
    not spuriously drift on int/float type), bools are NOT numbers."""

    def _num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    if _num(a) and _num(b):
        return float(a) == float(b)
    return a == b


@dataclasses.dataclass
class ConfigVerification:
    """Outcome of checking believed knob values against a run's resolved
    config.yaml (doc 08 §11 amendment 8)."""

    ok: bool
    # knob -> {"believed": ..., "resolved": ...} for every exact-match failure
    drifts: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    # knob -> resolved value seeded into the belief (no prior belief existed)
    adopted: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # knobs whose hydra_path was absent from the resolved config
    missing: List[str] = dataclasses.field(default_factory=list)
    # every knob that was looked up (has a hydra_path)
    checked: List[str] = dataclasses.field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    def raise_on_drift(self) -> "ConfigVerification":
        if not self.ok:
            detail = "; ".join(
                f"{k}: believed {v['believed']!r} but resolved config.yaml has {v['resolved']!r}"
                for k, v in self.drifts.items())
            raise ConfigDriftError(
                f"believed knob values diverge from the run's resolved config ({detail}); "
                "refusing — notch arithmetic from a wrong belief is how a 'one-notch' "
                "change became a 2x jump (doc 08 §11 amendment 8)")
        return self


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
    # knob currently under tripwire watch (one pending change at a time —
    # doc 08 §11 amendment 2). While set, validate_decision rejects any
    # further 'set'; the caller arms it on apply and clears it when the
    # change is scored (survived/effect-scored) or rolled back.
    pending: Optional[str] = None

    def apply(self, knob: str, value: Any) -> None:
        self.current_values[knob] = value
        self.last_changed_tick[knob] = self.tick

    def arm_pending(self, knob: str) -> None:
        self.pending = knob

    def clear_pending(self) -> None:
        self.pending = None


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
        digest: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Static gate (doc 08 §3 step 3). `digest`: when the caller supplies
        the tick's digest, the validator additionally machine-enforces
        playbook hard rule 4 for Family-B ('schedule') knobs — no held-out
        metric, no action. Callers without a held-out stream (e.g. the
        Phase-2 smoke, scope-noted in §11) omit it."""
        errors: List[str] = []
        warnings: List[str] = []

        action = decision.get("action")
        if action == "none":
            return ValidationResult(ok=True)
        if action != "set":
            return ValidationResult(ok=False, errors=[f"unknown action {action!r}"])

        # registry-level pending gate (doc 08 §11 amendment 2, defense in
        # depth with the driver's gate): while a prior change is under
        # tripwire watch — state.pending armed by the caller on apply,
        # cleared on score/rollback — every further 'set' is rejected.
        if getattr(state, "pending", None) is not None:
            return ValidationResult(ok=False, errors=[
                f"pending change on {state.pending!r} still under tripwire "
                "watch: one pending change at a time (observe-only until scored)"])

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

        # playbook hard rule 4, machine-enforced for Family-B ('schedule')
        # knobs when the caller supplies the tick's digest: no held-out
        # metric → no action. eval.heldout_success_rate must exist with a
        # non-null last value and a known trend; stock success_rate is NOT
        # sufficient evidence (eval/train decoupling, review 07:112).
        if digest is not None and k.get("family") == "schedule":
            heldout = (digest.get("eval") or {}).get("heldout_success_rate") or {}
            if heldout.get("last") is None or heldout.get("trend") in (None, "unknown"):
                errors.append(
                    f"hard rule 4: Family-B knob {name!r} requires a held-out "
                    "metric (eval.heldout_success_rate null or trend 'unknown' "
                    "-> no action)")

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

    # ── resolved-config verification (doc 08 §11 amendment 8) ────────
    def verify_against_config(
        self,
        state: RunState,
        resolved_cfg: Dict[str, Any],
        knob_paths: Dict[str, str],
        adopt_unseeded: bool = True,
    ) -> ConfigVerification:
        """Check believed knob values against a run's RESOLVED config.yaml
        (the ground truth of what the job actually ran with).

        `resolved_cfg`: the parsed config.yaml the run saved next to its
        checkpoints. `knob_paths`: knob name -> dotted path in that config
        (the adapter's Hydra override table minus the '+'/'++' markers).

        For every whitelisted knob in `knob_paths`:
        - path absent from the config -> listed in `missing` (flagged, not
          drift: a knob outside this run's config shape is a wiring gap,
          not a lie about a value);
        - belief exists (state.current_values) and differs from the
          resolved value beyond exact match -> drift, result not ok;
        - no belief yet and `adopt_unseeded` -> the resolved value is
          seeded into state.current_values (reconcile). This is the
          structural replacement for hand-seeding beliefs: current_of()
          would otherwise fall back to registry.yaml defaults, which
          describe a DIFFERENT config context — the v2 defect where a
          'one-notch' 0.30->0.35 was really 0.15->0.35;
        - no belief and not adopting -> the registry default (what
          current_of() would answer) is compared instead, so default-vs-
          config drift is caught rather than papered over.

        Never mutates beliefs that already exist; callers that must refuse
        on drift chain `.raise_on_drift()`.
        """
        result = ConfigVerification(ok=True)
        for name, path in knob_paths.items():
            if name not in self.knobs:
                continue  # outside the action space: nothing believed about it
            result.checked.append(name)
            resolved = resolve_config_value(resolved_cfg, path)
            if resolved is _MISSING:
                result.missing.append(name)
                continue
            if name in state.current_values:
                believed = state.current_values[name]
            elif adopt_unseeded:
                state.current_values[name] = resolved
                result.adopted[name] = resolved
                continue
            else:
                believed = self.default_of(name)
            if not _values_equal(believed, resolved):
                result.drifts[name] = {"believed": believed, "resolved": resolved}
        result.ok = not result.drifts
        return result


def load_registry(path: str = DEFAULT_REGISTRY_PATH) -> KnobRegistry:
    return KnobRegistry.load(path)
