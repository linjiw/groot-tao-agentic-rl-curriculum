# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tier-0 controller meta-knobs -> KnobRegistry wiring (doc 09 §7).

The three-tier control stack (doc 09 §7 amendment 1) requires that the
LLM (tier 2) touch the analytic controllers (tier 0) ONLY through their
scalar meta-parameters, and ONLY through the same static gate every other
knob goes through (whitelist, hard range, max-step, cooldown, pending
gate). This module supplies that wiring:

- `sigma_ema_knob_specs()` / `bin_lp_knob_specs()`: registry spec
  fragments for each controller's meta-parameters. All specs are plain
  data (dict), merged into the caller's action space via
  `merge_knob_specs`; ranges/steps/cooldowns carry conservative defaults
  and every one is caller-overridable (engine-agnostic: the core ships
  no mandatory action space, consistent with registry.py).

- `MetaKnobBinding`: knob name -> controller setter. `apply()` takes a
  decision that ALREADY passed `KnobRegistry.validate_decision` and
  routes the value to the setter, returning {requested, applied} so a
  clamping setter (BinLPSampler.set_uniform_mix grounding floor,
  SigmaEMAController.set_sigma_floor monotone guard) is journal-visible
  (doc 09 §7 amendment 4: a clamped request must never be silent).

Design constraints inherited from controllers.py: everything here is
deterministic pure data/dispatch — no RNG, no clock, no I/O — so the
wiring passes the E6 journal-equivalence gate unchanged.

Grounding note (doc 09 §7 amendment 4): `uniform_floor` is deliberately
NOT a registry knob. The floor is the guardrail that bounds the teacher;
exposing it to the teacher would let tier 2 remove its own leash. It is
set at construction time by the human/campaign config only.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Iterable, Optional

from .controllers import BinLPSampler, SigmaEMAController
from .registry import KnobRegistry, RunState, ValidationResult


def _knob(family: str, default: float, hard_range: Iterable[float],
          kind: str, cooldown_ticks: int, note: str,
          factor: Optional[float] = None,
          step: Optional[float] = None) -> Dict[str, Any]:
    max_step: Dict[str, Any] = {"kind": kind}
    if kind == "multiplicative":
        max_step["factor"] = factor
    elif kind == "additive":
        max_step["step"] = step
    return {
        "family": family,
        "type": "float",
        "default": default,
        "hard_range": list(hard_range),
        "max_step": max_step,
        "cooldown_ticks": cooldown_ticks,
        "status": "active",
        "note": note,
    }


def sigma_ema_knob_specs(prefix: str = "sigma_ema",
                         **overrides: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Registry spec fragment for SigmaEMAController meta-parameters.

    Knobs (names prefixed so multiple controller instances can coexist):
      {prefix}.ema_rate    — multiplicative, x2 per tick, cooldown 2
      {prefix}.sigma_floor — multiplicative, x2 per tick, cooldown 2

    `overrides`: per-knob spec-field overrides, keyed by the UNPREFIXED
    knob name, e.g. sigma_ema_knob_specs(ema_rate={"cooldown_ticks": 5}).
    """
    specs = {
        f"{prefix}.ema_rate": _knob(
            family="controller_meta", default=0.001,
            hard_range=(1e-5, 1.0), kind="multiplicative", factor=2.0,
            cooldown_ticks=2,
            note="PBHC sigma-EMA smoothing rate (controllers.SigmaEMAController)"),
        f"{prefix}.sigma_floor": _knob(
            family="controller_meta", default=1e-4,
            hard_range=(1e-6, 1.0), kind="multiplicative", factor=2.0,
            cooldown_ticks=2,
            note="hard lower bound on sigma; setter clamps to current sigma "
                 "(monotone invariant) — journal requested vs applied"),
    }
    _apply_overrides(specs, prefix, overrides)
    return specs


def bin_lp_knob_specs(prefix: str = "bin_lp",
                      **overrides: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Registry spec fragment for BinLPSampler meta-parameters.

    Knobs:
      {prefix}.uniform_mix — additive, +/-0.1 per tick, cooldown 2.
          Setter clamps UP to the grounding floor (amendment 4).
      {prefix}.power       — multiplicative, x2 per tick, cooldown 2.

    `uniform_floor` is intentionally ABSENT (see module docstring).
    fast_rate/slow_rate are also absent for now: changing EMA timescales
    mid-run invalidates accumulated per-bin statistics; expose them only
    with a stats-reset protocol (future work, journal-visible).
    """
    specs = {
        f"{prefix}.uniform_mix": _knob(
            family="controller_meta", default=0.2,
            hard_range=(0.0, 1.0), kind="additive", step=0.1,
            cooldown_ticks=2,
            note="EGM uniform-mix mass; setter clamps up to uniform_floor "
                 "(grounding, doc 09 §7 amendment 4)"),
        f"{prefix}.power": _knob(
            family="controller_meta", default=1.0,
            hard_range=(0.05, 8.0), kind="multiplicative", factor=2.0,
            cooldown_ticks=2,
            note="EGM LP power-smoothing exponent (0->flat, 1=proportional)"),
    }
    _apply_overrides(specs, prefix, overrides)
    return specs


def _apply_overrides(specs: Dict[str, Dict[str, Any]], prefix: str,
                     overrides: Dict[str, Dict[str, Any]]) -> None:
    for short_name, fields in overrides.items():
        full = f"{prefix}.{short_name}"
        if full not in specs:
            raise KeyError(f"unknown meta-knob {short_name!r} "
                           f"(have {sorted(specs)})")
        if not isinstance(fields, dict):
            raise TypeError(f"override for {short_name!r} must be a dict "
                            f"of spec fields, got {type(fields).__name__}")
        specs[full].update(fields)


def merge_knob_specs(*fragments: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Merge spec fragments into one 'knobs' mapping; duplicate knob
    names are a wiring bug and raise (never silently last-wins)."""
    merged: Dict[str, Dict[str, Any]] = {}
    for frag in fragments:
        for name, spec in frag.items():
            if name in merged:
                raise ValueError(f"duplicate knob name {name!r} across spec "
                                 "fragments (use distinct prefixes)")
            merged[name] = spec
    return merged


@dataclasses.dataclass
class MetaKnobApplied:
    """Result of routing one validated decision to a controller setter."""

    knob: str
    requested: float
    applied: float

    @property
    def clamped(self) -> bool:
        return self.requested != self.applied

    def journal_fields(self) -> Dict[str, Any]:
        """Fields the caller merges into the journal entry: a clamped
        request is never silent (amendment 4)."""
        out: Dict[str, Any] = {"applied_value": self.applied}
        if self.clamped:
            out["requested_value"] = self.requested
            out["clamped"] = True
        return out


class MetaKnobBinding:
    """knob name -> controller setter dispatch (tier-2 -> tier-0 seam).

    Construction wires each registry knob name to the bound method that
    applies it; `apply()` is the ONLY mutation path, and it refuses any
    decision the registry's static gate did not pass (defense in depth:
    the loop validates first, but the binding re-validates so it cannot
    be driven around the gate).
    """

    def __init__(self) -> None:
        self._setters: Dict[str, Callable[[float], float]] = {}

    # -- wiring -------------------------------------------------------
    def bind(self, knob: str, setter: Callable[[float], float]) -> "MetaKnobBinding":
        if knob in self._setters:
            raise ValueError(f"knob {knob!r} already bound")
        if not callable(setter):
            raise TypeError(f"setter for {knob!r} is not callable")
        self._setters[knob] = setter
        return self

    def bind_sigma_ema(self, controller: SigmaEMAController,
                       prefix: str = "sigma_ema") -> "MetaKnobBinding":
        return (self.bind(f"{prefix}.ema_rate", controller.set_ema_rate)
                    .bind(f"{prefix}.sigma_floor", controller.set_sigma_floor))

    def bind_bin_lp(self, sampler: BinLPSampler,
                    prefix: str = "bin_lp") -> "MetaKnobBinding":
        return (self.bind(f"{prefix}.uniform_mix", sampler.set_uniform_mix)
                    .bind(f"{prefix}.power", sampler.set_power))

    @property
    def bound_knobs(self) -> frozenset:
        return frozenset(self._setters)

    def check_coverage(self, registry: KnobRegistry,
                       family: str = "controller_meta") -> None:
        """Refuse a half-wired stack: every registry knob in `family`
        must have a setter, and every setter a registry knob. Call once
        at campaign setup — a knob the LLM can 'set' with no effect, or a
        setter the gate cannot reach, are both silent-failure modes."""
        reg = {n for n, k in registry.knobs.items() if k.get("family") == family}
        missing = reg - self.bound_knobs
        orphaned = self.bound_knobs - reg
        problems = []
        if missing:
            problems.append(f"registry knobs with no setter: {sorted(missing)}")
        if orphaned:
            problems.append(f"setters with no registry knob: {sorted(orphaned)}")
        if problems:
            raise ValueError("meta-knob wiring incomplete: " + "; ".join(problems))

    # -- the single mutation path --------------------------------------
    def apply(self, decision: Dict[str, Any], state: RunState,
              registry: KnobRegistry) -> MetaKnobApplied:
        """Validate through the registry's static gate, then route to the
        setter. On success the registry belief is updated with the value
        ACTUALLY applied (post-clamp) — believing the requested value
        would be exactly the drift amendment 8 exists to catch.

        Cooldown/pending bookkeeping (state.apply / arm_pending) stays
        with the caller (the loop owns watch lifecycle).
        """
        res: ValidationResult = registry.validate_decision(decision, state)
        if not res.ok:
            raise ValueError(
                f"meta-knob decision rejected by static gate: {res.errors}")
        knob = decision["knob"]
        setter = self._setters.get(knob)
        if setter is None:
            raise KeyError(f"knob {knob!r} passed the gate but has no bound "
                           "setter (run check_coverage at setup)")
        requested = float(decision["value"])
        applied = float(setter(requested))
        state.current_values[knob] = applied
        return MetaKnobApplied(knob=knob, requested=requested, applied=applied)
