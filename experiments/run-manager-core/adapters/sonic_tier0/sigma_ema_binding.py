# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""SigmaEMABinding — connects core.controllers.SigmaEMAController to the
SONIC reward kernel, with unit mapping, sidecar persistence, and a
sigma-trace journal. Torch-free / isaaclab-free so it is host-testable;
the torch shim (`sonic_sigma_ema_term.py`) owns tensors only.

Unit mapping (the subtle part)
------------------------------
SONIC parameterises the kernel by `std` in r = exp(-d / std**2); the
DENOMINATOR is std**2. SigmaEMAController tracks a tolerance `sigma` and
runs sigma <- min(sigma, EMA(err)) — a monotone shrink toward the typical
LINEAR error. So we:
  * init the controller at sigma_init = std (linear units),
  * feed it the mean LINEAR error sqrt(d) each step,
  * read sigma back and form the kernel denominator s = sigma**2,
  * retune via exponent p = std**2 / sigma**2 (sigma_ema_kernel).

At the first step (or in NO-OP mode) sigma == std => p == 1.0 => the
reward tensor is bit-identical to stock (G0 gate-0 requirement).

Modes
-----
  * ACTIVE:  update sigma every step, retune the reward.
  * NO_OP:   never update; sigma stays == std; p is always 1.0. This is
             the arm G0 compares bit-identically against stock, proving the
             insertion itself perturbs nothing.

Meta-parameters (tier-2, via the registry meta_knobs wiring): ema_rate,
sigma_floor. These are set at construction from config each segment; the
LLM (post-G2) proposes deltas through the same static gate as any knob.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

# core is a sibling package of adapters/ (both under run-manager-core/).
# The torch shim sets PYTHONPATH to the run-manager-core root inside the
# container; for host tests conftest puts that root on sys.path.
from core.controllers import SigmaEMAController

from . import sigma_ema_kernel as kern


class SigmaEMABinding:
    """Stateful per-term sigma controller with the SONIC unit mapping.

    Parameters
    ----------
    std : float
        The stock kernel std for the wrapped reward term (the no-op anchor).
    ema_rate, sigma_floor : float
        Controller meta-params. sigma_floor is in LINEAR (std) units.
    active : bool
        False => NO_OP mode (bit-identical passthrough, for G0/control arm).
    sidecar_path : Optional[str]
        If set, sigma state is persisted here on `save()` and reloaded on
        construction — the F8 segment-resume requirement. None => in-memory
        only (unit tests / single-segment smoke).
    """

    def __init__(self, std: float, *, ema_rate: float = 0.001,
                 sigma_floor_frac: float = 0.1, active: bool = True,
                 sidecar_path: Optional[str] = None,
                 term_name: str = "tracking_anchor_pos") -> None:
        if not (std > 0) or not math.isfinite(std):
            raise ValueError(f"std must be finite and > 0, got {std}")
        self.std = float(std)
        self.active = bool(active)
        self.term_name = term_name
        self.sidecar_path = sidecar_path
        # sigma_floor as a fraction of std keeps the meta-knob scale-free;
        # controller requires 0 <= floor < sigma_init.
        floor = max(0.0, min(sigma_floor_frac, 0.999)) * self.std
        self._ctrl = SigmaEMAController(
            sigma_init=self.std, ema_rate=ema_rate, sigma_floor=floor)
        self.n_steps = 0
        self._loaded_from_sidecar = False
        if sidecar_path and os.path.exists(sidecar_path):
            self._load(sidecar_path)

    # -- the two operations the torch shim calls each step ---------------
    def observe_and_factor(self, mean_r_stock_batch: Optional[List[float]],
                           mean_linear_err: Optional[float] = None) -> float:
        """Advance sigma from this step's errors, return the retune exponent.

        The torch shim reduces the reward tensor to a python scalar mean
        error (mean_linear_err) on-device and passes it here (cheap), OR
        passes the raw stock-reward batch for the CPU reference path.
        In NO_OP mode nothing updates and 1.0 is returned (passthrough).
        """
        if not self.active:
            return 1.0
        if mean_linear_err is None:
            if mean_r_stock_batch is None:
                raise ValueError("need mean_linear_err or mean_r_stock_batch")
            mean_linear_err = kern.mean_linear_error(mean_r_stock_batch, self.std)
        # A zero batch (no observation) must not shrink sigma.
        if mean_linear_err > 0:
            self._ctrl.update(mean_linear_err)
        self.n_steps += 1
        return kern.retune_factor(self.std, self._ctrl.sigma)

    def current_factor(self) -> float:
        """Retune exponent without advancing state (for logging/no-update)."""
        if not self.active:
            return 1.0
        return kern.retune_factor(self.std, self._ctrl.sigma)

    # -- meta-knob setters (tier-2 supervised) ---------------------------
    def set_ema_rate(self, value: float) -> float:
        return self._ctrl.set_ema_rate(value)

    def set_sigma_floor_frac(self, frac: float) -> float:
        applied = self._ctrl.set_sigma_floor(
            max(0.0, min(float(frac), 0.999)) * self.std)
        return applied / self.std

    # -- journal / persistence ------------------------------------------
    def trace_record(self) -> Dict[str, Any]:
        sd = self._ctrl.state_dict()
        return {
            "term": self.term_name,
            "active": self.active,
            "std": self.std,
            "sigma": sd["sigma"],
            "sigma_ratio": sd["sigma"] / self.std,
            "ema": sd["ema"],
            "retune_exponent": self.current_factor(),
            "n_steps": self.n_steps,
            "sigma_floor": self._ctrl.sigma_floor,
        }

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.sidecar_path
        if not p:
            return
        sd = self._ctrl.state_dict()
        payload = {
            "term_name": self.term_name, "std": self.std,
            "active": self.active,
            "sigma": sd["sigma"], "ema": sd["ema"],
            "n_updates": sd["n_updates"], "n_steps": self.n_steps,
            "ema_rate": self._ctrl.ema_rate,
            "sigma_floor": self._ctrl.sigma_floor,
        }
        tmp = p + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, p)  # atomic — a torn write must not corrupt resume

    def _load(self, path: str) -> None:
        with open(path) as fh:
            payload = json.load(fh)
        # A sidecar written under a DIFFERENT std (config changed std across
        # segments, or a foreign/copied sidecar) must NOT be trusted: its
        # sigma trajectory is in the old std's units, and restoring it could
        # push sigma > current std => retune exponent < 1 => reward INFLATED
        # above stock, silently breaking the monotone/no-op invariants
        # (adversarial review 2026-07-09, risk 5). Refuse and start fresh.
        saved_std = payload.get("std")
        if saved_std is not None and float(saved_std) != self.std:
            self._loaded_from_sidecar = False
            return
        # Restore controller state directly (bypassing update()) so a
        # resumed segment continues the exact sigma trajectory — but clamp
        # to the controller's OWN invariants (sigma <= sigma_init == std,
        # sigma >= floor) so a corrupt/edited payload cannot violate them.
        sigma_floor = float(payload.get("sigma_floor", self._ctrl.sigma_floor))
        # keep the fresher of the two floors so a segment that RAISED the
        # floor meta-knob still binds (floor is monotone-safe: clamped to sigma).
        sigma_floor = min(max(sigma_floor, self._ctrl.sigma_floor), self.std)
        sigma = float(payload["sigma"])
        sigma = min(sigma, self.std)            # never above sigma_init (=std)
        sigma = max(sigma, sigma_floor)         # never below the floor
        self._ctrl.sigma = sigma
        self._ctrl.sigma_floor = sigma_floor
        self._ctrl._ema = (None if payload["ema"] is None or
                           (isinstance(payload["ema"], float) and math.isnan(payload["ema"]))
                           else float(payload["ema"]))
        self._ctrl.n_updates = int(payload.get("n_updates", 0))
        self._ctrl.ema_rate = float(payload.get("ema_rate", self._ctrl.ema_rate))
        self.n_steps = int(payload.get("n_steps", 0))
        self._loaded_from_sidecar = True
