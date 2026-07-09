# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""SONIC reward-term shim wiring SigmaEMAController into live GR00T-WBC
training via a Hydra `func` override — NO edits to the pinned submodule.

This is the ONLY module that imports torch/isaaclab/gear_sonic; it runs
inside the `isaac-lab-base` container. The numeric logic lives in the
torch-free `sigma_ema_binding` / `sigma_ema_kernel` (host-unit-tested);
this file is the thin per-step tensor adapter.

How it is inserted (I1_INSERTION_RECON.md F1-F5)
------------------------------------------------
1. Expose run-manager-core on the container PYTHONPATH (bind-mounted under
   /workspace) so `adapters.sonic_tier0...` and `core...` import.
2. Override the reward term's func at launch:

     ++manager_env.rewards.tracking_anchor_pos.func=\
       adapters.sonic_tier0.sonic_sigma_ema_term:SigmaEMAAnchorPos

   isaaclab's string_to_callable importlib-loads this; RewardManager treats
   a ManagerTermBase SUBCLASS as a stateful callable term (reset/__call__).
3. NO-OP mode (SONIC_TIER0_ACTIVE=0, the default) is bit-identical to stock
   — the G0 gate-0 arm. ACTIVE mode (=1) runs PBHC sigma-EMA.

Config is read from env vars (not new Hydra keys) so the pinned config
schema is untouched and the manager's meta-knobs flow in without adding
registry-unknown fields to SONIC's config tree:
    SONIC_TIER0_ACTIVE        "1"/"0"   (default "0" = no-op)
    SONIC_TIER0_EMA_RATE      float     (default 0.001)
    SONIC_TIER0_SIGMA_FLOOR   float     fraction of std (default 0.1)
    SONIC_TIER0_SIDECAR_DIR   dir       sigma-state persistence (F8)
    SONIC_TIER0_TRACE         path      append-JSONL sigma trace (journal)

The wrapped stock function is imported by name so we call the EXACT SONIC
kernel (no float-order drift): gear_sonic.envs.manager_env.mdp:<STOCK_FN>.
"""

from __future__ import annotations

import json
import os
from typing import Optional

# Import guard: on the host (tests) torch/isaaclab are absent. We still
# want `import ... sonic_sigma_ema_term` to fail loudly ONLY if actually
# constructed without torch, so the binding stays host-importable.
try:
    import torch  # noqa: F401
    from isaaclab.managers import ManagerTermBase
    from isaaclab.utils.string import string_to_callable
    _HAVE_TORCH = True
except Exception:  # pragma: no cover - exercised only in-container
    _HAVE_TORCH = False
    ManagerTermBase = object  # type: ignore

from .sigma_ema_binding import SigmaEMABinding

# Which stock SONIC reward function this term wraps, and its std param name.
# Subclasses set STOCK_FN; std comes from the term's own `params.std`.
_STOCK_MODULE = "gear_sonic.envs.manager_env.mdp"


class _SigmaEMARewardTerm(ManagerTermBase):
    """Base callable-class reward term: wrap a stock SONIC kernel, retune
    its output by the live sigma-EMA exponent, observe error for the EMA.

    Subclasses set `STOCK_FN` (the attribute name in gear_sonic...mdp).
    """

    STOCK_FN: str = ""

    def __init__(self, cfg, env):  # cfg: RewardTermCfg, env: ManagerBasedRLEnv
        super().__init__(cfg, env)
        if not self.STOCK_FN:
            raise ValueError("subclass must set STOCK_FN")
        self._stock = string_to_callable(f"{_STOCK_MODULE}:{self.STOCK_FN}")
        self._std = float(cfg.params["std"])
        active = os.environ.get("SONIC_TIER0_ACTIVE", "0") == "1"
        ema_rate = float(os.environ.get("SONIC_TIER0_EMA_RATE", "0.001"))
        floor_frac = float(os.environ.get("SONIC_TIER0_SIGMA_FLOOR", "0.1"))
        sidecar_dir = os.environ.get("SONIC_TIER0_SIDECAR_DIR") or None
        sidecar = (os.path.join(sidecar_dir, f"sigma_{self.STOCK_FN}.json")
                   if sidecar_dir else None)
        self._binding = SigmaEMABinding(
            std=self._std, ema_rate=ema_rate, sigma_floor_frac=floor_frac,
            active=active, sidecar_path=sidecar, term_name=self.STOCK_FN)
        self._trace_path = os.environ.get("SONIC_TIER0_TRACE") or None
        self._log_every = int(os.environ.get("SONIC_TIER0_LOG_EVERY", "0")) or None

    def reset(self, env_ids=None):  # noqa: D401 - isaaclab hook
        # Persist sigma at episode/segment boundaries so a resume continues
        # the trajectory (F8). Do NOT reset sigma — PBHC sigma is monotone
        # across the whole run, not per-episode.
        self._binding.save()

    def __call__(self, env, command_name, std, body_names=None):
        # NOTE: params are declared EXPLICITLY (not **kwargs) because
        # isaaclab statically validates the __call__ signature against the
        # term's `params` keys (manager_base._resolve_common_term_cfg): a
        # bare **kwargs shows up as a param named 'kwargs' and FAILS the
        # set-equality check at startup. The three wrapped SONIC tracking
        # funcs all share (env, command_name, std, body_names=None).
        kwargs = {"command_name": command_name, "std": std}
        if body_names is not None:
            kwargs["body_names"] = body_names
        r_stock = self._stock(env, **kwargs)          # (num_envs,) in [0,1]
        if not self._binding.active:
            return r_stock                            # bit-identical no-op

        # Recover mean LINEAR error on-device: d = -std**2 * log(r), then
        # sqrt; clamp r to (floor,1] to avoid log(0). Mean over envs -> scalar.
        # Use the term's OWN std param (not construction std) so a config
        # override of std stays consistent with the recovered error.
        std2 = float(std) * float(std)
        r_clamped = r_stock.clamp(min=1e-30, max=1.0)
        d = -std2 * torch.log(r_clamped)
        mean_lin_err = float(torch.sqrt(d).mean().item())

        exponent = self._binding.observe_and_factor(
            mean_r_stock_batch=None, mean_linear_err=mean_lin_err)

        if self._trace_path and self._log_every and \
                self._binding.n_steps % self._log_every == 0:
            self._append_trace()

        if exponent == 1.0:
            return r_stock                            # passthrough
        return r_clamped ** exponent                  # r_active

    def _append_trace(self):
        try:
            with open(self._trace_path, "a") as fh:
                fh.write(json.dumps(self._binding.trace_record()) + "\n")
        except Exception:  # pragma: no cover - logging must never crash training
            pass


# ── concrete terms (one per wrappable std-bearing tracking term) ─────────
class SigmaEMAAnchorPos(_SigmaEMARewardTerm):
    STOCK_FN = "tracking_anchor_pos_error"


class SigmaEMAAnchorOri(_SigmaEMARewardTerm):
    STOCK_FN = "tracking_anchor_ori_error"


class SigmaEMARelBodyPos(_SigmaEMARewardTerm):
    STOCK_FN = "tracking_relative_body_pos_error"
