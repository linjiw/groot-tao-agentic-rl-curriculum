# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Pure-numeric core of the SONIC sigma-EMA reward-term shim (I1 spike).

This module is deliberately torch-free and isaaclab-free so it is
host-CPU unit-testable without the container. The thin torch/isaaclab
wrapper (`sonic_sigma_ema_term.py`) imports it and applies the retuning
to a real reward tensor inside SONIC training.

The one identity this whole insertion rests on
==============================================
SONIC's tracking-reward kernel is (rewards.py:78, and siblings):

    r_stock(x) = exp(-d / std**2)              # d = squared tracking error

PBHC (doc 09 §2) retunes the tolerance sigma in r(x)=exp(-x/sigma**2-form);
concretely we replace the fixed std**2 denominator with a live value s:

    r_active(x) = exp(-d / s)

Because d = -std**2 * log(r_stock), we can compute r_active from r_stock
ALONE — never touching SONIC internals or recomputing the error:

    r_active = exp( (std**2 / s) * log(r_stock) ) = r_stock ** (std**2 / s)

Consequences that make this safe:
  * NO-OP GUARANTEE: when s == std**2 the exponent is exactly 1.0 and
    r_active is r_stock unchanged (bit-identical: `x ** 1.0 == x` in IEEE
    for finite x >= 0). This is the G0 gate-0 property.
  * We recover the per-env squared error for the EMA as
    d = -std**2 * log(r_stock), clamped at r_stock>0 (r=0 => d=+inf; we
    treat r==0 as "error at/above the max the kernel can express" and feed
    a large finite sentinel rather than inf, see recover_error).
  * The sigma state machine itself is the already-tested
    core.controllers.SigmaEMAController — we do not reimplement it.

`s` here is sigma**2 (the DENOMINATOR), because SONIC parameterises the
kernel by std with exp(-d/std**2). SigmaEMAController tracks a tolerance
`sigma`; we map denominator s = sigma**2 and feed the controller the
LINEAR error sqrt(d) so its EMA lives in the same units as std. See
SigmaEMABinding for the exact mapping.
"""

from __future__ import annotations

import math
from typing import List, Sequence

# r below this is treated as "kernel-saturated" (error >= the max the
# exp kernel resolves) so we never take log(0) = -inf. 1e-30 corresponds
# to d/std**2 ~= 69, i.e. tracking error ~8.3 std out — already saturated.
_R_FLOOR = 1e-30


def recover_squared_error(r_stock: float, std: float) -> float:
    """Invert r_stock = exp(-d/std**2) to get the squared error d >= 0.

    r_stock is clamped to (_R_FLOOR, 1] first: r > 1 is impossible for the
    kernel (clamped to 1 -> d=0); r <= 0 is saturation (-> d from _R_FLOOR).
    """
    if not math.isfinite(r_stock):
        raise ValueError(f"r_stock must be finite, got {r_stock}")
    r = min(1.0, max(_R_FLOOR, r_stock))
    return -(std * std) * math.log(r)


def retune_factor(std: float, sigma: float) -> float:
    """Exponent p such that r_active = r_stock ** p, with s = sigma**2.

    p = std**2 / sigma**2. p == 1.0 exactly when sigma == std (no-op).
    """
    if not (std > 0) or not math.isfinite(std):
        raise ValueError(f"std must be finite and > 0, got {std}")
    if not (sigma > 0) or not math.isfinite(sigma):
        raise ValueError(f"sigma must be finite and > 0, got {sigma}")
    return (std * std) / (sigma * sigma)


def apply_retune(r_stock: float, exponent: float) -> float:
    """r_active = r_stock ** exponent, bit-identical passthrough at exp==1.0."""
    if exponent == 1.0:
        return r_stock            # exact passthrough — the no-op guarantee
    r = min(1.0, max(_R_FLOOR, r_stock))
    return r ** exponent


def mean_linear_error(r_stock_batch: Sequence[float], std: float) -> float:
    """Mean over the batch of the LINEAR tracking error sqrt(d).

    This is the scalar fed to SigmaEMAController.update() so its EMA lives
    in std units (a tolerance), matching the PBHC sigma<-EMA(err) form.
    Empty batch returns 0.0 (no observation).
    """
    if not r_stock_batch:
        return 0.0
    acc = 0.0
    for r in r_stock_batch:
        acc += math.sqrt(recover_squared_error(r, std))
    return acc / len(r_stock_batch)


def retune_batch(r_stock_batch: Sequence[float], exponent: float) -> List[float]:
    """Vectorised apply_retune over a python batch (test/reference path;
    the torch shim does this on-device)."""
    return [apply_retune(r, exponent) for r in r_stock_batch]
