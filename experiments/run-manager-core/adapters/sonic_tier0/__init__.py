# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""SONIC tier-0 controller shims (doc 10 G2/G3).

Insert the engine-agnostic tier-0 controllers (core.controllers) into live
GR00T-WBC training via Hydra `func` overrides + PYTHONPATH — no pinned-
submodule edits (see I1_INSERTION_RECON.md).

  * sigma_ema_kernel   — torch-free numeric core (the r_active = r_stock**p identity)
  * sigma_ema_binding  — SigmaEMAController + SONIC unit mapping + sidecar persistence
  * sonic_sigma_ema_term — torch/isaaclab ManagerTermBase reward-term shim (in-container)

The bin-LP sampler (G3) is not yet here: its seam is a hardcoded class
(commands.py:231), so it needs a monkeypatch entrypoint, deferred per doc 10.
"""
