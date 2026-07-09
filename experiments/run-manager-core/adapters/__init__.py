# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Adapters for the engine-agnostic run-manager core. `mock` carries the
formal in-memory EngineAdapter implementations (test/replay double);
real engine adapters (SONIC, TAO, ...) live outside this package."""
