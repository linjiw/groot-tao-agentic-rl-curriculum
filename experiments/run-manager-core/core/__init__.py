# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Engine-agnostic run-manager core (Phase B1+B2).

Extracted from the SONIC-specific curriculum-manager Phase-2 stack
(smoke_driver.py / job_adapter.py / knob_registry.py / digest_builder.py)
with every engine-specific constant moved behind constructor/config
injection. Nothing in this package imports gear_sonic, docker plumbing,
or any SONIC yaml default.
"""

from .protocols import (  # noqa: F401
    Decision,
    EngineAdapter,
    ParsedSegment,
    Policy,
    Segment,
    Tripwire,
)
from .journal import (  # noqa: F401
    JOURNAL_ENTRY_FIELD_ORDER,
    append_entry,
    build_segment_entry,
    load_journal,
    save_journal,
)
from .registry import (  # noqa: F401
    ConfigDriftError,
    ConfigVerification,
    KnobRegistry,
    RunState,
    ValidationResult,
    load_registry,
    resolve_config_value,
)
from .digest import build_digest  # noqa: F401
from .equivalence import (  # noqa: F401
    E5B_CHAOS_FLOOR_MEAN,
    E5B_CHAOS_FLOOR_POINTWISE,
    EquivalenceGate,
    GateReport,
    calibrate_tau,
    max_relative_deviation,
    mean_relative_deviation,
)
from .tripwire import (  # noqa: F401
    EVAL_ABS_MIN_DROP,
    TripwireVerdict,
    TripwireWatch,
    effect_value,
    score_effect,
    tripwire_value,
)
from .loop import (  # noqa: F401
    DiskSpaceError,
    LoopConfig,
    RunManager,
    control_config,
    digest_hash,
    scripted_config,
)

__all__ = [
    "Decision", "EngineAdapter", "ParsedSegment", "Policy", "Segment",
    "Tripwire",
    "JOURNAL_ENTRY_FIELD_ORDER", "append_entry", "build_segment_entry",
    "load_journal", "save_journal",
    "ConfigDriftError", "ConfigVerification", "KnobRegistry", "RunState",
    "ValidationResult", "load_registry", "resolve_config_value",
    "build_digest",
    "EquivalenceGate", "GateReport", "calibrate_tau",
    "mean_relative_deviation", "E5B_CHAOS_FLOOR_MEAN",
    "E5B_CHAOS_FLOOR_POINTWISE",
    "max_relative_deviation",
    "EVAL_ABS_MIN_DROP", "TripwireVerdict", "TripwireWatch",
    "effect_value", "score_effect", "tripwire_value",
    "DiskSpaceError", "LoopConfig", "RunManager", "control_config",
    "digest_hash", "scripted_config",
]
