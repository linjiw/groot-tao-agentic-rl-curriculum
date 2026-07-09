# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Per-tick observation (digest) builder — engine-agnostic.

Migrated from skills/agentic/sonic-run-digest/digest_builder.py with the
engine couplings removed:

- `TRAIN_SCALAR_KEYS` (the trainer metric names to summarize) was a
  SONIC-specific module constant (TRLPPOTrainer log keys). It is now a
  `train_scalar_keys` parameter of `build_digest` / `build_train_section`.
  `DEFAULT_TRAIN_SCALAR_KEYS` keeps only the two engine-neutral episode
  aggregates the run-manager loop itself reads (Episode/rew_mean,
  Episode/len_mean via journal rew/len fields); everything else must be
  injected per engine.
- The per-term prefixes ("Episode/", "Episode_Termination/",
  "scheduled_params/") are likewise parameters with the same defaults, so
  a TAO adapter can map its own metric namespace without touching core.

Stats/trend semantics (mean, slope, trend labels, entropy, cap
saturation, top-k share, windowing) are unchanged.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA_VERSION = "0.1.0"

# engine-neutral minimum: the aggregates the run-manager loop itself
# consumes. Engine adapters inject their full trainer-key list.
DEFAULT_TRAIN_SCALAR_KEYS = (
    "Episode/rew_mean",
    "Episode/len_mean",
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i + 1}: bad JSON: {e}") from e
    return records


# ── small stats helpers (no numpy dependency) ────────────────────────
def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _slope(xs: List[float]) -> Optional[float]:
    """Least-squares slope per index step; None if < 2 points."""
    n = len(xs)
    if n < 2:
        return None
    mx = (n - 1) / 2.0
    my = _mean(xs)
    denom = sum((i - mx) ** 2 for i in range(n))
    return sum((i - mx) * (y - my) for i, y in enumerate(xs)) / denom


def _trend_label(slope: Optional[float], scale: float, tol: float = 0.02) -> str:
    """'rising' / 'falling' / 'flat' with slope measured relative to scale."""
    if slope is None or scale <= 0:
        return "unknown"
    rel = slope / scale
    if rel > tol:
        return "rising"
    if rel < -tol:
        return "falling"
    return "flat"


def normalized_entropy(probs_or_rates: List[float]) -> Optional[float]:
    """Shannon entropy of the normalized vector, / log(n) → [0, 1]."""
    xs = [max(0.0, float(x)) for x in probs_or_rates]
    total = sum(xs)
    n = len(xs)
    if n < 2 or total <= 0:
        return None
    h = -sum((x / total) * math.log(x / total) for x in xs if x > 0)
    return h / math.log(n)


def cap_saturation_fraction(rates: List[float], max_over_mean: float) -> Optional[float]:
    """Fraction of bins at/above the cap = mean(rates) * max_over_mean."""
    if not rates:
        return None
    mean = _mean(rates)
    if mean is None or mean <= 0:
        return 0.0
    cap = mean * max_over_mean
    return sum(1 for r in rates if r >= cap) / len(rates)


def top_k_share(rates: List[float], k: int = 10) -> Optional[float]:
    """Share of total (normalized) mass held by the k largest bins."""
    xs = sorted((max(0.0, float(x)) for x in rates), reverse=True)
    total = sum(xs)
    if not xs or total <= 0:
        return None
    return sum(xs[:k]) / total


# ── section builders ─────────────────────────────────────────────────
def summarize_series(values: List[float], window: int) -> Dict[str, Any]:
    recent = values[-window:]
    scale = max(abs(v) for v in recent) if recent else 0.0
    slope = _slope(recent)
    return {
        "last": recent[-1] if recent else None,
        "mean_recent": _mean(recent),
        "slope_recent": slope,
        "trend": _trend_label(slope, scale if scale > 0 else 1.0),
        "n_points": len(recent),
    }


def build_eval_section(eval_records: List[Dict[str, Any]], window: int) -> Dict[str, Any]:
    succ = [r["success_rate"] for r in eval_records if "success_rate" in r]
    progress = [r["progress_rate"] for r in eval_records if "progress_rate" in r]
    heldout = [r["heldout_success_rate"] for r in eval_records if "heldout_success_rate" in r]
    mpjpe = [r["mpjpe_all_mean"] for r in eval_records if "mpjpe_all_mean" in r]

    section: Dict[str, Any] = {
        "n_evals": len(eval_records),
        "success_rate": summarize_series(succ, window),
        # smoke-scale scoreboard: success_rate can sit at 0.0 for entire
        # runs — progress_rate moves first. Read jointly with mpjpe
        # (executed-frame survivor bias).
        "progress_rate": summarize_series(progress, window) if progress else None,
        "heldout_success_rate": summarize_series(heldout, window) if heldout else None,
        "mpjpe_all_mean": summarize_series(mpjpe, window) if mpjpe else None,
    }

    # failed_keys diff: newly failing and newly recovered vs previous eval
    keyed = [r for r in eval_records if isinstance(r.get("failed_keys"), list)]
    if keyed:
        curr = set(keyed[-1]["failed_keys"])
        prev = set(keyed[-2]["failed_keys"]) if len(keyed) >= 2 else set()
        section["failed_keys"] = {
            "count": len(curr),
            "newly_failing": sorted(curr - prev),
            "newly_recovered": sorted(prev - curr),
            # persistent = failing in every eval in the window
            "persistent": sorted(
                set.intersection(*(set(r["failed_keys"]) for r in keyed[-window:]))
            )
            if keyed[-window:]
            else [],
        }
    else:
        section["failed_keys"] = None
    return section


def build_sampler_section(
    sampler_records: List[Dict[str, Any]],
    max_over_mean: float,
    window: int,
    top_k: int = 10,
) -> Optional[Dict[str, Any]]:
    recs = [r for r in sampler_records if isinstance(r.get("failure_rate"), list)]
    if not recs:
        return None
    entropies = [
        e for e in (normalized_entropy(r["failure_rate"]) for r in recs) if e is not None
    ]
    last = recs[-1]["failure_rate"]
    return {
        "n_snapshots": len(recs),
        "num_bins": len(last),
        "failure_rate_mean": _mean(last),
        "failure_rate_max": max(last) if last else None,
        "normalized_entropy": summarize_series(entropies, window),
        "cap_saturation_fraction": cap_saturation_fraction(last, max_over_mean),
        "top_k_share": {"k": top_k, "share": top_k_share(last, top_k)},
    }


def build_train_section(
    train_records: List[Dict[str, Any]],
    window: int,
    train_scalar_keys: Sequence[str] = DEFAULT_TRAIN_SCALAR_KEYS,
    episode_prefix: str = "Episode/",
    termination_prefix: str = "Episode_Termination/",
    scheduled_prefix: str = "scheduled_params/",
) -> Optional[Dict[str, Any]]:
    if not train_records:
        return None
    section: Dict[str, Any] = {"n_iterations": len(train_records)}
    for key in train_scalar_keys:
        vals = [r[key] for r in train_records if isinstance(r.get(key), (int, float))]
        if vals:
            section[key] = summarize_series(vals, window)
    # per-term episode reward means — last value only, sorted
    last = train_records[-1]
    episode_terms = {
        k.split("/", 1)[1]: v
        for k, v in last.items()
        if k.startswith(episode_prefix) and isinstance(v, (int, float))
    }
    section["episode_terms_last"] = dict(sorted(episode_terms.items())) or None
    # per-term termination fractions (which threshold is binding?) — last
    # value plus a windowed mean (single-iteration fractions are noisy at
    # small env counts; axis-selection decisions should use the mean)
    termination_terms = {
        k.split("/", 1)[1]: v
        for k, v in last.items()
        if k.startswith(termination_prefix) and isinstance(v, (int, float))
    }
    section["termination_terms_last"] = dict(sorted(termination_terms.items())) or None
    recent = train_records[-window:]
    term_means: Dict[str, float] = {}
    for term in termination_terms:
        vals = [r[f"{termination_prefix}{term}"] for r in recent
                if isinstance(r.get(f"{termination_prefix}{term}"), (int, float))]
        if vals:
            term_means[term] = _mean(vals)
    section["termination_terms_mean_recent"] = dict(sorted(term_means.items())) or None
    scheduled = {
        k.split("/", 1)[1]: v
        for k, v in last.items()
        if k.startswith(scheduled_prefix)
    }
    section["scheduled_params_last"] = scheduled or None
    return section


def build_digest(
    train_records: Iterable[Dict[str, Any]] = (),
    eval_records: Iterable[Dict[str, Any]] = (),
    sampler_records: Iterable[Dict[str, Any]] = (),
    knob_state: Optional[Dict[str, Any]] = None,
    decision_history: Optional[List[Dict[str, Any]]] = None,
    max_over_mean: float = 50.0,
    window: int = 5,
    train_scalar_keys: Sequence[str] = DEFAULT_TRAIN_SCALAR_KEYS,
) -> Dict[str, Any]:
    """Assemble the per-tick digest.

    `knob_state`: {knob: {"value": v, "ticks_since_change": n}} — supplied by
    the manager loop from its RunState + registry.
    `decision_history`: recent journal entries incl. "outcome" so the LLM
    sees what its last interventions did (doc 08 §6.5).
    `max_over_mean`: the CURRENT sampler cap ratio (needed to compute cap
    saturation exactly as the sampler does).
    `train_scalar_keys`: the trainer metric names to summarize —
    engine-specific, injected by the adapter/config.
    """
    train_records = list(train_records)
    eval_records = list(eval_records)
    sampler_records = list(sampler_records)

    last_it = None
    for recs in (train_records, eval_records, sampler_records):
        for r in reversed(recs):
            if isinstance(r.get("it"), (int, float)):
                last_it = max(last_it, r["it"]) if last_it is not None else r["it"]
                break

    return {
        "schema_version": SCHEMA_VERSION,
        "window": window,
        "last_iteration": last_it,
        "eval": build_eval_section(eval_records, window) if eval_records else None,
        "sampler": build_sampler_section(sampler_records, max_over_mean, window),
        "train": build_train_section(train_records, window,
                                     train_scalar_keys=train_scalar_keys),
        "knobs": knob_state,
        "decision_history": decision_history or [],
    }
