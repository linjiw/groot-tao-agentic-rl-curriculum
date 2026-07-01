# SPDX-License-Identifier: Apache-2.0
"""Build the Curriculum-Manager Agent's per-tick observation (digest.json).

Design doc: docs/design/08-curriculum-manager-agent.md §3 (decision loop,
step 1 "Digest") and §4 (observation space). Pure Python; no torch.

Inputs are three JSONL streams, one record per line, using SONIC's own
metric names (verified against pinned WBC 0e35637):

  train.jsonl   — one record per logged PPO iteration. Keys as emitted by
                  TRLPPOTrainer.log(): "it", "policy/approxkl_avg",
                  "loss/entropy_avg", "loss/value_avg", "loss/policy_avg",
                  "lr", "Policy/mean_noise_std", "Episode/<term>" reward
                  means, "scheduled_params/<name>", ... (ppo_trainer.py:
                  1578-1631, 1874-1905). Extra keys are ignored.
  eval.jsonl    — one record per eval pass (im_eval_callback). Keys:
                  "it", "success_rate", "progress_rate", "failed_keys"
                  (list of motion-key strings), optional "mpjpe_all_mean",
                  optional "heldout_success_rate" (the protected metric —
                  produced by the separate held-out watcher, NOT stock
                  SONIC), optional per-metric "eval/success/*", "eval/all/*"
                  (im_eval_callback.py:747, 811-815, 227-231).
  sampler.jsonl — one record per 200-step adaptive-sampling sync. Keys:
                  "it", "failure_rate" (list of per-bin floats =
                  adp_samp_failure_rate, motion_lib_base.py:2531-2552),
                  optional "cap" (the applied upper bound).

Output: a single JSON-serializable dict (the digest) with summary stats and
trend annotations over the last `window` evals — everything §4's table lists,
nothing per-step. The digest deliberately carries BOTH training-side and
eval-side signals so the playbook can enforce "never tighten on eval-side
evidence alone" (doc 08 §9, eval/train threshold decoupling).
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = "0.1.0"

# trainer keys we summarize (last value + short trend); Episode/* handled by prefix
TRAIN_SCALAR_KEYS = (
    "policy/approxkl_avg",
    "loss/entropy_avg",
    "loss/value_avg",
    "loss/policy_avg",
    "lr",
    "Policy/mean_noise_std",
    "fps",
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
    """Fraction of bins at/above the cap = mean(rates) * max_over_mean.

    Mirrors motion_lib_base.py:2570-2577 (upper bound then clip).
    """
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
    heldout = [r["heldout_success_rate"] for r in eval_records if "heldout_success_rate" in r]
    mpjpe = [r["mpjpe_all_mean"] for r in eval_records if "mpjpe_all_mean" in r]

    section: Dict[str, Any] = {
        "n_evals": len(eval_records),
        "success_rate": summarize_series(succ, window),
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


def build_train_section(train_records: List[Dict[str, Any]], window: int) -> Optional[Dict[str, Any]]:
    if not train_records:
        return None
    section: Dict[str, Any] = {"n_iterations": len(train_records)}
    for key in TRAIN_SCALAR_KEYS:
        vals = [r[key] for r in train_records if isinstance(r.get(key), (int, float))]
        if vals:
            section[key] = summarize_series(vals, window)
    # per-term episode reward means (Episode/<term>) — last value only, sorted
    last = train_records[-1]
    episode_terms = {
        k.split("/", 1)[1]: v
        for k, v in last.items()
        if k.startswith("Episode/") and isinstance(v, (int, float))
    }
    section["episode_terms_last"] = dict(sorted(episode_terms.items())) or None
    scheduled = {
        k.split("/", 1)[1]: v
        for k, v in last.items()
        if k.startswith("scheduled_params/")
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
) -> Dict[str, Any]:
    """Assemble the per-tick digest.

    `knob_state`: {knob: {"value": v, "ticks_since_change": n}} — supplied by
    the manager loop from its RunState + registry.
    `decision_history`: recent journal entries incl. "outcome" so the LLM
    sees what its last interventions did (doc 08 §6.5).
    `max_over_mean`: the CURRENT value of adp_samp_failure_rate_max_over_mean
    (needed to compute cap saturation exactly as the sampler does).
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
        "train": build_train_section(train_records, window),
        "knobs": knob_state,
        "decision_history": decision_history or [],
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build digest.json from SONIC run logs")
    p.add_argument("--train", help="train metrics JSONL")
    p.add_argument("--eval", dest="eval_", help="eval records JSONL")
    p.add_argument("--sampler", help="sampler snapshots JSONL")
    p.add_argument("--max-over-mean", type=float, default=50.0)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--out", default="digest.json")
    args = p.parse_args(argv)

    digest = build_digest(
        train_records=read_jsonl(args.train) if args.train else (),
        eval_records=read_jsonl(args.eval_) if args.eval_ else (),
        sampler_records=read_jsonl(args.sampler) if args.sampler else (),
        max_over_mean=args.max_over_mean,
        window=args.window,
    )
    with open(args.out, "w") as f:
        json.dump(digest, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
