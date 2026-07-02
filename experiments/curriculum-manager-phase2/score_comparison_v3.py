# SPDX-License-Identifier: Apache-2.0
"""Score the v3 ON-vs-OFF comparison from the two journals + the persisted
per-segment metrics_eval.json files.

Scoreboard (per baseline-eval-diagnosis RESULTS.md §6):
- PRIMARY: eval progress_rate at fixed relaxed thresholds (deterministic,
  cannot be inflated by loosening training thresholds)
- SECONDARY: eval mpjpe_l (local pose error; drift-insensitive)
- REPORTED WITH WARNING: mpjpe_g — executed-frame survivor bias makes it
  ANTI-correlated with survival (release model: 120.9 vs failing baseline's
  60.7); never read as lower-is-better across different survival lengths
- context only: training-side len/rew (definitionally coupled to the
  manager's own loosening actions)

Eval metrics are read from the persisted container files
(cmp_<arm>_<arm>_sN_eval/metrics_eval.json) so both arms are scored from
identical sources regardless of which driver version wrote the journal.

Usage: python3.10 score_comparison_v3.py control_journal_v3.json manager_journal_v3.json
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

BASE_DIR = "/workspace/wbc-training-logs"


def _container_eval(arm: str, n_segments: int):
    """Read per-segment metrics_eval.json from the container; None per
    segment on any failure (missing file, docker unavailable).

    Prefers `..._eval_pinned/` over `..._eval/` when present: the pinned
    dirs are re-evals with foot_pos_xyz re-pinned at stock 0.2 (review M1 —
    the original manager s3–s6 evals leaked the manager's own 0.25 into
    the scoreboard via the checkpoint config merge)."""
    out = []
    for i in range(1, n_segments + 1):
        metrics = None
        for suffix in ("_eval_pinned", "_eval"):
            path = f"{BASE_DIR}/cmp_{arm}_{arm}_s{i}{suffix}/metrics_eval.json"
            try:
                raw = subprocess.run(
                    ["docker", "exec", "isaac-lab-base", "cat", path],
                    capture_output=True, text=True, timeout=30)
                if raw.returncode == 0:
                    metrics = json.loads(raw.stdout)
                    metrics["_eval_source"] = suffix
                    break
            except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
                pass
        out.append(metrics)
    return out


def _f(metrics, key):
    if not metrics:
        return None
    v = metrics.get(key)
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def series(journal, path):
    out = []
    for e in journal:
        v = e
        for k in path:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        out.append(v)
    return out


def fmt(xs):
    return [round(x, 4) if isinstance(x, (int, float)) else None for x in xs]


def main(control_path: str, manager_path: str) -> int:
    ctrl = json.load(open(control_path))
    mgr = json.load(open(manager_path))

    report = {"segments": {"control": len(ctrl), "manager": len(mgr)}}

    # eval metrics from the persisted container files (source of truth)
    ev = {arm: _container_eval(arm, n)
          for arm, n in (("control", len(ctrl)), ("manager", len(mgr)))}
    for name, key in (
        ("eval_progress_rate_PRIMARY", "eval/success/progress_rate"),
        ("eval_mpjpe_l_SECONDARY", "eval/all/mpjpe_l"),
        ("eval_mpjpe_g_SURVIVOR_BIASED", "eval/all/mpjpe_g"),
    ):
        report[name] = {arm: fmt([_f(m, key) for m in ev[arm]]) for arm in ev}
    report["eval_sources"] = {
        arm: [(m or {}).get("_eval_source") for m in ev[arm]] for arm in ev}
    # per-motion progress (adaptive sampling acts on exactly this asymmetry)
    report["eval_per_motion_progress"] = {
        arm: [
            (dict(zip(m["eval/all_metrics_dict"]["motion_keys"],
                      fmt(m["eval/all_metrics_dict"]["progress"])))
             if m and isinstance(m.get("eval/all_metrics_dict"), dict) else None)
            for m in ev[arm]]
        for arm in ev}

    # training-side context (from journals)
    for name, path in (("train_len_mean_CONTEXT", ["len_mean_last"]),
                       ("train_rew_mean_CONTEXT", ["rew_mean_last"])):
        report[name] = {
            "control": fmt(series(ctrl, path)),
            "manager": fmt(series(mgr, path)),
        }
    report["manager_decisions"] = [
        {"tick": e["tick"], "knob": e["decision"]["knob"],
         "value": e["decision"]["value"], "outcome": e.get("outcome"),
         "applied_at_iter": e.get("applied_at_iter"),
         "digest_hash": e.get("digest_hash")}
        for e in mgr if e.get("applied")
    ]
    report["rollbacks"] = {
        "control": sum(1 for e in ctrl if e.get("event") == "rollback"),
        "manager": sum(1 for e in mgr if e.get("event") == "rollback"),
    }
    report["eval_errors"] = {
        "control": [e["eval_error"] for e in ctrl if e.get("eval_error")],
        "manager": [e["eval_error"] for e in mgr if e.get("eval_error")],
    }

    # divergence check: identical-prefix property under the pinned seed —
    # arms must match to high precision until the first applied decision
    first_change_tick = min((e["tick"] for e in mgr if e.get("applied")),
                            default=None)
    if first_change_tick is not None:
        pre = []
        for a, b in zip(ctrl, mgr):
            if a["tick"] >= first_change_tick:
                break
            la, lb = a.get("len_mean_last"), b.get("len_mean_last")
            pre.append(la == lb)
        report["prefix_identical_before_first_change"] = all(pre) if pre else None
        report["first_change_tick"] = first_change_tick

    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
