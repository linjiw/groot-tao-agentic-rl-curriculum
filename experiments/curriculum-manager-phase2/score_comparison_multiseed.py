# SPDX-License-Identifier: Apache-2.0
"""Score the v4 multi-seed ON-vs-OFF comparison from per-seed journals +
the persisted per-segment metrics_eval.json files in the container.

Scoreboard (per baseline-eval-diagnosis RESULTS.md §6, identical to
score_comparison_v3.py):
- PRIMARY: eval progress_rate at fixed relaxed thresholds (deterministic,
  cannot be inflated by loosening training thresholds)
- SECONDARY: eval mpjpe_l (local pose error; drift-insensitive)
- REPORTED WITH WARNING: mpjpe_g — executed-frame survivor bias makes it
  ANTI-correlated with survival; never read as lower-is-better across
  different survival lengths
- context only: training-side len/rew (definitionally coupled to the
  manager's own loosening actions)

Aggregation honesty rules baked into the report:
- per-seed tables FIRST, cross-seed mean ± range (min..max) second —
  with n=2..3 seeds a range is reported, NOT a standard deviation, and
  no significance is claimed.
- explicit per-seed prefix-identity check (arms identical until the
  manager's first applied decision) — a broken prefix invalidates that
  seed's attribution, and the report says so.
- progress_rate quantization caveat repeated verbatim: with 2 motions
  progress moves in units of 1/(2*2002) ≈ 0.00025; differences of one
  or two quanta are sub-noise and must not be cited as value evidence.

Usage:
  python3.10 score_comparison_multiseed.py --seeds 42 1337 \
      [--journal-template {arm}_journal_v4_seed{seed}.json] [--out FILE]

Eval metrics are read from the persisted container files
(cmp_<arm>_seed<seed>_<arm>_sN_eval/metrics_eval.json, preferring
*_eval_pinned/ — review M1) so both arms are scored from identical
sources regardless of driver version.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys

BASE_DIR = "/workspace/wbc-training-logs"

QUANTIZATION_CAVEAT = (
    "progress_rate moves in quanta of 1/(2*2002) ~= 0.00025 with 2 motions; "
    "differences of one or two quanta are sub-noise. Cross-seed spread is a "
    "range (min..max), not a CI; with <5 seeds no significance is claimed.")

MPJPE_G_WARNING = (
    "mpjpe_g is survivor-biased (ANTI-correlated with survival); never read "
    "as lower-is-better across different survival lengths.")


# ── pure helpers (unit-tested) ────────────────────────────────────────

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


def parse_seeds(values):
    """Parse seed CLI values: accepts ints and space/comma-separated
    strings ('42 1337' / '42,1337'); rejects empties and duplicates."""
    seeds = []
    for v in values:
        for tok in str(v).replace(",", " ").split():
            seeds.append(int(tok))
    if not seeds:
        raise ValueError("no seeds given")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    return seeds


def mean_range(values):
    """Cross-seed aggregate: {mean, min, max, n} over finite values;
    None if nothing finite (missing evals stay visibly missing rather
    than silently shrinking n without note)."""
    xs = [v for v in values
          if isinstance(v, (int, float)) and math.isfinite(v)]
    if not xs:
        return None
    return {"mean": round(sum(xs) / len(xs), 5),
            "min": round(min(xs), 5), "max": round(max(xs), 5),
            "n": len(xs), "n_missing": len(values) - len(xs)}


def aggregate_final(per_seed_series):
    """{seed: [per-segment values]} -> cross-seed mean±range of each
    seed's FINAL non-None value."""
    finals = {}
    for seed, xs in per_seed_series.items():
        finite = [v for v in xs
                  if isinstance(v, (int, float)) and math.isfinite(v)]
        finals[seed] = finite[-1] if finite else None
    return {"per_seed_final": finals,
            "cross_seed": mean_range(list(finals.values()))}


def aggregate_per_segment(per_seed_series):
    """{seed: [per-segment values]} -> per-segment cross-seed mean±range
    (segment i aggregated across seeds). Ragged lengths tolerated."""
    n_seg = max((len(xs) for xs in per_seed_series.values()), default=0)
    out = []
    for i in range(n_seg):
        out.append(mean_range([xs[i] if i < len(xs) else None
                               for xs in per_seed_series.values()]))
    return out


def prefix_identity(ctrl_journal, mgr_journal):
    """Arms must match len_mean_last exactly until the manager's first
    applied decision (same-seed determinism). Returns dict with
    first_change_tick, prefix_identical (True/False/None=no pre-change
    overlap or no applied change), and n_compared."""
    first_change_tick = min(
        (e["tick"] for e in mgr_journal if e.get("applied")), default=None)
    if first_change_tick is None:
        # control never acts; with no applied change the whole overlap
        # should be identical
        pairs = list(zip(ctrl_journal, mgr_journal))
    else:
        pairs = [(a, b) for a, b in zip(ctrl_journal, mgr_journal)
                 if a.get("tick", 0) < first_change_tick]
    checks = [a.get("len_mean_last") == b.get("len_mean_last")
              for a, b in pairs]
    return {"first_change_tick": first_change_tick,
            "prefix_identical": (all(checks) if checks else None),
            "n_compared": len(checks)}


def _per_motion_map(entry):
    """eval.per_motion_progress dict from a journal entry, or None."""
    pm = ((entry or {}).get("eval") or {}).get("per_motion_progress")
    if isinstance(pm, dict) and pm:
        return {k: v for k, v in pm.items()
                if isinstance(v, (int, float)) and math.isfinite(v)}
    return None


def paired_final_delta(mgr_pm, ctrl_pm):
    """(a)+(b): paired manager-vs-control win/loss/tie over shared motions
    at the final segment, median paired delta, and top-5 motions by
    |delta|. Deltas are manager - control (positive = manager wins)."""
    shared = sorted(set(mgr_pm) & set(ctrl_pm))
    if not shared:
        return None
    deltas = {k: mgr_pm[k] - ctrl_pm[k] for k in shared}
    vals = sorted(deltas.values())
    n = len(vals)
    median = (vals[n // 2] if n % 2 else
              0.5 * (vals[n // 2 - 1] + vals[n // 2]))
    top5 = sorted(shared, key=lambda k: abs(deltas[k]), reverse=True)[:5]
    return {
        "n_motions": n,
        "wins": sum(1 for d in deltas.values() if d > 0),
        "losses": sum(1 for d in deltas.values() if d < 0),
        "ties": sum(1 for d in deltas.values() if d == 0),
        "median_delta": round(median, 6),
        "top5_by_abs_delta": [
            {"motion": k, "manager": round(mgr_pm[k], 6),
             "control": round(ctrl_pm[k], 6), "delta": round(deltas[k], 6)}
            for k in top5],
    }


def largest_jump_decomposition(journal):
    """(c): find the largest |segment-to-segment jump| of the aggregate
    (mean over motions) per-motion progress, and the single motion
    contributing most to it (with its fraction of the jump). The v4
    single-motion artifact detector."""
    pms = [_per_motion_map(e) for e in journal]
    segs = [(i, pm) for i, pm in enumerate(pms) if pm]
    if len(segs) < 2:
        return None
    best = None
    for (i0, pm0), (i1, pm1) in zip(segs, segs[1:]):
        shared = set(pm0) & set(pm1)
        if not shared:
            continue
        jump = (sum(pm1[k] for k in shared) -
                sum(pm0[k] for k in shared)) / len(shared)
        if best is None or abs(jump) > abs(best[0]):
            best = (jump, i0, i1, pm0, pm1, shared)
    if best is None:
        return None
    jump, i0, i1, pm0, pm1, shared = best
    contrib = {k: (pm1[k] - pm0[k]) / len(shared) for k in shared}
    top = max(contrib, key=lambda k: abs(contrib[k]))
    frac = (contrib[top] / jump) if jump != 0 else None
    return {
        "from_segment": i0 + 1, "to_segment": i1 + 1,
        "jump": round(jump, 6),
        "top_motion": top,
        "top_motion_contribution": round(contrib[top], 6),
        "top_motion_fraction_of_jump": (round(frac, 4)
                                        if frac is not None else None),
    }


def leave_one_out(pm, exclude_motion=None):
    """(d): aggregate (mean) final progress with each single motion
    excluded one at a time — min/max and, if given, the value excluding
    `exclude_motion` (the arm's top jump contributor)."""
    if not pm or len(pm) < 2:
        return None
    total, n = sum(pm.values()), len(pm)
    loo = {k: (total - v) / (n - 1) for k, v in pm.items()}
    kmin = min(loo, key=lambda k: loo[k])
    kmax = max(loo, key=lambda k: loo[k])
    out = {
        "full_mean": round(total / n, 6),
        "min": {"excluded_motion": kmin, "value": round(loo[kmin], 6)},
        "max": {"excluded_motion": kmax, "value": round(loo[kmax], 6)},
    }
    if exclude_motion is not None and exclude_motion in loo:
        out["excluding_top_contributor"] = {
            "motion": exclude_motion,
            "value": round(loo[exclude_motion], 6)}
    return out


def per_motion_decomposition(journals):
    """Full per-seed per-motion decomposition (report['per_motion']).
    The standing checks that caught v4's single-motion artifact
    (postmortem_convulsions_* drove 51-96%% of all aggregate jumps).
    Returns {seed: {...}} with None sub-sections when journals carry no
    eval.per_motion_progress (older fixtures stay compatible)."""
    out = {}
    for seed in sorted(journals):
        ctrl_j = journals[seed].get("control") or []
        mgr_j = journals[seed].get("manager") or []
        ctrl_final = next((pm for pm in
                           (_per_motion_map(e) for e in reversed(ctrl_j))
                           if pm), None)
        mgr_final = next((pm for pm in
                          (_per_motion_map(e) for e in reversed(mgr_j))
                          if pm), None)
        paired = (paired_final_delta(mgr_final, ctrl_final)
                  if mgr_final and ctrl_final else None)
        jumps = {"control": largest_jump_decomposition(ctrl_j),
                 "manager": largest_jump_decomposition(mgr_j)}
        loo = {}
        for arm, final in (("control", ctrl_final), ("manager", mgr_final)):
            top = (jumps[arm] or {}).get("top_motion") if jumps[arm] else None
            loo[arm] = leave_one_out(final, exclude_motion=top)
        out[seed] = {
            "final_paired": paired,
            "largest_jump": jumps,
            "leave_one_out_final": loo,
        }
    return out


# ── container I/O (not unit-tested; same access pattern as v3) ────────

def _container_eval(project_prefix: str, arm: str, n_segments: int,
                    runner=subprocess.run):
    """Per-segment metrics_eval.json from the container; None per segment
    on any failure. Prefers `*_eval_pinned/` (review M1 re-pins)."""
    out = []
    for i in range(1, n_segments + 1):
        metrics = None
        for suffix in ("_eval_pinned", "_eval"):
            path = f"{BASE_DIR}/{project_prefix}_{arm}_s{i}{suffix}/metrics_eval.json"
            try:
                raw = runner(["docker", "exec", "isaac-lab-base", "cat", path],
                             capture_output=True, text=True, timeout=30)
                if raw.returncode == 0:
                    metrics = json.loads(raw.stdout)
                    metrics["_eval_source"] = suffix
                    break
            except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
                pass
        out.append(metrics)
    return out


# ── report assembly ───────────────────────────────────────────────────

EVAL_KEYS = (
    ("eval_progress_rate_PRIMARY", "eval/success/progress_rate"),
    ("eval_mpjpe_l_SECONDARY", "eval/all/mpjpe_l"),
    ("eval_mpjpe_g_SURVIVOR_BIASED", "eval/all/mpjpe_g"),
)


def build_report(journals, evals):
    """journals: {seed: {arm: journal_list}}; evals: {seed: {arm:
    [metrics|None per segment]}}. Pure — unit-testable with fixtures."""
    seeds = sorted(journals)
    arms = ("control", "manager")
    report = {
        "seeds": seeds,
        "caveats": {"quantization": QUANTIZATION_CAVEAT,
                    "mpjpe_g": MPJPE_G_WARNING},
        "segments": {s: {a: len(journals[s][a]) for a in arms}
                     for s in seeds},
    }

    # eval scoreboard: per-seed per-segment tables, then aggregates
    for name, key in EVAL_KEYS:
        per_seed = {a: {s: fmt([_f(m, key) for m in evals[s][a]])
                        for s in seeds} for a in arms}
        report[name] = {
            "per_seed_per_segment": per_seed,
            "final_cross_seed": {a: aggregate_final(per_seed[a])
                                 for a in arms},
            "per_segment_cross_seed": {a: aggregate_per_segment(per_seed[a])
                                       for a in arms},
        }
    report["eval_sources"] = {
        s: {a: [(m or {}).get("_eval_source") for m in evals[s][a]]
            for a in arms} for s in seeds}

    # training-side context (journals)
    for name, path in (("train_len_mean_CONTEXT", ["len_mean_last"]),
                       ("train_rew_mean_CONTEXT", ["rew_mean_last"])):
        per_seed = {a: {s: fmt(series(journals[s][a], path)) for s in seeds}
                    for a in arms}
        report[name] = {
            "per_seed_per_segment": per_seed,
            "final_cross_seed": {a: aggregate_final(per_seed[a])
                                 for a in arms},
        }

    report["manager_decisions"] = {
        s: [{"tick": e["tick"], "knob": e["decision"]["knob"],
             "value": e["decision"]["value"], "outcome": e.get("outcome"),
             "applied_at_iter": e.get("applied_at_iter"),
             "digest_hash": e.get("digest_hash")}
            for e in journals[s]["manager"] if e.get("applied")]
        for s in seeds}
    report["rollbacks"] = {
        s: {a: sum(1 for e in journals[s][a] if e.get("event") == "rollback")
            for a in arms} for s in seeds}
    report["eval_errors"] = {
        s: {a: [e["eval_error"] for e in journals[s][a] if e.get("eval_error")]
            for a in arms} for s in seeds}

    # prefix identity PER SEED — the attribution guarantee
    report["prefix_identity_per_seed"] = {
        s: prefix_identity(journals[s]["control"], journals[s]["manager"])
        for s in seeds}
    report["prefix_identity_all_seeds"] = all(
        v["prefix_identical"] is True
        for v in report["prefix_identity_per_seed"].values())

    # per-motion decomposition (additive; v4 single-motion-artifact checks)
    report["per_motion"] = per_motion_decomposition(journals)

    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="multi-seed v4 comparison scorer")
    p.add_argument("--seeds", nargs="+", required=True,
                   help="seeds, e.g. --seeds 42 1337")
    p.add_argument("--journal-template",
                   default="{arm}_journal_v4_seed{seed}.json")
    p.add_argument("--project-template", default="cmp_{arm}_seed{seed}",
                   help="container artifact prefix per arm+seed")
    p.add_argument("--out", help="also write the report JSON here")
    args = p.parse_args(argv)
    seeds = parse_seeds(args.seeds)

    journals, evals = {}, {}
    for s in seeds:
        journals[s], evals[s] = {}, {}
        for arm in ("control", "manager"):
            path = args.journal_template.format(arm=arm, seed=s)
            journals[s][arm] = json.load(open(path))
            prefix = args.project_template.format(arm=arm, seed=s)
            evals[s][arm] = _container_eval(prefix, arm, len(journals[s][arm]))

    report = build_report(journals, evals)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
