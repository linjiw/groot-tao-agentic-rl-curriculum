# SPDX-License-Identifier: Apache-2.0
"""Effective-replicate collapse (doc 10 §0.2, made structural per V2).

Fixed-seed RL runs on this box are bit-deterministic modulo rare kernel
events (E5c). The v4 seed-42 "4 replicates" were really 3 distinct
trajectories: rep2 == rep3 bit-exactly on every shared segment. Computing a
standard deviation or CI over such a set OVERSTATES n and understates the
noise band. This module collapses a set of replicate metric series into
DISTINCT trajectories BEFORE any spread statistic is taken.

Definition of "same trajectory": two per-segment metric series are the same
trajectory if they are bit-identical (exact float equality) on every segment
index they share, AND they share at least `min_overlap` segments (so two
1-segment stubs that happen to match the warm-start don't merge). This
mirrors the E6 journal-equivalence gate-0 (bit_identical) semantics but at
the metric-series level and without a tolerance — collapse is only for TRUE
duplicates; near-but-not-equal runs are distinct (that IS the noise we want
to measure).

Engine-agnostic and dependency-free (stdlib only): a "series" is just a
list of floats/None. The caller extracts series from journals however it
likes (e.g. score_comparison_multiseed.series(...)).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

Number = Optional[float]


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def series_bit_identical(a: Sequence[Number], b: Sequence[Number],
                         min_overlap: int = 2) -> bool:
    """True iff a and b are EXACTLY equal on every shared index and overlap
    on >= min_overlap finite-comparable segments.

    None vs None counts as equal (both missing); None vs a number is a
    mismatch (one ran, one didn't) => distinct. Uses exact float equality
    (no tolerance) — this detects TRUE duplicate runs, not merely-close ones.
    """
    n = min(len(a), len(b))
    if n == 0:
        return False
    overlap = 0
    for i in range(n):
        x, y = a[i], b[i]
        x_none, y_none = x is None, y is None
        if x_none and y_none:
            continue                      # both missing at i — not counted, not a mismatch
        if x_none != y_none:
            return False                  # one ran, one didn't => distinct
        if not (_finite(x) and _finite(y)):
            return False                  # NaN/inf present => cannot call identical
        if float(x) != float(y):          # exact equality — the whole point
            return False
        overlap += 1
    return overlap >= min_overlap


def collapse_replicates(named_series: Dict[str, Sequence[Number]],
                        min_overlap: int = 2) -> Dict[str, object]:
    """Group replicate series by bit-identity into distinct trajectories.

    named_series: {label -> per-segment metric series} (e.g.
        {"rep1": [...], "rep2": [...], ...}).

    Returns:
      {
        "groups": [[labels in group1], [labels in group2], ...],  # each group == one trajectory
        "representatives": [label, ...],   # one canonical label per group (first-seen)
        "n_launched": int,                 # len(named_series)
        "n_distinct": int,                 # len(groups) — the EFFECTIVE n
        "collapsed": bool,                 # n_distinct < n_launched
        "duplicate_note": str | None,      # human-readable if collapsed
      }

    Grouping is transitive-by-construction here (identity is an equivalence
    relation on exact-equal-on-overlap series when overlaps are consistent);
    we assign each series to the first existing group whose representative it
    matches, else start a new group. Determinism: input insertion order.
    """
    labels = list(named_series.keys())
    groups: List[List[str]] = []
    reps: List[str] = []
    for label in labels:
        s = named_series[label]
        placed = False
        for gi, rep in enumerate(reps):
            if series_bit_identical(s, named_series[rep], min_overlap=min_overlap):
                groups[gi].append(label)
                placed = True
                break
        if not placed:
            groups.append([label])
            reps.append(label)
    n_launched = len(labels)
    n_distinct = len(groups)
    collapsed = n_distinct < n_launched
    note = None
    if collapsed:
        dups = [g for g in groups if len(g) > 1]
        note = ("bit-identical replicate(s) collapsed before aggregation "
                f"(launched {n_launched}, distinct {n_distinct}): "
                + "; ".join("==".join(g) for g in dups)
                + " — a spread statistic over the launched count would "
                  "overstate n (doc 10 §0.2).")
    return {
        "groups": groups,
        "representatives": reps,
        "n_launched": n_launched,
        "n_distinct": n_distinct,
        "collapsed": collapsed,
        "duplicate_note": note,
    }


def mapping_bit_identical(a: Dict[str, Number], b: Dict[str, Number],
                          min_overlap: int = 2) -> bool:
    """Like series_bit_identical but keyed by SEGMENT NAME, not list index.

    This is the correct comparison for real runs: a replicate that dropped
    one segment's journal entry (v4 rep2 is missing control_s2) is still the
    same trajectory as its twin on every segment they BOTH recorded. Keying
    on position would spuriously call them distinct (the raw-index method is
    deliberately conservative; this one matches the doc-10 §0.2 finding).
    """
    shared = set(a) & set(b)
    overlap = 0
    for k in shared:
        x, y = a[k], b[k]
        x_none, y_none = x is None, y is None
        if x_none and y_none:
            continue
        if x_none != y_none:
            return False
        if not (_finite(x) and _finite(y)):
            return False
        if float(x) != float(y):
            return False
        overlap += 1
    return overlap >= min_overlap


def collapse_by_segment(named_maps: Dict[str, Dict[str, Number]],
                        min_overlap: int = 2) -> Dict[str, object]:
    """Collapse replicates whose per-SEGMENT-NAME metric maps are
    bit-identical on their shared segments. Same return shape as
    collapse_replicates. Use this on real journals (segment names are the
    stable key; list position is not — a dropped entry shifts positions)."""
    labels = list(named_maps.keys())
    groups: List[List[str]] = []
    reps: List[str] = []
    for label in labels:
        m = named_maps[label]
        placed = False
        for gi, rep in enumerate(reps):
            if mapping_bit_identical(m, named_maps[rep], min_overlap=min_overlap):
                groups[gi].append(label)
                placed = True
                break
        if not placed:
            groups.append([label])
            reps.append(label)
    n_launched, n_distinct = len(labels), len(groups)
    collapsed = n_distinct < n_launched
    note = None
    if collapsed:
        dups = [g for g in groups if len(g) > 1]
        note = ("bit-identical replicate(s) collapsed by segment name "
                f"(launched {n_launched}, distinct {n_distinct}): "
                + "; ".join("==".join(g) for g in dups)
                + " — doc 10 §0.2.")
    return {"groups": groups, "representatives": reps,
            "n_launched": n_launched, "n_distinct": n_distinct,
            "collapsed": collapsed, "duplicate_note": note}


def distinct_finals(named_series: Dict[str, Sequence[Number]],
                    min_overlap: int = 2) -> Dict[str, object]:
    """Convenience: collapse, then return the FINAL finite value of each
    distinct trajectory's representative — the correct input to a noise-band
    / sigma_rep computation (one value per distinct run, not per launch)."""
    info = collapse_replicates(named_series, min_overlap=min_overlap)
    finals = []
    for rep in info["representatives"]:
        finite = [v for v in named_series[rep] if _finite(v)]
        finals.append(finite[-1] if finite else None)
    info = dict(info)
    info["distinct_finals"] = finals
    return info


def noise_band_from_replicates(named_maps: Dict[str, Dict[str, Number]],
                               endpoint_segments: Optional[Sequence[str]] = None,
                               min_overlap: int = 2) -> Dict[str, object]:
    """G1 noise-floor helper (doc 10 §4-G1, V2).

    Given replicate runs as per-segment-name metric maps, collapse
    bit-identical duplicates, then compute the endpoint noise band over the
    DISTINCT trajectories only. Endpoint = mean over `endpoint_segments`
    (e.g. the final-2 segment names, the doc-10 endpoint) if given, else the
    single max-named segment per run.

    Returns the collapse info plus:
      distinct_endpoints, mean, min, max, range, rel_range (range/mean),
      sigma (population std, n_distinct), rel_sigma (sigma/mean).
    Refuses (sigma=None) with n_distinct < 2 — a noise band needs >= 2
    distinct runs; that refusal is the honest output, not a fabricated 0.
    """
    info = dict(collapse_by_segment(named_maps, min_overlap=min_overlap))

    def _endpoint(m: Dict[str, Number]) -> Number:
        if endpoint_segments:
            vals = [m[s] for s in endpoint_segments if _finite(m.get(s))]
            return sum(vals) / len(vals) if vals else None
        finite = [(k, v) for k, v in m.items() if _finite(v)]
        if not finite:
            return None
        return max(finite, key=lambda kv: kv[0])[1]   # lexicographic max seg name

    endpoints = []
    for rep in info["representatives"]:
        ep = _endpoint(named_maps[rep])
        if ep is not None:
            endpoints.append(ep)
    info["distinct_endpoints"] = endpoints
    n = len(endpoints)
    if n < 2:
        info.update({"mean": (endpoints[0] if endpoints else None),
                     "min": None, "max": None, "range": None,
                     "rel_range": None, "sigma": None, "rel_sigma": None,
                     "band_note": "need >= 2 distinct trajectories for a "
                                  f"noise band; have {n}."})
        return info
    mean = sum(endpoints) / n
    var = sum((x - mean) ** 2 for x in endpoints) / n     # population std (n)
    sigma = math.sqrt(var)
    rng = max(endpoints) - min(endpoints)
    info.update({
        "mean": mean, "min": min(endpoints), "max": max(endpoints),
        "range": rng, "rel_range": (rng / mean if mean else None),
        "sigma": sigma, "rel_sigma": (sigma / mean if mean else None),
        "band_note": (f"noise band over {n} DISTINCT trajectories "
                      f"(launched {info['n_launched']}); rel_range="
                      f"{(rng/mean if mean else float('nan')):.3g}."),
    })
    return info
