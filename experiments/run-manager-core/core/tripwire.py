# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tripwire watch state machine + effect scoring (engine-agnostic).

Extracted verbatim from the Phase-2 SmokeDriver
(experiments/curriculum-manager-phase2/smoke_driver.py):

- the armed-watch dict {knob, prev_value, prev_ckpt, baseline, breaches,
  clean, tw} (smoke_driver.py:792-796) becomes `TripwireWatch`;
- the per-tick judgment block (:692-741) becomes `TripwireWatch.assess()`
  returning a structured `TripwireVerdict` — the LOOP owns all journal
  writes and state restoration, so field names/ordering stay byte-
  compatible with the old driver;
- `_tripwire_value` (:437-456) -> `tripwire_value` — including the
  review-M3 guard: with `this_segment_it` set, an eval-side read returns
  None unless the newest eval record came from THIS segment (a failed
  eval must never silently reuse the stale pre-change record, which
  equals the armed baseline and can never breach);
- `_effect_value` / `_score_effect` (:458-490) -> `effect_value` /
  `score_effect` (three-state scoring: survived /
  survived_effect_confirmed / survived_effect_not_observed);
- `EVAL_ABS_MIN_DROP` (:325): at noise-level baselines a big RELATIVE
  drop is not a regression — an eval-side breach additionally requires
  an absolute drop larger than this floor.

Semantics follow the Phase-2 driver, NOT the Phase-0 ArmedTripwire
prototype (replay_harness.py:155-172), which lacks the clean counter,
the absolute floor and the stale-eval refusal.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

# below this baseline, relative progress_rate drops are noise-level
# (baseline curve: progress_rate ~0.003 at 2k iters); an armed tripwire
# additionally requires an absolute drop of this size to breach
# (smoke_driver.py:322-325)
EVAL_ABS_MIN_DROP = 0.002


def tripwire_value(metric: str, train_rew: Optional[float],
                   eval_records: List[Dict[str, Any]],
                   this_segment_it: Optional[int] = None) -> Optional[float]:
    """Current value of a tripwire metric: eval-side metrics read the
    newest eval record, anything else reads training-side reward.

    `this_segment_it`: when set, an eval-side read returns None unless
    the newest eval record came from THIS segment — otherwise a failed
    eval pass would silently reuse the stale pre-change record, which
    equals the armed baseline and can never breach (review M3: two
    consecutive eval failures scored a change `survived` on zero
    post-change evidence). (smoke_driver.py:437-456)"""
    if metric.startswith("eval/"):
        key = metric.removeprefix("eval/")
        if not eval_records or key not in eval_records[-1]:
            return None
        rec = eval_records[-1]
        if this_segment_it is not None and rec.get("it") != this_segment_it:
            return None
        return rec[key]
    return train_rew


def effect_value(metric: str, cum_it: Optional[int],
                 train_records: List[Dict[str, Any]],
                 eval_records: List[Dict[str, Any]]) -> Optional[float]:
    """Current value of an expected_effect_check metric, from streams the
    loop already records — no new plumbing. `eval/*` reads THIS segment's
    eval record (same staleness rule as the tripwire); anything else reads
    the newest train record. (smoke_driver.py:458-468)"""
    if metric.startswith("eval/"):
        return tripwire_value(metric, None, eval_records,
                              this_segment_it=cum_it)
    if train_records and isinstance(train_records[-1].get(metric), (int, float)):
        return train_records[-1][metric]
    return None


_EFFECT_OPS = {">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t,
               ">": lambda v, t: v > t, "<": lambda v, t: v < t}


def score_effect(origin: Dict[str, Any], cum_it: Optional[int],
                 train_records: List[Dict[str, Any]],
                 eval_records: List[Dict[str, Any]]) -> str:
    """Outcome for a change that SURVIVED its tripwire watch (doc 08 §11
    amendment 4 payoff): check the decision's machine-readable
    `expected_effect_check` ({metric, op, value}) against the metric's
    current value. No check / no data → plain 'survived' (honest: the
    effect claim is unscored, not confirmed). Mutates `origin` in place
    (writes origin["effect"]) exactly like the old driver.
    (smoke_driver.py:473-490)"""
    check = origin["decision"].get("expected_effect_check")
    op = _EFFECT_OPS.get((check or {}).get("op", ""))
    if not check or op is None:
        return "survived"
    value = effect_value(check["metric"], cum_it, train_records, eval_records)
    if value is None:
        origin["effect"] = {"check": check, "observed": None,
                            "note": f"no {check['metric']} value at scoring time"}
        return "survived"
    confirmed = op(value, check["value"])
    origin["effect"] = {"check": check, "observed": value, "confirmed": confirmed}
    return "survived_effect_confirmed" if confirmed else "survived_effect_not_observed"


@dataclasses.dataclass
class TripwireVerdict:
    """Structured result of one watch assessment. The loop translates it
    into journal fields / state restoration; this module never touches
    the journal.

    status:
      "hold"     — metric missing this segment (e.g. failed eval): neither
                   breach nor clean, the watch window simply extends
                   (`note` carries the journal tripwire_note text);
      "watching" — counted a breach or a clean segment, window still open;
      "rollback" — consecutive breaches reached tw["evals"]: restore
                   `restored` ({knob: prev_value}) and the pre-change
                   checkpoint;
      "survived" — consecutive clean segments reached tw["evals"]: score
                   the originating decision and disarm.
    """

    status: str
    note: Optional[str] = None
    restored: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class TripwireWatch:
    """One armed tripwire watch — the old driver's `self.armed` dict
    (smoke_driver.py:406, :792-796) as a class.

    `tw` is the decision's tripwire mapping {metric, drop_pct, evals};
    `prev_ckpt` is the pre-change rollback snapshot captured at apply
    time; `baseline` the metric value the watch was armed against.
    """

    knob: str
    prev_value: Any
    prev_ckpt: Optional[str]
    baseline: float
    tw: Dict[str, Any]
    breaches: int = 0
    clean: int = 0
    eval_abs_min_drop: float = EVAL_ABS_MIN_DROP

    @property
    def metric(self) -> str:
        return self.tw["metric"]

    def assess(self, value: Optional[float]) -> TripwireVerdict:
        """One segment's judgment (smoke_driver.py:692-741, judgment part
        only — restoration/journaling stays in the loop)."""
        if value is None:
            # missing metric (e.g. failed eval): neither breach nor
            # clean — the watch window simply extends
            return TripwireVerdict(
                "hold",
                note=f"no {self.tw['metric']} value this segment; watch unchanged")
        threshold = self.baseline * (1 - self.tw["drop_pct"] / 100.0)
        breached = value < threshold
        if self.tw["metric"].startswith("eval/"):
            # absolute-floor guard: at tiny baselines a relative drop is
            # noise, not regression
            breached = breached and (self.baseline - value) > self.eval_abs_min_drop
        if breached:
            self.breaches += 1
            self.clean = 0
        else:
            self.breaches = 0
            self.clean = self.clean + 1
        if self.breaches >= self.tw["evals"]:
            return TripwireVerdict("rollback",
                                   restored={self.knob: self.prev_value})
        if self.clean >= self.tw["evals"]:
            return TripwireVerdict("survived")
        return TripwireVerdict("watching")
