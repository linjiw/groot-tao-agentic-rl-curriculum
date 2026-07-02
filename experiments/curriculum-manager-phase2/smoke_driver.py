# SPDX-License-Identifier: Apache-2.0
"""Manager ON-vs-OFF driver v3: per-segment eval passes + eval-side tripwire
(design doc 08 §8 Phase 2; v2 review residuals 4–6 partially retired).

Loop per segment (manager arm):
  launch_segment → wait → parse_segment → eval_segment (im_eval at FIXED
  relaxed thresholds) → build_digest → policy.observe → [pending gate] →
  policy.propose → registry.validate → apply (next segment's knobs) or
  rollback → journal

Control arm: identical segments incl. the eval pass (its eval curve is the
comparison anchor), no knob changes ever.

v3 changes over v2 (SMOKE_RESULTS.md "Next" items 2, 5, 6):
- **Per-segment eval pass** (`JobAdapter.eval_segment`): after every segment
  both arms run an eval-only `im_eval` at the relaxed FIXED thresholds
  (`terminations/tracking/eval.yaml`) on the segment snapshot. Eval is
  deterministic (verified 2026-07-02: same ckpt → byte-identical metrics),
  so per-segment arm comparisons are noise-free given the pinned seed.
- **Eval-side tripwire**: when eval records exist the policy's tripwire is
  `eval/progress_rate`. IMPORTANT (review M1): the eval scoreboard is NOT
  automatically outside the action space — eval loads the checkpoint's
  config.yaml (with manager overrides) and merges eval.yaml on top, so
  terms eval.yaml does not name leak through. build_eval_command re-pins
  foot_pos_xyz at stock; the boundary holds only for explicitly pinned/
  overridden knobs (structurally tested in the adapter). Falls back to
  train-side `Episode/rew_mean` when no eval stream exists (unit tests).
  NOTE: this is still not the doc-08 held-out PROTECTED metric — both
  motions are in the training set (bones-seed gated). The eval pass breaks
  the threshold-inflation coupling (once pinned), not the
  train-on-eval-keys coupling.
- **Observe-during-gated-ticks** (v2 residual 5): the policy sees every
  digest via `observe()`; the pending gate only suppresses `propose()`.
  Sustain histories no longer have holes across gated segments.
- **Journal provenance** (v2 residual 6, partial): applied decisions carry
  `digest_hash` (sha256 of the digest they were made from) and
  `applied_at_iter` (cumulative training iteration). `expected_effect`
  scoring is still NOT implemented — outcome remains `survived`.

**Honest scope (bones-seed still gated, 2-motion library):**
- 2 motions make sampler-health decisions near-meaningless; curriculum-VALUE
  claims still need the real library + held-out split. What v3 adds is an
  honest per-segment SCOREBOARD: progress_rate + mpjpe at fixed thresholds,
  which loosening cannot mechanically inflate (unlike episode length/reward).
- progress_rate and mpjpe must be read JOINTLY: mpjpe is averaged over
  executed frames only, so short-episode runs look artificially precise
  (measured on the 10k baseline: mpjpe_g 36→61 as survival lengthened).

Policy: TrainSideBandPolicy — decision signal is still the training-side
episode-length band targeting the BINDING termination axis (v2), but the
tripwire guard is eval-side when available.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


knob_registry = _load("knob_registry", "skills/agentic/sonic-knob-registry/knob_registry.py")
digest_builder = _load("digest_builder", "skills/agentic/sonic-run-digest/digest_builder.py")
job_adapter = _load("job_adapter", "skills/agentic/sonic-job-adapter/job_adapter.py")


def digest_hash(digest: Dict[str, Any]) -> str:
    """Stable short hash of a digest (journal provenance, doc 08 §3 step 4)."""
    canon = json.dumps(digest, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


@dataclasses.dataclass
class TrainSideBandPolicy:
    """Episode-length band stepper (training-side decision signal).

    Targets the BINDING termination axis: the threshold knob whose term has
    the highest termination fraction in the digest (loosening a term that
    never fires provably changes nothing — measured in the first smoke run,
    where anchor_pos sat at 0.0000 while foot_pos_xyz terminated 56% of
    episodes and the "managed" arm's trajectory was identical to control).

    Tripwire: eval-side `eval/progress_rate` when the digest carries an eval
    section (fixed relaxed thresholds outside the action space), else
    training-side `Episode/rew_mean` (labeled; unit-test/fallback mode).
    """

    len_low: float = 20.0     # mean episode length below this = terminating too early
    len_high: float = 200.0   # above this = competent enough to tighten
    sustain: int = 2          # consecutive segments required
    notch: float = 0.05

    # digest termination-term name -> registry knob
    TERM_TO_KNOB = {
        "anchor_pos": "termination_threshold.anchor_pos",
        "ee_body_pos": "termination_threshold.ee_body_pos",
        "foot_pos_xyz": "termination_threshold.foot_pos_xyz",
    }

    def __post_init__(self):
        self._len_history: List[float] = []

    def observe(self, digest: Dict[str, Any]) -> None:
        """Update sustain history from a digest WITHOUT proposing — called
        every tick, including gated ones (v2 residual 5: gated segments must
        not leave holes in the sustain history)."""
        train = digest.get("train") or {}
        len_stats = train.get("Episode/len_mean")
        if len_stats and len_stats.get("last") is not None:
            self._len_history.append(len_stats["last"])

    def _binding_knob(self, digest) -> tuple:
        """(knob, fraction) for the axis with highest WINDOWED termination
        mean (single-iteration fractions are too noisy — review finding 6)."""
        train = digest.get("train") or {}
        terms = (train.get("termination_terms_mean_recent")
                 or train.get("termination_terms_last") or {})
        candidates = {t: f for t, f in terms.items()
                      if t in self.TERM_TO_KNOB and isinstance(f, (int, float))}
        if not candidates or max(candidates.values()) <= 0:
            return None, None
        top = max(candidates, key=candidates.get)
        return self.TERM_TO_KNOB[top], candidates[top]

    def _tripwire(self, digest: Dict[str, Any]) -> Dict[str, Any]:
        if digest.get("eval"):
            # fixed-threshold eval metric: outside the action space, cannot be
            # inflated by loosening training thresholds. drop_pct=30: stock
            # training showed ~8% natural progress_rate dips between distant
            # checkpoints; 30% relative + the driver's absolute-floor guard
            # avoids spurious rollbacks at tiny early-run values.
            return {"metric": "eval/progress_rate", "drop_pct": 30, "evals": 2}
        return {"metric": "Episode/rew_mean", "drop_pct": 20, "evals": 2}

    def propose(self, digest: Dict[str, Any], state, registry) -> Dict[str, Any]:
        # history is maintained by observe(); propose() only reads it
        if not self._len_history:
            return {"action": "none", "reason": "no episode-length signal"}
        recent = self._len_history[-self.sustain:]
        full = len(recent) == self.sustain

        knob, frac = self._binding_knob(digest)
        if knob is None:
            return {"action": "none",
                    "reason": "no binding threshold-termination term in digest"}
        cur = float(registry.current_of(knob, state))
        lo, hi = (float(x) for x in registry.knobs[knob]["hard_range"])
        tripwire = self._tripwire(digest)
        guard = ("eval-side fixed-threshold guard" if tripwire["metric"].startswith("eval/")
                 else "training-side guard (no eval stream)")

        if full and all(v < self.len_low for v in recent) and cur + self.notch <= hi:
            return {
                "action": "set", "knob": knob,
                "value": round(cur + self.notch, 4),
                "rationale": (f"mean episode length {[round(v,1) for v in recent]} < "
                              f"{self.len_low} for {self.sustain} segments; {knob} is the "
                              f"binding termination axis (windowed mean {frac:.2f} of "
                              f"episodes; training-side signal; {guard}; no held-out here)"),
                "expected_effect": "episodes lengthen; eval progress_rate does not fall",
                "tripwire": tripwire,
            }
        if full and all(v > self.len_high for v in recent) and cur - self.notch >= lo:
            return {
                "action": "set", "knob": knob,
                "value": round(cur - self.notch, 4),
                "rationale": (f"mean episode length {[round(v,1) for v in recent]} > "
                              f"{self.len_high} for {self.sustain} segments "
                              f"(training-side signal; {guard})"),
                "expected_effect": "tracking precision demand rises; length dips then recovers",
                "tripwire": tripwire,
            }
        return {"action": "none",
                "reason": f"len {[round(v,1) for v in recent]} inside "
                          f"[{self.len_low}, {self.len_high}] band or sustain unmet"}


class SmokeDriver:
    """Composes adapter + digest + policy + registry over real segments."""

    # below this baseline, relative progress_rate drops are noise-level
    # (baseline curve: progress_rate ~0.003 at 2k iters); an armed tripwire
    # additionally requires an absolute drop of this size to breach
    EVAL_ABS_MIN_DROP = 0.002

    def __init__(self, policy, adapter=None, arm: str = "manager",
                 iterations_per_segment: int = 10, window: int = 5,
                 base_knobs: Optional[Dict[str, Any]] = None,
                 num_envs: int = 64, eval_envs: int = 64,
                 run_eval: bool = True, seed: int = 42,
                 initial_checkpoint: Optional[str] = None,
                 project: Optional[str] = None):
        self.policy = policy
        # default project keeps v2's smoke_{arm}; comparison runs pass their
        # own prefix so v2 artifacts aren't overwritten
        self.adapter = adapter or job_adapter.JobAdapter(
            project=project or f"smoke_{arm}", num_envs=num_envs,
            save_last_frequency=5, seed=seed)
        self.arm = arm
        self.iters = iterations_per_segment
        self.window = window
        self.eval_envs = eval_envs
        # eval requires an adapter that implements eval_segment (the fake
        # adapters in the v2 unit tests don't; they exercise the train-side
        # fallback path)
        self.run_eval = run_eval and hasattr(self.adapter, "eval_segment")
        self.registry = knob_registry.load_registry()
        self.state = knob_registry.RunState(tick=0)
        self.knobs: Dict[str, Any] = dict(base_knobs or {})
        # seed the registry's belief with the run's ACTUAL starting values —
        # registry.yaml defaults describe the Stage-2 patch context (loose
        # start 0.30/0.35), but these runs apply overrides on the STOCK
        # config (strict 0.15/0.2). Without seeding, "one notch" from the
        # believed default is a 2x jump from the real value (v2 defect,
        # found 2026-07-02). current_values only — not apply(), which would
        # start cooldown clocks for changes the manager never made.
        for name, value in (base_knobs or {}).items():
            if name in self.registry.knobs:
                self.state.current_values[name] = value
        self.journal: List[Dict[str, Any]] = []
        self.all_train: List[dict] = []
        self.all_sampler: List[dict] = []
        self.all_eval: List[dict] = []
        self.initial_checkpoint = initial_checkpoint
        self.armed: Optional[Dict[str, Any]] = None  # {knob, prev_value, prev_ckpt, baseline, breaches, tw}

    def _knob_state(self):
        return {
            name: {"value": self.registry.current_of(name, self.state),
                   "ticks_since_change": (self.state.tick - self.state.last_changed_tick[name]
                                          if name in self.state.last_changed_tick else None)}
            for name in self.knobs or {}
        }

    def _build_digest(self) -> Dict[str, Any]:
        return digest_builder.build_digest(
            train_records=self.all_train, sampler_records=self.all_sampler,
            eval_records=self.all_eval,
            knob_state=self._knob_state(), decision_history=self.journal[-5:],
            max_over_mean=float(self.knobs.get(
                "adp_samp_failure_rate_max_over_mean", 50.0)),
            window=self.window)

    def _tripwire_value(self, metric: str, train_rew: Optional[float],
                        this_segment_it: Optional[int] = None) -> Optional[float]:
        """Current value of a tripwire metric: eval-side metrics read the
        newest eval record, anything else reads training-side reward.

        `this_segment_it`: when set, an eval-side read returns None unless
        the newest eval record came from THIS segment — otherwise a failed
        eval pass would silently reuse the stale pre-change record, which
        equals the armed baseline and can never breach (review M3: two
        consecutive eval failures scored a change `survived` on zero
        post-change evidence)."""
        if metric.startswith("eval/"):
            key = metric.removeprefix("eval/")
            if not self.all_eval or key not in self.all_eval[-1]:
                return None
            rec = self.all_eval[-1]
            if this_segment_it is not None and rec.get("it") != this_segment_it:
                return None
            return rec[key]
        return train_rew

    def run(self, n_segments: int) -> Dict[str, Any]:
        checkpoint = self.initial_checkpoint
        for i in range(n_segments):
            self.state.tick += 1
            name = f"{self.arm}_s{i+1}"
            seg = self.adapter.launch_segment(name, self.iters, self.knobs,
                                              checkpoint_in=checkpoint)
            self.adapter.wait(seg, poll_s=10, timeout_s=3600)
            parsed = self.adapter.parse_segment(seg)
            if seg.status != "done":
                self.journal.append({"tick": self.state.tick, "segment": name,
                                     "event": "segment_failed",
                                     "tracebacks": parsed.tracebacks})
                break
            # offset iteration numbers so records accumulate monotonically.
            # Console iteration numbering restarts at 1 every segment even on
            # resume (verified: seg2 logs "Learning iteration 1" after
            # "Loaded checkpoint from step 5") — normalize against the first
            # parsed iteration so a numbering change upstream can't double-offset.
            base_it = self.all_train[-1]["it"] if self.all_train else 0
            first_it = parsed.train[0]["it"] if parsed.train else 1
            for r in parsed.train:
                r["it"] += base_it - (first_it - 1)
            for r in parsed.sampler:
                r["it"] += base_it - (first_it - 1)
            self.all_train.extend(parsed.train)
            self.all_sampler.extend(parsed.sampler)
            checkpoint = seg.snapshot or checkpoint
            cum_it = self.all_train[-1]["it"] if self.all_train else 0

            rew = parsed.train[-1].get("Episode/rew_mean") if parsed.train else None
            entry: Dict[str, Any] = {
                "tick": self.state.tick, "segment": name,
                "knobs_in": dict(self.knobs),
                "rew_mean_last": rew,
                "len_mean_last": parsed.train[-1].get("Episode/len_mean") if parsed.train else None,
            }

            # per-segment eval pass at FIXED relaxed thresholds (both arms —
            # the scoreboard neither arm's knobs can reach)
            if self.run_eval:
                try:
                    ev = self.adapter.eval_segment(seg, it=cum_it,
                                                   num_envs=self.eval_envs)
                    self.all_eval.append(ev)
                    entry["eval"] = {k: ev[k] for k in
                                     ("success_rate", "progress_rate",
                                      "mpjpe_all_mean", "mpjpe_l_all_mean",
                                      "mpjpe_pa_all_mean")
                                     if k in ev}
                    if "per_motion" in ev:
                        entry["eval"]["per_motion_progress"] = {
                            k: v.get("progress") for k, v in ev["per_motion"].items()}
                except (RuntimeError, TimeoutError) as e:
                    # an eval failure must not kill the run; the tripwire
                    # watch skips this segment (neither breach nor clean)
                    entry["eval_error"] = str(e)[:300]

            digest = self._build_digest()
            # the policy observes EVERY segment (control excepted) so gated
            # ticks leave no holes in its sustain history (v2 residual 5)
            if self.arm != "control" and hasattr(self.policy, "observe"):
                self.policy.observe(digest)

            # tripwire watch on the armed decision's stated metric
            if self.armed is not None:
                tw = self.armed
                value = self._tripwire_value(tw["tw"]["metric"], rew,
                                             this_segment_it=cum_it)
                if value is None:
                    # missing metric (e.g. failed eval): neither breach nor
                    # clean — the watch window simply extends
                    entry["tripwire_note"] = (
                        f"no {tw['tw']['metric']} value this segment; watch unchanged")
                else:
                    threshold = tw["baseline"] * (1 - tw["tw"]["drop_pct"] / 100.0)
                    breached = value < threshold
                    if tw["tw"]["metric"].startswith("eval/"):
                        # absolute-floor guard: at tiny baselines a relative
                        # drop is noise, not regression
                        breached = breached and (tw["baseline"] - value) > self.EVAL_ABS_MIN_DROP
                    if breached:
                        tw["breaches"] += 1
                        tw["clean"] = 0
                    else:
                        tw["breaches"] = 0
                        tw["clean"] = tw.get("clean", 0) + 1
                    if tw["breaches"] >= tw["tw"]["evals"]:
                        self.knobs[tw["knob"]] = tw["prev_value"]
                        self.state.apply(tw["knob"], tw["prev_value"])
                        checkpoint = tw["prev_ckpt"]
                        entry["event"] = "rollback"
                        entry["restored"] = {tw["knob"]: tw["prev_value"]}
                        # mark the originating decision failed (doc 08 §3 step 5)
                        for prev in reversed(self.journal):
                            if prev.get("applied") and prev["decision"]["knob"] == tw["knob"]:
                                prev["outcome"] = "failed_rolled_back"
                                break
                        self.armed = None
                        self.journal.append(entry)
                        continue
                    if tw["clean"] >= tw["tw"]["evals"]:
                        # survived its watch window: score it and disarm (NOT "met" —
                        # survival of the tripwire is weaker than expected_effect
                        # satisfaction, which is unchecked here)
                        for prev in reversed(self.journal):
                            if prev.get("applied") and prev["decision"]["knob"] == tw["knob"]:
                                prev["outcome"] = "survived"
                                break
                        self.armed = None

            if self.arm == "control":
                entry["decision"] = {"action": "none", "reason": "control arm"}
                self.journal.append(entry)
                continue

            # pending-decision gate (playbook tick-procedure step 2; review
            # finding 1-2): while a change is armed/unscored, no new change —
            # overlapping changes orphan the first tripwire and destroy
            # attribution. propose() is not called (observe() already was).
            if self.armed is not None:
                entry["decision"] = {
                    "action": "none",
                    "reason": f"pending decision on {self.armed['knob']} still "
                              "under tripwire watch (one change at a time)"}
                self.journal.append(entry)
                continue

            decision = self.policy.propose(digest, self.state, self.registry)
            entry["decision"] = decision
            if decision.get("action") == "set":
                res = self.registry.validate_decision(decision, self.state)
                entry["validation"] = {"ok": res.ok, "errors": res.errors}
                # baseline must come from THIS segment's eval — a stale
                # record would arm the watch against pre-change state
                baseline = self._tripwire_value(
                    decision.get("tripwire", {}).get("metric", "Episode/rew_mean"),
                    rew, this_segment_it=cum_it)
                if res.ok and baseline is None:
                    # no baseline → the tripwire would be unarmed in practice;
                    # refuse the change rather than apply it unguarded
                    entry["validation"] = {"ok": False, "errors":
                        [f"no {decision['tripwire']['metric']} baseline available "
                         "to arm the tripwire"]}
                elif res.ok:
                    prev = self.registry.current_of(decision["knob"], self.state)
                    self.armed = {"knob": decision["knob"], "prev_value": prev,
                                  "prev_ckpt": checkpoint,
                                  "baseline": baseline,
                                  "breaches": 0, "clean": 0,
                                  "tw": decision["tripwire"]}
                    self.knobs[decision["knob"]] = decision["value"]
                    self.state.apply(decision["knob"], decision["value"])
                    entry["applied"] = True
                    entry["outcome"] = "pending"
                    entry["digest_hash"] = digest_hash(digest)
                    entry["applied_at_iter"] = cum_it
            self.journal.append(entry)
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        applied = [e for e in self.journal if e.get("applied")]
        return {
            "arm": self.arm,
            "segments": len([e for e in self.journal if "segment" in e]),
            "decisions_applied": len(applied),
            "rollbacks": len([e for e in self.journal if e.get("event") == "rollback"]),
            "rejected": len([e for e in self.journal
                             if e.get("validation") and not e["validation"]["ok"]]),
            "final_knobs": dict(self.knobs),
            "rew_series": [e.get("rew_mean_last") for e in self.journal],
            "len_series": [e.get("len_mean_last") for e in self.journal],
            "eval_progress_series": [
                (e.get("eval") or {}).get("progress_rate") for e in self.journal],
            "eval_mpjpe_series": [
                (e.get("eval") or {}).get("mpjpe_all_mean") for e in self.journal],
        }


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="manager ON-vs-OFF driver (v3: per-segment eval)")
    p.add_argument("--arm", choices=["manager", "control"], required=True)
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--eval-envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-eval", action="store_true",
                   help="skip per-segment eval passes (v2 behavior)")
    p.add_argument("--initial-checkpoint",
                   help="start segment 1 from this checkpoint instead of fresh")
    p.add_argument("--project",
                   help="container log/artifact prefix (default smoke_<arm>)")
    p.add_argument("--base-knobs",
                   help="JSON dict of the run's ACTUAL starting knob values "
                        "(e.g. stock strict thresholds); applied as explicit "
                        "overrides from segment 1 and seeded into the "
                        "registry belief so notch arithmetic starts from the "
                        "real values, not registry.yaml defaults")
    p.add_argument("--len-low", type=float, default=20.0)
    p.add_argument("--sustain", type=int, default=2)
    p.add_argument("--journal-out")
    args = p.parse_args(argv)

    base_knobs = json.loads(args.base_knobs) if args.base_knobs else None
    policy = TrainSideBandPolicy(len_low=args.len_low, sustain=args.sustain)
    driver = SmokeDriver(policy, arm=args.arm, iterations_per_segment=args.iters,
                         num_envs=args.num_envs, eval_envs=args.eval_envs,
                         run_eval=not args.no_eval, seed=args.seed,
                         initial_checkpoint=args.initial_checkpoint,
                         project=args.project, base_knobs=base_knobs)
    summary = driver.run(args.segments)
    print(json.dumps(summary, indent=2))
    if args.journal_out:
        with open(args.journal_out, "w") as f:
            json.dump(driver.journal, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
