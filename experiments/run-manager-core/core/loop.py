# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Engine-agnostic run-manager loop (Phase B3).

`RunManager.run()` is the Phase-2 SmokeDriver.run() state machine
(experiments/curriculum-manager-phase2/smoke_driver.py:570-808) with every
SONIC coupling moved behind injection. Per-segment step order — preserved
EXACTLY, including the three order invariants (tripwire judgment before
propose; the rollback branch `continue`s without deciding; observe()
before the pending gate):

    [1]  disk gate                       (:576, impl :522-543)
    [2]  adapter.launch_segment          (:577-578)
    [3]  adapter.wait                    (:579)
    [4]  adapter.parse_segment           (:580)
    [5]  failed-segment check -> journal `segment_failed` + break (:581-585)
    [6]  iteration renumbering + stream accumulation; checkpoint =
         seg.snapshot or checkpoint      (:586-599)
    [7]  cum_it / rew + base entry       (:600-608)
    [8]  config verification (raises ConfigDriftError pre-decision) (:615)
    [9]  standard eval pass (non-fatal eval_error)   (:619-646)
    [10] held-out protected-metric pass (hook)       (:653-676)
    [11] checkpoint purge (hook)                     (:683)
    [12] digest build                                (:685)
    [13] policy.observe (flag-gated)                 (:688-689)
    [14] tripwire watch judgment (TripwireWatch.assess) incl. rollback /
         survived scoring                            (:692-741)
    [15] propose-disabled (control) branch           (:743-746)
    [16] pending gate (flag-gated)                   (:755-761)
    [17] policy.propose                              (:763-764)
    [18] open-watch retirement on 'set' (flag-gated) (:765-775)
    [19] 'set' handling: validate, baseline, arm TripwireWatch, apply,
         provenance fields                           (:776-806)
    [20] journal.append(entry)                       (:807)

SONIC impurities removed (each replaced by injection — see LoopConfig):

- job_adapter.JobAdapter default construction (:377-379): adapter is a
  REQUIRED constructor argument.
- knob_registry.load_registry() default action space (:388): registry is
  a REQUIRED constructor argument.
- job_adapter.KNOB_TO_CONFIG_PATH (:513): `knob_config_paths` injection,
  falling back to the adapter's knob_to_config_path() seam.
- job_adapter.BASE_DIR disk-gate auto-arming (:415-416): the gate arms
  ONLY from an explicit `disk_gate_path`.
- Hydra motion_file override strings (:373-374, :628-630, :658-660):
  `eval_extra_overrides` (opaque list, passed through) and the held-out
  pass as a whole becomes `heldout_hook`.
- purge (CHECKPOINT_PURGE_PATTERNS / docker-exec fallback, :113-149,
  :545-568): `purge_fn(run_dir) -> (deleted, bytes_freed)` hook; None
  disables the step.
- arm string branches (:688 control-observe, :743-746 control decision,
  :755 scripted gate bypass, :765-775 scripted watch retirement):
  LoopConfig flags observe_enabled / propose_enabled / gate_on_pending /
  retire_open_watch_on_set. The arm STRING survives only for segment
  naming and journal text (byte-compat).
- SONIC trainer scalar keys in the digest: `train_scalar_keys` injection.

Journal field names and insertion order are byte-compatible with the old
driver: the base entry comes from core.journal.build_segment_entry and
every later field is added in the exact order the old control flow did
(config_verify -> eval/eval_error -> heldout/heldout_error ->
checkpoint_purge -> tripwire_note -> event/restored -> decision ->
validation -> applied/outcome/digest_hash/applied_at_iter; 'effect' is
written into the ORIGIN entry at scoring time).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

from .digest import build_digest
from .journal import build_event_entry, build_segment_entry
from .tripwire import TripwireWatch, score_effect, tripwire_value


def digest_hash(digest: Dict[str, Any]) -> str:
    """Stable short hash of a digest (journal provenance, doc 08 §3 step 4).
    (smoke_driver.py:94-97)"""
    canon = json.dumps(digest, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


class DiskSpaceError(RuntimeError):
    """Raised (after journaling `disk_gate_failed`) when the filesystem
    holding the run dirs has less than the required headroom BEFORE a
    segment launch. (smoke_driver.py:101-106)"""


# eval-record keys copied into the journal's per-segment eval summary
# (smoke_driver.py:635-639)
DEFAULT_EVAL_SUMMARY_KEYS = ("success_rate", "progress_rate",
                             "mpjpe_all_mean", "mpjpe_l_all_mean",
                             "mpjpe_pa_all_mean")
# held-out record keys copied into the journal (smoke_driver.py:663-666)
DEFAULT_HELDOUT_SUMMARY_KEYS = ("heldout_success_rate",
                                "heldout_failed_count", "heldout_mpjpe_g")

MIN_FREE_BYTES = 8 * 1024 ** 3  # smoke_driver.py:327-329


@dataclasses.dataclass
class LoopConfig:
    """Everything engine-, arm- or campaign-specific, injected.

    The old driver's arm-string branches map to flags:
      control  -> observe_enabled=False, propose_enabled=False
      manager  -> defaults
      scripted -> gate_on_pending=False, retire_open_watch_on_set=True
    """

    arm: str = "manager"                      # naming + summary only
    iterations_per_segment: int = 10
    window: int = 5
    eval_envs: int = 64
    run_eval: bool = True
    seed: int = 42                            # recorded only; adapter owns it
    initial_checkpoint: Optional[str] = None
    base_knobs: Optional[Dict[str, Any]] = None

    # arm behavior flags (old arm-string branches)
    observe_enabled: bool = True              # smoke_driver.py:688
    propose_enabled: bool = True              # :743-746 (control)
    none_reason: str = "control arm"          # decision text when disabled
    gate_on_pending: bool = True              # :755 (scripted bypass)
    retire_open_watch_on_set: bool = False    # :765-775 (scripted)

    # engine seams
    eval_extra_overrides: Optional[List[str]] = None   # standard-eval pin
    eval_summary_keys: Sequence[str] = DEFAULT_EVAL_SUMMARY_KEYS
    heldout_hook: Optional[Callable[..., Dict[str, Any]]] = None
    heldout_summary_keys: Sequence[str] = DEFAULT_HELDOUT_SUMMARY_KEYS
    purge_fn: Optional[Callable[[str], tuple]] = None
    knob_config_paths: Optional[Dict[str, str]] = None
    train_scalar_keys: Optional[Sequence[str]] = None
    cap_ratio_knob: str = "adp_samp_failure_rate_max_over_mean"
    name_fn: Optional[Callable[[int], str]] = None     # tick -> segment name

    # E0 disk gate: explicit path ONLY (no BASE_DIR auto-arming in core)
    disk_gate_path: Optional[str] = None
    min_free_bytes: int = MIN_FREE_BYTES

    def __post_init__(self) -> None:
        # structural guard (v4 post-mortem 2026-07-07, generalized): held-out
        # wiring REQUIRES an explicit standard-eval pin — without one the
        # standard eval pass inherits the checkpoint config's motion set
        # (the curriculum pin -> a 116,924-motion eval in the SONIC case).
        # Old driver: smoke_driver.py:363-372 constructor refusal.
        if self.heldout_hook is not None and not self.eval_extra_overrides:
            raise ValueError(
                "heldout_hook requires eval_extra_overrides (an explicit "
                "eval_motion_file pin): without it the standard eval pass "
                "inherits the checkpoint config's motion set "
                "(v4 post-mortem 2026-07-07)")


class RunManager:
    """Composes adapter + digest + policy + registry over segments —
    the engine-agnostic SmokeDriver."""

    def __init__(self, policy: Any, adapter: Any, registry: Any,
                 config: Optional[LoopConfig] = None,
                 state: Optional[Any] = None):
        if adapter is None:
            raise ValueError("RunManager requires an adapter (the core has "
                             "no default engine)")
        if registry is None:
            raise ValueError("RunManager requires a KnobRegistry (the core "
                             "ships no default action space)")
        self.policy = policy
        self.adapter = adapter
        self.registry = registry
        self.config = config or LoopConfig()
        cfg = self.config
        self.arm = cfg.arm
        self.iters = cfg.iterations_per_segment
        self.window = cfg.window
        self.eval_envs = cfg.eval_envs
        # eval requires an adapter that implements eval_segment (mocks
        # without it exercise the train-side fallback path) —
        # smoke_driver.py:384-387
        self.run_eval = cfg.run_eval and hasattr(adapter, "eval_segment")
        if state is None:
            from .registry import RunState
            state = RunState(tick=0)
        self.state = state
        self.knobs: Dict[str, Any] = dict(cfg.base_knobs or {})
        # seed the registry's belief with the run's ACTUAL starting values
        # (current_values only — not apply(), which would start cooldown
        # clocks for changes the manager never made). smoke_driver.py:391-400
        for name, value in (cfg.base_knobs or {}).items():
            if name in self.registry.knobs:
                self.state.current_values[name] = value
        self.journal: List[Dict[str, Any]] = []
        self.all_train: List[dict] = []
        self.all_sampler: List[dict] = []
        self.all_eval: List[dict] = []
        self.initial_checkpoint = cfg.initial_checkpoint
        # the armed tripwire watch (old driver's `self.armed` dict)
        self.watch: Optional[TripwireWatch] = None
        self.disk_gate_path = cfg.disk_gate_path

    # legacy-name convenience: tests/tools that inspected driver.armed
    @property
    def armed(self) -> Optional[TripwireWatch]:
        return self.watch

    # ── digest plumbing (smoke_driver.py:420-435) ─────────────────────
    def _knob_state(self) -> Dict[str, Any]:
        return {
            name: {"value": self.registry.current_of(name, self.state),
                   "ticks_since_change": (self.state.tick - self.state.last_changed_tick[name]
                                          if name in self.state.last_changed_tick else None)}
            for name in self.knobs or {}
        }

    def _build_digest(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.config.train_scalar_keys is not None:
            kwargs["train_scalar_keys"] = tuple(self.config.train_scalar_keys)
        return build_digest(
            train_records=self.all_train, sampler_records=self.all_sampler,
            eval_records=self.all_eval,
            knob_state=self._knob_state(), decision_history=self.journal[-5:],
            max_over_mean=float(self.knobs.get(self.config.cap_ratio_knob, 50.0)),
            window=self.window, **kwargs)

    # ── config verification (smoke_driver.py:492-519) ─────────────────
    def _verify_config(self, seg: Any, entry: Dict[str, Any]) -> None:
        """Reconcile believed knob values against the segment's resolved
        config (doc 08 §11 amendment 8); refuses (ConfigDriftError) on any
        exact-match divergence. Adapters without the seam and segments
        without a resolved config are journaled, not fatal."""
        getter = getattr(self.adapter, "resolved_config_text", None)
        if getter is None:
            return
        text = getter(seg)
        if not text:
            entry["config_verify"] = {"status": "no_config_yaml"}
            return
        cfg = yaml.safe_load(text)
        if not isinstance(cfg, dict):
            entry["config_verify"] = {"status": "unparseable_config_yaml"}
            return
        paths = self.config.knob_config_paths
        if paths is None:
            fn = getattr(self.adapter, "knob_to_config_path", None)
            paths = fn() if callable(fn) else {}
        res = self.registry.verify_against_config(self.state, cfg, paths)
        entry["config_verify"] = {
            "status": "ok" if res.ok else "drift",
            "checked": res.checked, "missing": res.missing,
            "adopted": res.adopted, "drifts": res.drifts,
        }
        res.raise_on_drift()

    # ── E0 disk gate (smoke_driver.py:522-543) ────────────────────────
    def _disk_gate(self, segment_name: str) -> None:
        if not self.disk_gate_path:
            return
        usage = shutil.disk_usage(self.disk_gate_path)
        if usage.free < self.config.min_free_bytes:
            self.journal.append(build_event_entry(
                self.state.tick, segment_name, "disk_gate_failed",
                path=self.disk_gate_path,
                free_bytes=usage.free,
                required_bytes=self.config.min_free_bytes))
            raise DiskSpaceError(
                f"disk gate: {usage.free / 1024**3:.1f} GB free on "
                f"{self.disk_gate_path} < required "
                f"{self.config.min_free_bytes / 1024**3:.0f} GB — refusing to "
                f"launch segment {segment_name}")

    # ── checkpoint purge hook (smoke_driver.py:545-568) ───────────────
    def _purge_segment(self, seg: Any, entry: Dict[str, Any]) -> None:
        """After a segment is verified done, invoke the injected purge
        hook on its run dir. A missing hook or missing dir is a silent
        no-op; a FAILED purge is journaled, never fatal (hygiene, not
        correctness)."""
        if self.config.purge_fn is None:
            return
        run_dir = getattr(seg, "experiment_dir", None)
        if not run_dir or not os.path.isdir(run_dir):
            return
        try:
            deleted, freed = self.config.purge_fn(run_dir)
        except Exception as e:  # noqa: BLE001 — a failed delete must NEVER
            # kill a multi-hour campaign; the next disk gate is the backstop
            entry["checkpoint_purge"] = {
                "run_dir": run_dir, "error": f"{type(e).__name__}: {e}"[:300]}
            return
        entry["checkpoint_purge"] = {
            "run_dir": run_dir, "deleted": deleted, "bytes_freed": freed}

    # ── the loop (smoke_driver.py:570-808) ────────────────────────────
    def run(self, n_segments: int) -> Dict[str, Any]:
        cfg = self.config
        checkpoint = self.initial_checkpoint
        for i in range(n_segments):
            self.state.tick += 1
            name = (cfg.name_fn(self.state.tick) if cfg.name_fn
                    else f"{self.arm}_s{i+1}")
            # [1] E0: never launch into a nearly-full volume
            self._disk_gate(name)
            # [2-4]
            seg = self.adapter.launch_segment(name, self.iters, self.knobs,
                                              checkpoint_in=checkpoint)
            self.adapter.wait(seg, poll_s=10, timeout_s=3600)
            parsed = self.adapter.parse_segment(seg)
            # [5]
            if seg.status != "done":
                self.journal.append(build_event_entry(
                    self.state.tick, name, "segment_failed",
                    tracebacks=parsed.tracebacks))
                break
            # [6] offset iteration numbers so records accumulate
            # monotonically (per-segment consoles restart numbering at 1)
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

            # [7]
            rew = parsed.train[-1].get("Episode/rew_mean") if parsed.train else None
            entry: Dict[str, Any] = build_segment_entry(
                tick=self.state.tick, segment=name, knobs_in=self.knobs,
                rew_mean_last=rew,
                len_mean_last=(parsed.train[-1].get("Episode/len_mean")
                               if parsed.train else None))

            # [8] believed-vs-resolved config verification — raises
            # ConfigDriftError BEFORE any decision is made on a wrong belief
            self._verify_config(seg, entry)

            # [9] per-segment eval pass at FIXED thresholds (all arms)
            if self.run_eval:
                try:
                    if cfg.eval_extra_overrides:
                        ev = self.adapter.eval_segment(
                            seg, it=cum_it, num_envs=self.eval_envs,
                            extra_overrides=list(cfg.eval_extra_overrides))
                    else:
                        # kwarg passed only when pinned so adapters without
                        # extra_overrides in their signature keep working
                        ev = self.adapter.eval_segment(seg, it=cum_it,
                                                       num_envs=self.eval_envs)
                    self.all_eval.append(ev)
                    entry["eval"] = {k: ev[k] for k in cfg.eval_summary_keys
                                     if k in ev}
                    if "per_motion" in ev:
                        entry["eval"]["per_motion_progress"] = {
                            k: v.get("progress") for k, v in ev["per_motion"].items()}
                except (RuntimeError, TimeoutError) as e:
                    # an eval failure must not kill the run; the tripwire
                    # watch skips this segment (neither breach nor clean)
                    entry["eval_error"] = str(e)[:300]

            # [10] held-out protected-metric pass (hook): the hook returns
            # one record (heldout_* keys + "it"); merged into the eval
            # stream so tripwires can name eval/heldout_success_rate.
            if self.run_eval and cfg.heldout_hook is not None:
                try:
                    hrec = cfg.heldout_hook(self.adapter, seg, cum_it,
                                            self.eval_envs)
                    entry["heldout"] = {k: hrec[k]
                                        for k in cfg.heldout_summary_keys
                                        if k in hrec}
                    if self.all_eval and self.all_eval[-1].get("it") == cum_it:
                        self.all_eval[-1].update(
                            {k: v for k, v in hrec.items() if k != "it"})
                    else:
                        self.all_eval.append(hrec)
                except (RuntimeError, TimeoutError, ValueError) as e:
                    # ValueError = integrity guard. Never emit a wrong
                    # protected metric; the watch sees no value this segment.
                    entry["heldout_error"] = str(e)[:300]

            # [11] purge intermediates once the segment is verified done
            self._purge_segment(seg, entry)

            # [12]
            digest = self._build_digest()
            # [13] the policy observes EVERY segment (when enabled) so
            # gated ticks leave no holes in its sustain history
            if cfg.observe_enabled and hasattr(self.policy, "observe"):
                self.policy.observe(digest)

            # [14] tripwire watch on the armed decision's stated metric
            if self.watch is not None:
                value = tripwire_value(self.watch.metric, rew, self.all_eval,
                                       this_segment_it=cum_it)
                verdict = self.watch.assess(value)
                if verdict.status == "hold":
                    entry["tripwire_note"] = verdict.note
                elif verdict.status == "rollback":
                    knob = self.watch.knob
                    self.knobs[knob] = self.watch.prev_value
                    self.state.apply(knob, self.watch.prev_value)
                    checkpoint = self.watch.prev_ckpt
                    entry["event"] = "rollback"
                    entry["restored"] = verdict.restored
                    # mark the originating decision failed (doc 08 §3 step 5)
                    for prev in reversed(self.journal):
                        if prev.get("applied") and prev["decision"]["knob"] == knob:
                            prev["outcome"] = "failed_rolled_back"
                            break
                    self.watch = None
                    self.state.clear_pending()
                    self.journal.append(entry)
                    continue
                elif verdict.status == "survived":
                    # survived its watch window: score it and disarm
                    for prev in reversed(self.journal):
                        if prev.get("applied") and prev["decision"]["knob"] == self.watch.knob:
                            prev["outcome"] = score_effect(
                                prev, cum_it, self.all_train, self.all_eval)
                            break
                    self.watch = None
                    self.state.clear_pending()

            # [15] propose disabled (control arm)
            if not cfg.propose_enabled:
                entry["decision"] = {"action": "none",
                                     "reason": cfg.none_reason}
                self.journal.append(entry)
                continue

            # [16] pending-decision gate: while a change is armed/unscored,
            # no new change. Flag-off (scripted): the policy replays its
            # ladder unconditionally; the open watch is retired below.
            if self.watch is not None and cfg.gate_on_pending:
                entry["decision"] = {
                    "action": "none",
                    "reason": f"pending decision on {self.watch.knob} still "
                              "under tripwire watch (one change at a time)"}
                self.journal.append(entry)
                continue

            # [17]
            decision = self.policy.propose(digest, self.state, self.registry)
            entry["decision"] = decision
            # [18] flag-gated: a still-open watch is retired unscored so an
            # open-loop ladder stays exactly on schedule (its origin keeps
            # outcome 'pending' — honest: never scored)
            if (decision.get("action") == "set"
                    and cfg.retire_open_watch_on_set
                    and self.watch is not None):
                entry["tripwire_note"] = (
                    f"{self.arm} arm: watch on {self.watch.knob} retired "
                    "unscored (open-loop ladder takes precedence)")
                self.watch = None
                self.state.clear_pending()
            # [19]
            if decision.get("action") == "set":
                res = self.registry.validate_decision(decision, self.state)
                entry["validation"] = {"ok": res.ok, "errors": res.errors}
                # baseline must come from THIS segment's eval — a stale
                # record would arm the watch against pre-change state
                baseline = tripwire_value(
                    decision.get("tripwire", {}).get("metric", "Episode/rew_mean"),
                    rew, self.all_eval, this_segment_it=cum_it)
                if res.ok and baseline is None:
                    # no baseline → the tripwire would be unarmed in
                    # practice; refuse rather than apply unguarded
                    entry["validation"] = {"ok": False, "errors":
                        [f"no {decision['tripwire']['metric']} baseline available "
                         "to arm the tripwire"]}
                elif res.ok:
                    prev = self.registry.current_of(decision["knob"], self.state)
                    self.watch = TripwireWatch(
                        knob=decision["knob"], prev_value=prev,
                        prev_ckpt=checkpoint, baseline=baseline,
                        tw=decision["tripwire"])
                    self.knobs[decision["knob"]] = decision["value"]
                    self.state.apply(decision["knob"], decision["value"])
                    # registry-level pending gate (defense in depth)
                    self.state.arm_pending(decision["knob"])
                    entry["applied"] = True
                    entry["outcome"] = "pending"
                    entry["digest_hash"] = digest_hash(digest)
                    entry["applied_at_iter"] = cum_it
            # [20]
            self.journal.append(entry)
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        """(smoke_driver.py:810-826)"""
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


# convenience factories for the old arm names (string stays OUT of the
# loop's control flow — it only selects flag bundles)
def control_config(**kw: Any) -> LoopConfig:
    kw.setdefault("arm", "control")
    return LoopConfig(observe_enabled=False, propose_enabled=False, **kw)


def scripted_config(**kw: Any) -> LoopConfig:
    kw.setdefault("arm", "scripted")
    return LoopConfig(gate_on_pending=False, retire_open_watch_on_set=True,
                      **kw)
