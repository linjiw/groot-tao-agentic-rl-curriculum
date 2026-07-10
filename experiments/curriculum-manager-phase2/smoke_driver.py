# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Manager ON-vs-OFF driver v3: per-segment eval passes + eval-side tripwire
(design doc 08 §8 Phase 2; v2 review residuals 4–6 partially retired).

Loop per segment (manager arm):
  launch_segment → wait → parse_segment → eval_segment (im_eval at FIXED
  relaxed thresholds) → build_digest → policy.observe → [pending gate] →
  policy.propose → registry.validate → apply (next segment's knobs) or
  rollback → journal

Control arm: identical segments incl. the eval pass (its eval curve is the
comparison anchor), no knob changes ever.

Phase B5 (2026-07-08): SmokeDriver is now a THIN SHELL over the
engine-agnostic run-manager core (experiments/run-manager-core). The
20-step per-segment state machine that used to live in SmokeDriver.run()
(tripwire judgment, pending gate, rollback branch, effect scoring, journal
assembly) is executed by core.loop.RunManager — byte-equivalence of
journal + summary verified against the pre-B5 driver on the four B3/B4
scenarios (normal apply / rollback / control inertness / eval-failure
watch hold). Everything SONIC-specific stays HERE and is injected through
the B3 seams:

- default JobAdapter construction + training-launch curriculum pin
  (adapter argument / constructor),
- knob_registry.load_registry() action space + RunState (registry/state
  arguments),
- job_adapter.KNOB_TO_CONFIG_PATH (LoopConfig.knob_config_paths),
- Hydra motion_file override strings (LoopConfig.eval_extra_overrides and
  the held-out pass as LoopConfig.heldout_hook),
- checkpoint purge incl. the docker-exec root-owned-file fallback
  (LoopConfig.purge_fn = purge_intermediate_checkpoints below),
- BASE_DIR disk-gate auto-arming (resolved here, passed as the explicit
  LoopConfig.disk_gate_path),
- SONIC trainer scalar keys (LoopConfig.train_scalar_keys =
  digest_builder.TRAIN_SCALAR_KEYS),
- arm strings mapped to LoopConfig flags (observe_enabled /
  propose_enabled / gate_on_pending / retire_open_watch_on_set); the arm
  STRING survives only for segment naming and journal text.

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
  scoring: when a decision carries a machine-readable
  `expected_effect_check` ({metric, op, value}), a change that survives its
  tripwire watch is scored `survived_effect_confirmed` /
  `survived_effect_not_observed` against metrics the driver already records
  (train stream or this segment's eval record); without a check the outcome
  stays plain `survived`. Rolled-back changes stay `failed_rolled_back`.
- **Registry-level pending gate** (doc 08 §11 amendment 2, defense in
  depth): on apply the driver arms `state.pending`, so
  `validate_decision` itself rejects any further `set` until the change is
  scored or rolled back — in addition to the loop's own gate.

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
import importlib.util
import json
import os
import shutil  # noqa: F401 — kept as the disk-gate monkeypatch seam
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
# run-manager core (Phase B3-B5): relative sys.path insertion — no package
# install, no .pth files
_CORE = os.path.join(_REPO, "experiments", "run-manager-core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from core.loop import (DiskSpaceError, LoopConfig,  # noqa: E402
                       RunManager, digest_hash)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


knob_registry = _load("knob_registry", "skills/agentic/sonic-knob-registry/knob_registry.py")
digest_builder = _load("digest_builder", "skills/agentic/sonic-run-digest/digest_builder.py")
job_adapter = _load("job_adapter", "skills/agentic/sonic-job-adapter/job_adapter.py")
holdout = _load("holdout", "skills/agentic/sonic-heldout-watcher/holdout.py")


# ── E0: disk hygiene (SONIC-side purge adapter) ──────────────────────
# files safe to delete once the segment's rollback snapshot exists:
# last.pt is superseded by snapshot_<segment>.pt (every later consumer —
# resume, eval, rollback — reads seg.snapshot, never last.pt), and
# model_step_*.pt intermediates are never read by the driver at all.
CHECKPOINT_PURGE_PATTERNS = ("last.pt", "model_step_*.pt")
# container that owns root-owned checkpoint files (fallback deleter):
PURGE_CONTAINER = "isaac-lab-base"


def purge_intermediate_checkpoints(run_dir: str) -> tuple:
    """Delete intermediate checkpoints from a VERIFIED segment's run dir.

    Deletes plain files matching CHECKPOINT_PURGE_PATTERNS directly inside
    `run_dir`; keeps snapshot_*.pt (the rollback/resume points) and every
    subdirectory (eval output dirs). Returns (deleted_names, bytes_freed).
    """
    import glob as _glob
    deleted: List[str] = []
    freed = 0
    for pattern in CHECKPOINT_PURGE_PATTERNS:
        for path in sorted(_glob.glob(os.path.join(run_dir, pattern))):
            if not os.path.isfile(path):
                continue  # never touch directories (eval outputs)
            size = os.path.getsize(path)
            try:
                os.remove(path)
            except PermissionError:
                # Training runs inside the container as root, so checkpoints
                # are root-owned (rw------- root) in a root-owned dir; the
                # host-side driver (ec2-user) cannot unlink them. Fall back
                # to deleting through the container (same mount path).
                import subprocess as _sp
                _sp.run(["docker", "exec", PURGE_CONTAINER,
                         "rm", "-f", path], check=True, timeout=60)
                if os.path.exists(path):
                    raise RuntimeError(
                        f"docker-exec rm reported success but {path} "
                        f"still exists")
            freed += size
            deleted.append(os.path.basename(path))
    return deleted, freed


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
                # machine-checkable form of expected_effect (doc 08 §11
                # amendment 4 payoff): scored when the watch clears
                "expected_effect_check": {"metric": "Episode/len_mean",
                                          "op": ">=", "value": self.len_low},
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
                "expected_effect_check": {"metric": "Episode/len_mean",
                                          "op": "<=", "value": self.len_high},
                "tripwire": tripwire,
            }
        return {"action": "none",
                "reason": f"len {[round(v,1) for v in recent]} inside "
                          f"[{self.len_low}, {self.len_high}] band or sustain unmet"}


# ── E1: scripted-decision ablation arm ───────────────────────────────
# The EXACT decision stream both v4 manager seeds walked (verified against
# knobs_in in manager_journal_v4_seed42.json / _seed1337.json): tick -> (knob,
# value). E1 asks the cheapest project-killing question — does an OPEN-LOOP
# replay of the same knob ladder reproduce the manager runs?
V4_MANAGER_LADDER = {
    2: ("termination_threshold.foot_pos_xyz", 0.25),
    4: ("termination_threshold.ee_body_pos", 0.20),
    6: ("termination_threshold.foot_pos_xyz", 0.30),
    8: ("termination_threshold.ee_body_pos", 0.25),
    10: ("termination_threshold.foot_pos_xyz", 0.35),
}


@dataclasses.dataclass
class ScriptedPolicy:
    """Open-loop ladder replay (E1 ablation arm).

    Replays a fixed tick-indexed knob ladder UNCONDITIONALLY: the choice
    reads nothing but ``state.tick`` — no digest, no eval state, no sustain
    history, no watch-window gating. The digest argument is used ONLY to
    pick the tripwire metric (eval-side when an eval stream exists), i.e.
    the eval-side tripwire SAFETY stays armed exactly as in the manager arm,
    but the decisions themselves are fixed.

    Same propose() interface as TrainSideBandPolicy; deliberately NO
    observe() — this policy keeps no run-state whatsoever.
    """

    ladder: Dict[int, tuple] = dataclasses.field(
        default_factory=lambda: dict(V4_MANAGER_LADDER))

    def propose(self, digest: Optional[Dict[str, Any]], state, registry) -> Dict[str, Any]:
        rung = self.ladder.get(state.tick)
        if rung is None:
            return {"action": "none",
                    "reason": f"scripted ladder: no rung scheduled at tick {state.tick}"}
        knob, value = rung
        # tripwire safety mirrors TrainSideBandPolicy._tripwire: eval-side
        # fixed-threshold guard when an eval stream exists. This is the ONLY
        # digest read and it does not influence WHICH decision is made.
        if (digest or {}).get("eval"):
            tripwire = {"metric": "eval/progress_rate", "drop_pct": 30, "evals": 2}
        else:
            tripwire = {"metric": "Episode/rew_mean", "drop_pct": 20, "evals": 2}
        return {
            "action": "set", "knob": knob, "value": value,
            "rationale": (f"scripted open-loop replay (E1 ablation): fixed v4 "
                          f"manager ladder rung at tick {state.tick} — no "
                          "digest/eval input to the choice"),
            "expected_effect": ("reproduces the v4 manager arm's knob "
                                "trajectory without closed-loop decisions"),
            "tripwire": tripwire,
        }


class SmokeDriver(RunManager):
    """Composes adapter + digest + policy + registry over real segments.

    Phase B5 thin shell: the per-segment state machine (run(), summary(),
    tripwire judgment, pending gate, rollback, effect scoring, journal
    assembly) is entirely core.loop.RunManager — this class only assembles
    the SONIC side (JobAdapter, knob registry, motion_file pins, held-out
    hook, checkpoint purge, disk-gate path, trainer scalar keys) into a
    LoopConfig and delegates.
    """

    # below this baseline, relative progress_rate drops are noise-level
    # (baseline curve: progress_rate ~0.003 at 2k iters); an armed tripwire
    # additionally requires an absolute drop of this size to breach
    # (enforced by core.tripwire.TripwireWatch, same constant)
    EVAL_ABS_MIN_DROP = 0.002

    # E0 disk gate: refuse to launch a segment with less than this free on
    # the filesystem holding the run dirs (per-segment checkpoints ~3.6 GB)
    MIN_FREE_BYTES = 8 * 1024 ** 3

    def __init__(self, policy, adapter=None, arm: str = "manager",
                 iterations_per_segment: int = 10, window: int = 5,
                 base_knobs: Optional[Dict[str, Any]] = None,
                 num_envs: int = 64, eval_envs: int = 64,
                 run_eval: bool = True, seed: int = 42,
                 initial_checkpoint: Optional[str] = None,
                 project: Optional[str] = None,
                 heldout_manifest: Optional[str] = None,
                 heldout_eval_motion_file: Optional[str] = None,
                 curriculum_motion_file: Optional[str] = None,
                 eval_motion_file: Optional[str] = None,
                 disk_gate_path: Optional[str] = None,
                 train_env: Optional[Dict[str, Any]] = None,
                 train_pythonpath: Optional[str] = None,
                 tier0_train_override: Optional[str] = None):
        # held-out protected-metric wiring (doc 08 §5): when a manifest is
        # given, (a) every TRAINING launch is pinned to the curriculum-only
        # motion directory so the held-out split can never enter the training
        # set (hard rule 4), and (b) each segment runs a second eval-only
        # pass on the held-out subset whose record is merged into the eval
        # stream as heldout_* keys — reachable by tripwires as
        # eval/heldout_success_rate but invisible to the manager's knobs.
        self.heldout_manifest = (holdout.load_manifest(heldout_manifest)
                                 if heldout_manifest else None)
        self.heldout_eval_motion_file = heldout_eval_motion_file
        # v4 post-mortem (2026-07-07): the standard eval pass previously
        # passed NO motion_file override, so eval_agent_trl.py inherited the
        # checkpoint-sibling config's motion_file — with the curriculum pin
        # that meant a 116,924-motion eval (28+ h projected), driver timeout,
        # and an orphaned GPU eval starving every later launch. motion_file
        # is checkpoint-config state that eval.yaml does not re-pin (same
        # leak class as review-M1 foot_pos_xyz), so pin it EXPLICITLY.
        self.eval_motion_file = eval_motion_file
        extra = None
        if self.heldout_manifest:
            if not (heldout_eval_motion_file and curriculum_motion_file):
                raise ValueError("heldout_manifest requires both "
                                 "heldout_eval_motion_file and curriculum_motion_file")
            if not eval_motion_file:
                raise ValueError(
                    "heldout_manifest requires eval_motion_file: without an "
                    "explicit pin the standard eval pass inherits the "
                    "checkpoint config's motion_file (the curriculum pin -> "
                    "116,924-motion eval; v4 post-mortem 2026-07-07)")
            extra = ["++manager_env.commands.motion.motion_lib_cfg."
                     f"motion_file={curriculum_motion_file}"]
        # tier-0 reward-func swap (doc 10 G0/G2): a TRAINING-only Hydra
        # override, appended to `extra` (the eval pass never gets it — the
        # scoreboard stays stock). Orthogonal to the manager action space.
        if tier0_train_override:
            extra = (extra or []) + [tier0_train_override]
        # default project keeps v2's smoke_{arm}; comparison runs pass their
        # own prefix so v2 artifacts aren't overwritten
        injected_adapter = adapter
        adapter = adapter or job_adapter.JobAdapter(
            project=project or f"smoke_{arm}", num_envs=num_envs,
            save_last_frequency=5, seed=seed, extra_overrides=extra,
            train_env=train_env, train_pythonpath=train_pythonpath)

        # E0 disk gate: the path whose filesystem must hold >= MIN_FREE_BYTES
        # before every segment launch. Explicit path wins; otherwise the
        # gate self-arms only for the REAL adapter (adapter=None above ->
        # JobAdapter writing under BASE_DIR on the training volume). Injected
        # adapters (unit-test fakes) get no gate unless a path is passed —
        # tests must not depend on this host's disk state. The core arms
        # ONLY from an explicit path, so the auto-arming stays HERE.
        if disk_gate_path is not None:
            gate = disk_gate_path
        elif injected_adapter is None and os.path.isdir(job_adapter.BASE_DIR):
            gate = job_adapter.BASE_DIR
        else:
            gate = None

        # standard-eval motion_file pin (opaque Hydra string: SONIC-side)
        eval_extra = None
        if self.eval_motion_file:
            eval_extra = ["++manager_env.commands.motion.motion_lib_cfg."
                          f"motion_file={self.eval_motion_file}"]

        # held-out protected-metric pass as a core hook (doc 08 §5): a
        # SECOND eval-only run on the held-out subset; the core merges its
        # record into the eval stream (heldout_* keys) so a tripwire can
        # name eval/heldout_success_rate, but it never feeds the policy's
        # band signal and no knob can move its thresholds or motion set.
        heldout_hook = None
        if self.heldout_manifest is not None:
            def heldout_hook(adp, seg, cum_it, eval_envs_):
                raw = adp.eval_segment(
                    seg, it=cum_it, num_envs=eval_envs_,
                    out_suffix="_heldout_eval", raw=True,
                    extra_overrides=[
                        "++manager_env.commands.motion.motion_lib_cfg."
                        f"motion_file={self.heldout_eval_motion_file}"])
                return holdout.heldout_record_from_metrics_eval(
                    raw, self.heldout_manifest, it=cum_it)

        # arm string -> core behavior flags (the string itself survives only
        # for segment naming and journal text, exactly as before):
        #   control  -> no observe, no propose ("control arm" decisions)
        #   scripted -> no pending gate; open watch retired on a new rung
        #   manager  -> defaults
        cfg = LoopConfig(
            arm=arm,
            iterations_per_segment=iterations_per_segment,
            window=window,
            eval_envs=eval_envs,
            run_eval=run_eval,
            seed=seed,
            initial_checkpoint=initial_checkpoint,
            base_knobs=base_knobs,
            observe_enabled=(arm != "control"),
            propose_enabled=(arm != "control"),
            gate_on_pending=(arm != "scripted"),
            retire_open_watch_on_set=(arm == "scripted"),
            eval_extra_overrides=eval_extra,
            heldout_hook=heldout_hook,
            purge_fn=purge_intermediate_checkpoints,
            knob_config_paths=job_adapter.KNOB_TO_CONFIG_PATH,
            train_scalar_keys=tuple(digest_builder.TRAIN_SCALAR_KEYS),
            disk_gate_path=gate,
            min_free_bytes=self.MIN_FREE_BYTES,
        )
        super().__init__(policy, adapter, knob_registry.load_registry(),
                         cfg, state=knob_registry.RunState(tick=0))

    # run() / summary() / armed / journal / knobs / state — all inherited
    # from core.loop.RunManager (byte-compatible journal + summary).


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="manager ON-vs-OFF driver (v3: per-segment eval)")
    p.add_argument("--arm", choices=["manager", "control", "scripted"], required=True,
                   help="'scripted' = E1 ablation: open-loop replay of the "
                        "fixed v4 manager knob ladder (no digest reads, no "
                        "watch-window gating; eval-side tripwire stays armed)")
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
    p.add_argument("--heldout-manifest",
                   help="path to holdout manifest JSON; enables the per-"
                        "segment held-out protected-metric pass and pins "
                        "training to --curriculum-motion-file")
    p.add_argument("--heldout-motion-file",
                   help="container path of the held-out motion dir "
                        "(e.g. data/motion_lib_bones_seed/robot_heldout)")
    p.add_argument("--curriculum-motion-file",
                   help="container path of the curriculum-only motion dir "
                        "(e.g. data/motion_lib_bones_seed/robot_curriculum)")
    p.add_argument("--eval-motion-file",
                   help="container path of the FIXED standard-eval motion dir "
                        "(e.g. data/motion_lib_bones_seed/robot_curriculum_eval64). "
                        "REQUIRED with --heldout-manifest: pins the standard "
                        "per-segment eval so it can never inherit the training "
                        "config's motion set (v4 post-mortem 2026-07-07)")
    # tier-0 sigma-EMA shim (doc 10 G0/G2): injected via env + PYTHONPATH +
    # a reward-func Hydra override, applied to TRAINING launches only (eval
    # stays stock — the scoreboard must not see the shim). See
    # experiments/run-manager-core/adapters/sonic_tier0/README.md.
    p.add_argument("--tier0", choices=["off", "noop", "active"], default="off",
                   help="off=no shim (stock); noop=shim inserted but inert "
                        "(G0 bit-identity arm); active=PBHC sigma-EMA (G2)")
    p.add_argument("--tier0-pythonpath", default="/workspace/rmc_tier0",
                   help="container PYTHONPATH for the tier-0 shim package")
    p.add_argument("--tier0-term", default="tracking_anchor_pos",
                   help="reward term to wrap (maps to func + class name)")
    p.add_argument("--tier0-ema-rate", type=float, default=0.01)
    p.add_argument("--tier0-sigma-floor", type=float, default=0.1,
                   help="sigma floor as a FRACTION of the term's std")
    p.add_argument("--tier0-sidecar-dir",
                   help="dir for sigma-state persistence across segment resumes")
    p.add_argument("--tier0-trace",
                   help="append-JSONL sigma-trace path (journal)")
    args = p.parse_args(argv)

    base_knobs = json.loads(args.base_knobs) if args.base_knobs else None
    if args.arm == "scripted":
        policy = ScriptedPolicy()
    else:
        policy = TrainSideBandPolicy(len_low=args.len_low, sustain=args.sustain)

    # tier-0 sigma-EMA shim assembly (doc 10 G0/G2). off => nothing; noop/
    # active => env vars + PYTHONPATH + a reward-func Hydra override appended
    # to TRAINING launches. The func override rides the same extra_overrides
    # path as the curriculum motion pin (SmokeDriver builds `extra`), but the
    # shim's func/env is orthogonal to the manager action space.
    tier0_env = None
    tier0_pp = None
    tier0_override = None
    if args.tier0 != "off":
        term_to_class = {
            "tracking_anchor_pos": "SigmaEMAAnchorPos",
            "tracking_anchor_ori": "SigmaEMAAnchorOri",
            "tracking_relative_body_pos": "SigmaEMARelBodyPos",
        }
        cls = term_to_class.get(args.tier0_term)
        if cls is None:
            raise SystemExit(f"--tier0-term {args.tier0_term} has no shim class "
                             f"(known: {sorted(term_to_class)})")
        tier0_pp = args.tier0_pythonpath
        tier0_env = {
            "SONIC_TIER0_ACTIVE": "1" if args.tier0 == "active" else "0",
            "SONIC_TIER0_EMA_RATE": repr(args.tier0_ema_rate),
            "SONIC_TIER0_SIGMA_FLOOR": repr(args.tier0_sigma_floor),
        }
        if args.tier0_sidecar_dir:
            tier0_env["SONIC_TIER0_SIDECAR_DIR"] = args.tier0_sidecar_dir
        if args.tier0_trace:
            tier0_env["SONIC_TIER0_TRACE"] = args.tier0_trace
            tier0_env["SONIC_TIER0_LOG_EVERY"] = "10"
        tier0_override = (f"++manager_env.rewards.{args.tier0_term}.func="
                          f"adapters.sonic_tier0.sonic_sigma_ema_term:{cls}")

    driver = SmokeDriver(policy, arm=args.arm, iterations_per_segment=args.iters,
                         num_envs=args.num_envs, eval_envs=args.eval_envs,
                         run_eval=not args.no_eval, seed=args.seed,
                         initial_checkpoint=args.initial_checkpoint,
                         project=args.project, base_knobs=base_knobs,
                         heldout_manifest=args.heldout_manifest,
                         heldout_eval_motion_file=args.heldout_motion_file,
                         curriculum_motion_file=args.curriculum_motion_file,
                         eval_motion_file=args.eval_motion_file,
                         train_env=tier0_env, train_pythonpath=tier0_pp,
                         tier0_train_override=tier0_override)
    summary = driver.run(args.segments)
    print(json.dumps(summary, indent=2))
    if args.journal_out:
        with open(args.journal_out, "w") as f:
            json.dump(driver.journal, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
