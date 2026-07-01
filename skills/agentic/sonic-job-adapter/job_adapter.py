# SPDX-License-Identifier: Apache-2.0
"""SONIC job adapter: launch / observe / rollback lifecycle for the
Curriculum-Manager Agent (design doc 08, Phase 2).

Three responsibilities, all grounded in the verified infra
(docs/infra-guide.md, docs/progress/2026-07-01-infra.md):

1. **Launch** — build and run the docker-exec training invocation with
   manager knob overrides mapped from registry names to verified Hydra
   paths (KNOB_TO_HYDRA). Knob mutation model: per run-segment — each
   segment starts from the previous segment's `last.pt` with new overrides.

2. **Parse** — turn the training console log (the boxed rich output) into
   the digest builder's train/sampler JSONL streams. The console block is
   the only live metrics surface when use_wandb=false; format captured in
   testdata/train_log_excerpt.txt from a real run.

3. **Checkpoint/rollback** — locate a run's newest experiment dir and
   `last.pt`, snapshot checkpoints per segment, and build the rollback
   launch (relaunch from the pre-change segment's snapshot with the
   pre-change knob values).

Pure Python stdlib. Only `launch()`/`wait()` touch docker; everything else
(command building, log parsing, checkpoint bookkeeping) is CPU-testable
offline — which is what the tests cover.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional

CONTAINER = "isaac-lab-base"
WBC_DIR = "/workspace/GR00T-WholeBodyControl"
PYTHON_SH = "/isaac-sim/python.sh"
BASE_DIR = "/workspace/wbc-training-logs"
EXP_CONFIG = "+exp/manager/universal_token/all_modes=sonic_bones_seed"

# Registry knob name -> verified Hydra override path (infra-guide.md table).
KNOB_TO_HYDRA = {
    "uniform_sampling_rate":
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.uniform_sampling_rate",
    "adp_samp_failure_rate_max_over_mean":
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.adp_samp_failure_rate_max_over_mean",
    "bin_size":
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.bin_size",
    "termination_threshold.anchor_pos":
        "++manager_env.terminations.anchor_pos.params.threshold",
    "termination_threshold.ee_body_pos":
        "++manager_env.terminations.ee_body_pos.params.threshold",
    "termination_threshold.foot_pos_xyz":
        "++manager_env.terminations.foot_pos_xyz.params.threshold",
    "desired_kl": "++algo.config.desired_kl",
    "entropy_coef": "++algo.config.entropy_coef",
}


# ── command building (pure) ──────────────────────────────────────────
def build_overrides(knobs: Dict[str, Any]) -> List[str]:
    """Map registry knob values to Hydra override strings.

    Raises KeyError on any knob without a verified Hydra path — the adapter
    must never invent config paths.
    """
    out = []
    for name, value in sorted(knobs.items()):
        if name not in KNOB_TO_HYDRA:
            raise KeyError(
                f"knob {name!r} has no verified Hydra mapping (KNOB_TO_HYDRA); "
                "verify the config path live before adding it"
            )
        out.append(f"{KNOB_TO_HYDRA[name]}={value}")
    return out


def build_train_command(
    experiment_name: str,
    num_envs: int = 256,
    iterations: int = 100,
    knobs: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[str] = None,
    project_name: str = "manager",
    steps_per_env: int = 16,
    save_last_frequency: int = 10,
    log_path: Optional[str] = None,
) -> List[str]:
    """The docker-exec launch, exactly as verified in infra-guide.md."""
    parts = [
        f"cd {WBC_DIR} &&",
        "nohup" if log_path else "",
        PYTHON_SH, "gear_sonic/train_agent_trl.py",
        EXP_CONFIG,
        f"num_envs={num_envs}",
        "headless=true",
        "use_wandb=false",
        f"project_name={project_name}",
        f"experiment_name={experiment_name}",
        f"base_dir={BASE_DIR}",
        f"++algo.config.num_learning_iterations={iterations}",
        f"++algo.config.num_steps_per_env={steps_per_env}",
        f"++callbacks.model_save.save_last_frequency={save_last_frequency}",
    ]
    if checkpoint:
        parts.insert(parts.index(f"base_dir={BASE_DIR}") + 1, f"checkpoint={checkpoint}")
    parts.extend(build_overrides(knobs or {}))
    if log_path:
        parts.append(f"> {log_path} 2>&1 &")
    inner = " ".join(p for p in parts if p)
    return ["docker", "exec", CONTAINER, "bash", "-c", inner]


# ── console-log parsing (pure) ───────────────────────────────────────
_ANSI = re.compile(r"\x1b\[[0-9;]*m|\[\d+m")
_ITER = re.compile(r"Learning iteration\s+(\d+)")
_KV = re.compile(r"([A-Za-z_][\w/ ]*?):\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)\s*$")
_CKPT_LOADED = re.compile(r"Loaded checkpoint from step (\d+)")
_LOGDIR = re.compile(r"(/[\w./-]*wbc-training-logs/[\w./-]+)")

# console label -> digest train-stream key (ppo_trainer log_dict / rich box)
_LABEL_MAP = {
    "Mean rewards": "Episode/rew_mean",
    "Mean length": "Episode/len_mean",
    "Mean entropy": "loss/entropy_avg",
    "Mean action noise std": "Policy/mean_noise_std",
    "Iteration time": "iteration_time_s",
}


def _clean(line: str) -> str:
    line = _ANSI.sub("", line)
    return line.replace("│", " ").replace("╭", " ").replace("╰", " ").replace("─", " ").strip()


@dataclasses.dataclass
class ParsedLog:
    train: List[dict]
    sampler: List[dict]
    checkpoint_loaded_step: Optional[int] = None
    experiment_dir: Optional[str] = None
    tracebacks: int = 0


def parse_console_log(
    lines: Iterable[str],
    steps_per_iteration: int = 1,
) -> ParsedLog:
    """Parse the rich console blocks into digest train/sampler records.

    One train record and one sampler record per `Learning iteration N`
    block. `it` is the iteration number scaled by `steps_per_iteration`
    (=1 keeps raw iteration numbers). The sampler stream carries the
    Env/adp_samp/* summary stats — the console prints aggregates, not the
    per-bin vector, so `failure_rate` is NOT populated here; the digest's
    sampler section still gets entropy-free health stats via
    `sampler_stats` records (digest builder treats absent `failure_rate`
    records as no sampler section — pass these to the eval/train side or
    extend the digest; see SKILL.md "Sampler stream caveat").
    """
    train: List[dict] = []
    sampler: List[dict] = []
    ckpt_step: Optional[int] = None
    exp_dir: Optional[str] = None
    tracebacks = 0

    cur_it: Optional[int] = None
    cur_train: Dict[str, Any] = {}
    cur_samp: Dict[str, Any] = {}

    def flush():
        nonlocal cur_train, cur_samp
        if cur_it is not None and cur_train:
            train.append({"it": cur_it, **cur_train})
        if cur_it is not None and cur_samp:
            sampler.append({"it": cur_it, **cur_samp})
        cur_train, cur_samp = {}, {}

    for raw in lines:
        line = _clean(raw)
        if not line:
            continue
        if "Traceback" in line:
            tracebacks += 1
        m = _CKPT_LOADED.search(line)
        if m:
            ckpt_step = int(m.group(1))
        if "Logging Directory" not in line and exp_dir is None:
            m = _LOGDIR.search(line)
            if m and "/demo/" in m.group(1) or (m and re.search(r"-\d{8}_\d{6}", m.group(1))):
                exp_dir = m.group(1)
        m = _ITER.search(line)
        if m:
            flush()
            cur_it = int(m.group(1)) * steps_per_iteration
            continue
        if cur_it is None:
            continue
        m = _KV.search(line)
        if not m:
            continue
        label, value = m.group(1).strip(), float(m.group(2))
        if label in _LABEL_MAP:
            cur_train[_LABEL_MAP[label]] = value
        elif label.startswith("Env/adp_samp/"):
            cur_samp[label.removeprefix("Env/adp_samp/")] = value
        elif label.startswith(("Env/Episode_Reward/", "Env/Episode_Termination/",
                               "Env/Metrics/motion/")):
            cur_train[label.replace("Env/Episode_Reward/", "Episode/")
                      .replace("Env/", "")] = value
    flush()
    return ParsedLog(train=train, sampler=sampler,
                     checkpoint_loaded_step=ckpt_step,
                     experiment_dir=exp_dir, tracebacks=tracebacks)


def write_jsonl(records: List[dict], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── docker plumbing (thin, not unit-tested) ──────────────────────────
def _dexec(cmd: str, timeout: int = 60) -> str:
    proc = subprocess.run(["docker", "exec", CONTAINER, "bash", "-c", cmd],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"docker exec failed ({proc.returncode}): {proc.stderr[:300]}")
    return proc.stdout


def read_container_log(log_path: str) -> List[str]:
    return _dexec(f"cat {shlex.quote(log_path)}", timeout=120).splitlines()


def find_experiment_dir(project: str, experiment: str) -> Optional[str]:
    out = _dexec(
        f"ls -d {BASE_DIR}/{shlex.quote(project)}/{shlex.quote(experiment)}-* 2>/dev/null | sort | tail -1"
    ).strip()
    return out or None


def latest_checkpoint(experiment_dir: str) -> Optional[str]:
    # `|| true`: absence of last.pt (e.g. a segment shorter than the save
    # cadence) is a normal condition, not an error
    out = _dexec(f"ls {shlex.quote(experiment_dir)}/last.pt 2>/dev/null || true").strip()
    return out or None


def snapshot_checkpoint(experiment_dir: str, tag: str) -> str:
    """Copy last.pt aside so a later segment's overwrite can't destroy the
    rollback point. Returns the snapshot path (inside the container)."""
    src = f"{experiment_dir}/last.pt"
    dst = f"{experiment_dir}/snapshot_{tag}.pt"
    _dexec(f"cp {shlex.quote(src)} {shlex.quote(dst)}", timeout=300)
    return dst


def is_training_running() -> bool:
    # "[t]rain" bracket trick: the pattern doesn't match the pgrep/bash
    # wrapper's own command line, only a real python process
    out = _dexec("pgrep -f '[t]rain_agent_trl.py' | xargs -r ps -o comm= -p 2>/dev/null | grep -c python || true").strip()
    return int(out or 0) > 0


# ── the segment lifecycle ────────────────────────────────────────────
@dataclasses.dataclass
class Segment:
    """One knob-constant stretch of training (the manager's unit of change)."""

    name: str
    iterations: int
    knobs: Dict[str, Any]
    checkpoint_in: Optional[str] = None   # None = fresh start
    log_path: str = ""
    experiment_dir: Optional[str] = None
    snapshot: Optional[str] = None        # rollback point (checkpoint BEFORE this segment)
    status: str = "pending"               # pending|running|done|failed


class JobAdapter:
    """Drives run-segments; owns checkpoint bookkeeping for rollback."""

    def __init__(self, project: str = "manager", num_envs: int = 256,
                 steps_per_env: int = 16, save_last_frequency: int = 10):
        self.project = project
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.save_last_frequency = save_last_frequency
        self.segments: List[Segment] = []

    def launch_segment(self, name: str, iterations: int,
                       knobs: Dict[str, Any],
                       checkpoint_in: Optional[str] = None) -> Segment:
        if is_training_running():
            raise RuntimeError("a training process is already running in the container")
        seg = Segment(name=name, iterations=iterations, knobs=dict(knobs),
                      checkpoint_in=checkpoint_in,
                      log_path=f"{BASE_DIR}/{self.project}_{name}.log")
        cmd = build_train_command(
            experiment_name=name, num_envs=self.num_envs,
            iterations=iterations, knobs=knobs, checkpoint=checkpoint_in,
            project_name=self.project, steps_per_env=self.steps_per_env,
            save_last_frequency=self.save_last_frequency,
            log_path=seg.log_path,
        )
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        seg.status = "running"
        self.segments.append(seg)
        return seg

    def wait(self, seg: Segment, poll_s: int = 20, timeout_s: int = 3600) -> Segment:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if not is_training_running():
                break
            time.sleep(poll_s)
        else:
            seg.status = "failed"
            raise TimeoutError(f"segment {seg.name} still running after {timeout_s}s")
        parsed = self.parse_segment(seg)
        seg.experiment_dir = parsed.experiment_dir or find_experiment_dir(self.project, seg.name)
        seg.status = "failed" if parsed.tracebacks else "done"
        if seg.status == "done" and seg.experiment_dir:
            ckpt = latest_checkpoint(seg.experiment_dir)
            if ckpt:
                seg.snapshot = snapshot_checkpoint(seg.experiment_dir, seg.name)
        return seg

    def parse_segment(self, seg: Segment) -> ParsedLog:
        return parse_console_log(read_container_log(seg.log_path))

    def rollback_launch(self, failed: Segment, name: str,
                        iterations: int) -> Segment:
        """Relaunch from the state BEFORE `failed`: its input checkpoint and
        its predecessor's knob values."""
        idx = self.segments.index(failed)
        prev_knobs = self.segments[idx - 1].knobs if idx > 0 else {}
        return self.launch_segment(name=name, iterations=iterations,
                                   knobs=prev_knobs,
                                   checkpoint_in=failed.checkpoint_in)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="SONIC job adapter")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("command", help="print the launch command (dry run)")
    c.add_argument("--name", required=True)
    c.add_argument("--iterations", type=int, default=100)
    c.add_argument("--num-envs", type=int, default=256)
    c.add_argument("--checkpoint")
    c.add_argument("--knob", action="append", default=[],
                   help="name=value, repeatable")

    g = sub.add_parser("parse", help="console log -> digest JSONL streams")
    g.add_argument("--log", required=True, help="host path OR container path with --container")
    g.add_argument("--container", action="store_true")
    g.add_argument("--out-prefix", default="run")

    args = p.parse_args(argv)
    if args.cmd == "command":
        knobs = dict(kv.split("=", 1) for kv in args.knob)
        cmd = build_train_command(args.name, num_envs=args.num_envs,
                                  iterations=args.iterations, knobs=knobs,
                                  checkpoint=args.checkpoint)
        print(" ".join(shlex.quote(c) for c in cmd))
    else:
        lines = (read_container_log(args.log) if args.container
                 else open(args.log).read().splitlines())
        parsed = parse_console_log(lines)
        write_jsonl(parsed.train, f"{args.out_prefix}_train.jsonl")
        write_jsonl(parsed.sampler, f"{args.out_prefix}_sampler.jsonl")
        print(f"train: {len(parsed.train)} records, sampler: {len(parsed.sampler)} records, "
              f"tracebacks: {parsed.tracebacks}, ckpt_step: {parsed.checkpoint_loaded_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
