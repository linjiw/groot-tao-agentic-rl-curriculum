# SPDX-License-Identifier: Apache-2.0
"""Held-out eval watcher core: the producer of `heldout_success_rate`.

Design doc 08, axiom 5 ("protect the manager's metric from the manager") and
§4 (the protected metric). This module is the CPU-testable core; the live
wiring (running eval_agent_trl.py on a schedule) is documented in SKILL.md.

Three responsibilities:

1. `select_holdout(keys, fraction, salt)` — deterministic, hash-based split
   of the motion library into held-out vs curriculum keys. Hash-based (not
   index-based) so membership is stable when motions are added/removed, and
   salted so the composition is not reproducible without the salt. The salt
   lives in the watcher's manifest — a file the manager process never reads.

2. `write_manifest / load_manifest` — the watcher's private record: salt,
   fraction, the held-out key list, and an integrity digest.

3. `heldout_record_from_metrics_eval(metrics_eval, manifest, it)` — turn one
   eval-only pass's saved `metrics_eval.json` (SONIC's own output format,
   im_eval_callback.save_metrics_eval) into the single eval.jsonl record the
   digest builder consumes: {"it", "heldout_success_rate", ...}. Refuses to
   produce a number if the eval demonstrably did not run on the held-out
   subset (failed_keys outside the manifest), so a mis-wired eval can never
   silently feed the manager a wrong protected metric.

Verified SONIC seams (pinned WBC 0e35637):
- subset restriction: `commands.motion.filter_motion_keys` +
  `motion_lib_cfg.filter_motion_keys` (eval_agent_trl.py:316-318, 367-369;
  commands.py:201).
- fixed relaxed thresholds: `terminations/tracking/eval.yaml` (0.25) — NOT
  in the knob registry, by design.
- eval-only output: metrics_eval.json with keys "eval/success/success_rate",
  "failed_keys", "failed_idxes" (im_eval_callback.py:135-155, 227-231).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Optional

MANIFEST_VERSION = "0.1.0"


def _key_bucket(key: str, salt: str) -> float:
    """Map a motion key to a stable pseudo-uniform value in [0, 1)."""
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def select_holdout(keys: Iterable[str], fraction: float, salt: str) -> Dict[str, List[str]]:
    """Deterministic hash split. Same (key, salt) → same side, always."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    if not salt:
        raise ValueError("salt must be non-empty (it is the composition secret)")
    heldout, curriculum = [], []
    seen = set()
    for k in keys:
        if k in seen:
            raise ValueError(f"duplicate motion key: {k!r}")
        seen.add(k)
        (heldout if _key_bucket(k, salt) < fraction else curriculum).append(k)
    if not heldout or not curriculum:
        raise ValueError(
            f"degenerate split: {len(heldout)} held-out / {len(curriculum)} curriculum"
        )
    return {"heldout": sorted(heldout), "curriculum": sorted(curriculum)}


def _digest_of(keys: List[str], salt: str) -> str:
    return hashlib.sha256((salt + "\n" + "\n".join(sorted(keys))).encode()).hexdigest()


def write_manifest(
    path: str,
    keys: Iterable[str],
    fraction: float,
    salt: str,
    eval_terminations: str = "terminations/tracking/eval.yaml",
) -> Dict[str, Any]:
    """Create and persist the watcher's private manifest.

    Store OUTSIDE any directory the manager process reads (doc 08: the
    protected metric is protected by process boundary, not instruction).
    """
    split = select_holdout(keys, fraction, salt)
    manifest = {
        "version": MANIFEST_VERSION,
        "fraction": fraction,
        "salt": salt,
        "eval_terminations": eval_terminations,
        "heldout_keys": split["heldout"],
        "n_curriculum": len(split["curriculum"]),
        "integrity": _digest_of(split["heldout"], salt),
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path) as f:
        manifest = json.load(f)
    expect = _digest_of(manifest["heldout_keys"], manifest["salt"])
    if manifest.get("integrity") != expect:
        raise ValueError(f"manifest integrity check failed: {path}")
    return manifest


def curriculum_keys(manifest: Dict[str, Any], all_keys: Iterable[str]) -> List[str]:
    """The training-side allowlist: everything not held out.

    Recomputed from the hash, so a key added to the library after the
    manifest was written still lands on its stable side.
    """
    held = set(manifest["heldout_keys"])
    salt, fraction = manifest["salt"], manifest["fraction"]
    out = []
    for k in all_keys:
        if k in held:
            continue
        if _key_bucket(k, salt) < fraction:
            continue  # new key that hashes into the held-out side
        out.append(k)
    return sorted(out)


def heldout_record_from_metrics_eval(
    metrics_eval: Dict[str, Any],
    manifest: Dict[str, Any],
    it: int,
    strict: bool = True,
) -> Dict[str, Any]:
    """One eval-only pass on the held-out subset → one protected record.

    `metrics_eval` is the parsed metrics_eval.json SONIC writes in eval_only
    mode. Integrity guards (strict): failed_keys must be a subset of the
    manifest's held-out keys — if the eval ran on the wrong motion set, we
    raise instead of emitting a wrong protected metric.
    """
    failed = metrics_eval.get("failed_keys")
    success = metrics_eval.get("eval/success/success_rate")
    if success is None:
        # eval-only runs may carry the scalar under the print-dict name
        success = metrics_eval.get("success_rate")
    if success is None and failed is not None:
        success = 1.0 - len(failed) / len(manifest["heldout_keys"])
    if success is None:
        raise ValueError(
            "metrics_eval has neither 'eval/success/success_rate', "
            "'success_rate', nor 'failed_keys'"
        )
    if strict and failed is not None:
        outside = sorted(set(failed) - set(manifest["heldout_keys"]))
        if outside:
            raise ValueError(
                f"eval ran outside the held-out subset ({len(outside)} foreign "
                f"failed keys, e.g. {outside[:3]}); refusing to emit the "
                "protected metric"
            )
    if not (0.0 <= float(success) <= 1.0):
        raise ValueError(f"success_rate out of range: {success}")

    record: Dict[str, Any] = {
        "it": it,
        "heldout_success_rate": round(float(success), 6),
        "heldout_n_motions": len(manifest["heldout_keys"]),
        "heldout_manifest_integrity": manifest["integrity"],
    }
    if failed is not None:
        record["heldout_failed_count"] = len(failed)
    mpjpe = metrics_eval.get("eval/all/mpjpe_g")
    if isinstance(mpjpe, (int, float)):
        record["heldout_mpjpe_g"] = float(mpjpe)

    # RESOLVING held-out metric (doc 10 §2 gap "held-out metric with
    # resolution" / I2.1). success_rate is all-or-nothing full-clip
    # completion -> 0.0 everywhere at this scale (E3A [verified]); the
    # per-motion PROGRESS (fraction of clip completed before termination) is
    # continuous in [0,1] and already persisted at 0 GPU-h in
    # eval/all_metrics_dict.progress. We surface the aggregate progress_rate
    # AND a per-motion spread so the metric can actually distinguish arms.
    progress_rate = metrics_eval.get("eval/success/progress_rate")
    if progress_rate is None:
        progress_rate = metrics_eval.get("progress_rate")
    if isinstance(progress_rate, (int, float)) and math.isfinite(progress_rate):
        record["heldout_progress_rate"] = round(float(progress_rate), 6)
    amd = metrics_eval.get("eval/all_metrics_dict") or {}
    prog = amd.get("progress")
    keys = amd.get("motion_keys")
    if isinstance(prog, list) and prog:
        vals = [float(p) for p in prog
                if isinstance(p, (int, float)) and math.isfinite(p)]
        if vals:
            # integrity: per-motion arrays must be keyed inside the held-out
            # subset too (strict) — a wrong motion set here would poison the
            # resolving metric just like a wrong success_rate.
            if strict and isinstance(keys, list) and keys:
                outside = sorted(set(keys) - set(manifest["heldout_keys"]))
                if outside:
                    raise ValueError(
                        f"held-out per-motion progress ran outside the subset "
                        f"({len(outside)} foreign keys, e.g. {outside[:3]}); "
                        "refusing to emit the resolving metric")
            n = len(vals)
            mean = sum(vals) / n
            record["heldout_progress_per_motion"] = {
                "mean": round(mean, 6),
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
                "nonzero": sum(1 for v in vals if v > 0),
                "n": n,
            }
    return record


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Held-out watcher core (CPU)")
    sub = p.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make-manifest", help="split a motion-key list")
    mk.add_argument("--keys-file", required=True, help="one motion key per line")
    mk.add_argument("--fraction", type=float, default=0.1)
    mk.add_argument("--salt", required=True)
    mk.add_argument("--out", required=True)

    rec = sub.add_parser("record", help="metrics_eval.json -> protected record")
    rec.add_argument("--metrics-eval", required=True)
    rec.add_argument("--manifest", required=True)
    rec.add_argument("--it", type=int, required=True)
    rec.add_argument("--append-to", help="eval.jsonl to append the record to")

    args = p.parse_args(argv)
    if args.cmd == "make-manifest":
        with open(args.keys_file) as f:
            keys = [line.strip() for line in f if line.strip()]
        m = write_manifest(args.out, keys, args.fraction, args.salt)
        print(f"manifest -> {args.out} ({len(m['heldout_keys'])} held-out / "
              f"{m['n_curriculum']} curriculum)")
    else:
        with open(args.metrics_eval) as f:
            metrics_eval = json.load(f)
        record = heldout_record_from_metrics_eval(
            metrics_eval, load_manifest(args.manifest), args.it
        )
        line = json.dumps(record)
        if args.append_to:
            with open(args.append_to, "a") as f:
                f.write(line + "\n")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
