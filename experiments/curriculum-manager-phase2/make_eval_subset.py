# SPDX-License-Identifier: Apache-2.0
"""Build a fixed 64-motion eval subset from one side of the held-out split.

Same selection rule as heldout/eval_subset64.json (verified reproducible
2026-07-07: sorted(keys, key=_key_bucket)[:64] with the manifest's salt):
deterministic, salt-stable, and — because heldout/curriculum membership is
itself decided by _key_bucket vs fraction — the two subsets can never
intersect.

Why this exists (v4 post-mortem 2026-07-07): the standard per-segment eval
pass passed NO motion_file override, so eval_agent_trl.py inherited the
checkpoint-sibling config's motion_file. In v4 that was robot_curriculum
(116,924 motions) => one eval pass projected 28+ h, timed out the driver at
900 s, kept the GPU busy ~8 h, and starved every subsequent eval AND both
seed-1337 arms. motion_file is the same class of leak as review-M1's
foot_pos_xyz: a checkpoint-config knob eval.yaml does not re-pin. Fix =
pin the standard eval pass to this fixed subset (see smoke_driver.py).

Usage (host):
  python3 make_eval_subset.py --manifest heldout/manifest.json \
      --side curriculum --keys-file /tmp/curriculum_keys.txt \
      --n 64 --out heldout/curriculum_eval_subset64.json

Then materialize the container directory (symlinks are fine; eval reads pkl):
  python3 make_eval_subset.py --materialize heldout/curriculum_eval_subset64.json \
      --src-dir data/motion_lib_bones_seed/robot_curriculum \
      --dst-dir data/motion_lib_bones_seed/robot_curriculum_eval64
(prints the docker exec command; run it, then verify the count is n.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys


def _key_bucket(key: str, salt: str) -> float:
    # identical to sonic-heldout-watcher/holdout.py:_key_bucket
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def build_subset(manifest: dict, side: str, keys: list[str], n: int) -> dict:
    """Select the n lowest-bucket keys from `keys`, verifying every key
    actually belongs to `side` of the manifest's split (fraction boundary)."""
    salt, fraction = manifest["salt"], manifest["fraction"]
    held = set(manifest["heldout_keys"])
    for k in keys:
        on_heldout_side = k in held or _key_bucket(k, salt) < fraction
        if side == "curriculum" and on_heldout_side:
            raise ValueError(f"key {k!r} belongs to the held-out side; "
                             "keys-file is not a curriculum listing")
        if side == "heldout" and not on_heldout_side:
            raise ValueError(f"key {k!r} belongs to the curriculum side")
    if len(keys) < n:
        raise ValueError(f"only {len(keys)} keys for n={n}")
    subset = sorted(sorted(keys, key=lambda k: _key_bucket(k, salt))[:n])
    return {
        "subset_keys": subset,
        "side": side,
        "n": n,
        "manifest_integrity": manifest["integrity"],
        "selection": f"lowest-{n} _key_bucket",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest")
    p.add_argument("--side", choices=["curriculum", "heldout"])
    p.add_argument("--keys-file", help="one motion key per line (.pkl suffix ok)")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out")
    p.add_argument("--materialize", help="subset json -> print docker cp/link cmd")
    p.add_argument("--src-dir")
    p.add_argument("--dst-dir")
    args = p.parse_args(argv)

    if args.materialize:
        with open(args.materialize) as f:
            sub = json.load(f)
        if not (args.src_dir and args.dst_dir):
            p.error("--materialize needs --src-dir and --dst-dir")
        links = " && ".join(
            f"ln -s ../{shlex.quote(args.src_dir.rstrip('/').split('/')[-1])}/"
            f"{shlex.quote(k)}.pkl {shlex.quote(k)}.pkl"
            for k in sub["subset_keys"])
        wbc = "/workspace/GR00T-WholeBodyControl"
        print(f"docker exec isaac-lab-base bash -c "
              f"'mkdir -p {wbc}/{args.dst_dir} && cd {wbc}/{args.dst_dir} && {links}'")
        return 0

    if not (args.manifest and args.side and args.keys_file and args.out):
        p.error("subset mode needs --manifest --side --keys-file --out")
    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.keys_file) as f:
        keys = [line.strip().removesuffix(".pkl") for line in f if line.strip()]
    sub = build_subset(manifest, args.side, keys, args.n)
    with open(args.out, "w") as f:
        json.dump(sub, f, indent=1)
    print(f"{args.out}: {len(sub['subset_keys'])} keys ({args.side} side), "
          f"e.g. {sub['subset_keys'][:2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
