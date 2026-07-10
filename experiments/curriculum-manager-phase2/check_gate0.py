# SPDX-License-Identifier: Apache-2.0
"""G0 gate-0 checker (doc 10 §4-G0): the tier-0 INSERTION must be inert.

Compare a `stock` smoke journal against a `noop` smoke journal (tier-0 shim
inserted with SONIC_TIER0_ACTIVE=0). The insertion is inert iff the two
journals are BIT_IDENTICAL on the gated fields — the no-op reward term
returns the stock reward object unchanged, so anything but bit-identity is a
real defect (import side effects, op reordering), not a tolerance question.

Usage:
  python3 check_gate0.py --stock stock_journal.json --noop noop_journal.json

Exit 0 = PASS (bit_identical), exit 1 = FAIL (anything else) + a diagnostic.
Uses the shipped run-manager-core equivalence gate (E6), imported by path so
this stays runnable from the phase-2 dir without install.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_RMC = os.path.join(os.path.dirname(__file__), "..", "run-manager-core")


def _load_equivalence():
    # import core.equivalence + its intra-core deps by adding run-manager-core
    # to sys.path (same trick the phase-2 driver uses for job_adapter).
    sys.path.insert(0, os.path.abspath(_RMC))
    from core import equivalence  # noqa: E402
    return equivalence


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="G0 gate-0 bit-identity check")
    p.add_argument("--stock", required=True, help="stock arm journal JSON")
    p.add_argument("--noop", required=True, help="noop (tier-0 inert) journal JSON")
    p.add_argument("--fields", nargs="+",
                   default=["rew_mean_last", "len_mean_last"],
                   help="gated per-segment journal fields")
    args = p.parse_args(argv)

    eq = _load_equivalence()
    with open(args.stock) as fh:
        stock = json.load(fh)
    with open(args.noop) as fh:
        noop = json.load(fh)

    # tau is irrelevant for a bit-identity check, but compare_journals needs a
    # numeric tau to render a non-bit-identical verdict without raising; pass
    # the measured production tau so a near-miss is DESCRIBED, not an error.
    report = eq.compare_journals(stock, noop, tau=eq.measured_tau(),
                                 fields=args.fields)
    passed = report.verdict == eq.VERDICT_BIT_IDENTICAL
    out = {
        "gate": "G0 gate-0 (insertion inert / bit-identical)",
        "verdict": report.verdict,
        "pass": passed,
        "tau": report.tau,
        "fields": {f: {"verdict": r.verdict,
                       "mean_rel_dev": getattr(r, "mean_rel_dev", None)}
                   for f, r in report.fields.items()},
    }
    print(json.dumps(out, indent=2))
    if not passed:
        print("\nFAIL: the tier-0 insertion is NOT inert. The no-op shim must "
              "return the stock reward object unchanged; a non-bit-identical "
              "result is a defect (import side effect / op reorder), not "
              "tolerance. Do NOT proceed to G1/G2 until this is bit_identical.",
              file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
