---
name: sonic-heldout-watcher
description: >-
  Producer of the Curriculum-Manager's protected metric
  (heldout_success_rate). Deterministically hash-splits the motion library
  into a held-out subset the manager can never see, reweight, or filter;
  turns SONIC eval-only passes on that subset (at fixed relaxed thresholds)
  into eval.jsonl records for sonic-run-digest; refuses to emit a metric if
  the eval demonstrably ran on the wrong motion set. Use when setting up the
  protected metric for a managed run, generating heldout_success_rate
  records, or auditing metric integrity.
license: Apache-2.0
compatibility: >-
  Core (split/manifest/record) is pure Python 3.9+ stdlib, CPU-only,
  tested now. The live watcher loop additionally requires the SONIC eval
  stack (IsaacLab + eval_agent_trl.py) — wiring documented below, runnable
  only where SONIC eval runs.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- tao
- sonic
- curriculum
- rl
- agentic
- guardrails
- evaluation
---

# SONIC Held-Out Watcher (the protected metric)

Design doc 08, axiom 5: *"protect the manager's metric from the manager"* —
the sharpest 2024–26 agentic-ML lesson (METR o3 reward-hacked 10/10 runs;
prompt-level honesty demonstrably fails). Protection here is **structural**:

1. The held-out subset's composition lives in a **manifest the manager
   process never reads** (different directory/user; the manager only ever
   sees the scalar records).
2. Membership is **hash-based + salted** — stable under library growth, not
   reconstructible without the salt.
3. Eval runs at the **fixed relaxed thresholds**
   (`terminations/tracking/eval.yaml`, 0.25) which are deliberately absent
   from `sonic-knob-registry`.
4. The record producer **refuses** to emit a metric if the eval's
   `failed_keys` contain motions outside the manifest — a mis-wired eval
   can never silently feed the manager a wrong number.
5. The manifest carries an integrity digest; tampering (e.g. dropping a
   hard motion from the held-out set) fails `load_manifest`.

## Files

- `holdout.py` — `select_holdout` (deterministic salted hash split),
  `write_manifest`/`load_manifest` (integrity-checked),
  `curriculum_keys` (training-side allowlist: excludes held-out keys AND
  new keys that hash held-out-side), `heldout_record_from_metrics_eval`
  (SONIC `metrics_eval.json` → one protected eval.jsonl record), CLI.
- `test_holdout.py` — 13 tests, including an end-to-end check that the
  produced records flow through `sonic-run-digest` into
  `eval.heldout_success_rate` with correct trends.

## Quick start (CPU, now)

```bash
cd skills/agentic/sonic-heldout-watcher
python3 -m pytest test_holdout.py -q     # 13 passed

# 1) one-time: split the motion library (salt is the composition secret —
#    store the manifest OUTSIDE anything the manager reads)
python3 holdout.py make-manifest \
  --keys-file motion_keys.txt --fraction 0.1 \
  --salt "$(head -c16 /dev/urandom | xxd -p)" \
  --out /secure/heldout_manifest.json

# 2) per eval pass: metrics_eval.json -> protected record
python3 holdout.py record \
  --metrics-eval outputs/eval/metrics_eval.json \
  --manifest /secure/heldout_manifest.json \
  --it 12500 --append-to run_logs/eval.jsonl
```

## Live wiring (requires SONIC eval stack — Phase 2)

Verified seams in pinned WBC `0e35637`:

- **Subset restriction:** `eval_agent_trl.py` already assigns
  `env_config.commands.motion.filter_motion_keys` (and
  `motion_lib_cfg.filter_motion_keys`) to run eval on a chosen key list
  (`eval_agent_trl.py:316–318`, `367–369`; consumed `commands.py:201`).
  The watcher passes `manifest["heldout_keys"]` there.
- **Fixed thresholds:** launch eval with
  `manager_env/terminations=tracking/eval` (relaxed 0.25 set) — never the
  strict training composition.
- **Output:** eval-only mode writes `metrics_eval.json` with
  `eval/success/success_rate`, `failed_keys`, `failed_idxes`
  (`im_eval_callback.py:135–155`, `227–231`) → feed to `record`.
- **Cadence:** one held-out pass per manager tick (~every 250 training
  steps' eval slot, or coarser — the digest tolerates gaps).
- **Training-side counterpart:** restrict the *curriculum* to
  `curriculum_keys(manifest, all_keys)` via the same `filter_motion_keys`
  seam on the training config, so held-out motions are never trained on.

## Honest caveats

- Process separation is a **deployment obligation, not a property of this
  code**: run the watcher under a different user/directory, and never point
  the manager's digest builder at the manifest. The code makes violations
  detectable (integrity digest, foreign-key refusal), not impossible.
- The foreign-key guard is one-sided: it catches evals on the wrong set
  when failures reveal it, but an eval on a *subset* of the held-out set
  with no foreign failures passes. `heldout_n_motions` is in every record
  so downstream can spot shrinkage.
- Splitting by motion key does not prevent *distributional* leakage
  (similar clips on both sides). That is acceptable: the metric guards
  against the manager gaming its curriculum, not against generalization
  claims.

## Related

- `sonic-run-digest` — consumes the records this produces
  (`heldout_success_rate`).
- `sonic-knob-registry` — deliberately excludes everything this skill owns.
- `sonic-curriculum-manager` — hard rule 4: no held-out metric → no action.
