# ON-vs-OFF comparison v3 — first eval-scored manager run

**Status: COMPLETE (2026-07-02). Both arms measured; scored by
`score_comparison_v3.py` from the persisted per-segment metrics_eval.json
files (`comparison_v3_scored.json`). This run demonstrates the
EVAL-SCORED MECHANISM end-to-end; the value signal is at quantization
scale and is NOT claimed — see "What this does NOT show".**

## What v3 changes over the v2 mechanism smoke

The v2 smoke (SMOKE_RESULTS.md) demonstrated the closed loop but carried
the caveat "manager gains are partly definitional — loosening thresholds
lengthens episodes by construction; the tripwire watches training-side
reward, which the manager's own actions inflate." v3 retires the
measurement half of that caveat:

1. **Per-segment eval pass, both arms** (`JobAdapter.eval_segment`):
   eval-only `im_eval` at FIXED relaxed thresholds
   (`terminations/tracking/eval.yaml`) after every segment. Eval is
   deterministic [measured 2026-07-02: same checkpoint → byte-identical
   metrics_eval.json; and both arms' segment-1/2 evals matched to all
   printed digits].
   **The scoreboard boundary must be pinned, not assumed** (adversarial
   review, finding M1 — see "Bugs caught" #3): eval loads the
   checkpoint-sibling config.yaml (which carries manager overrides) and
   merges eval.yaml on top, so a term eval.yaml does not name
   (foot_pos_xyz) leaks the manager's own action into the scoreboard.
   `build_eval_command` now re-pins foot_pos_xyz at stock 0.2, with a
   structural test that fails if a termination knob enters the action
   space without an eval pin.
2. **Eval-side tripwire**: the manager's guard metric is
   `eval/progress_rate` (30% relative drop AND >0.002 absolute, 2
   consecutive evals), not training reward. With the M1 pin in place,
   loosening training thresholds cannot move the guard metric; without
   the pin this claim was FALSE for foot_pos_xyz (it held only for knobs
   eval.yaml explicitly overrides).
3. **Scoring per the baseline-eval-diagnosis** (../baseline-eval-diagnosis/
   RESULTS.md §6): PRIMARY = progress_rate; SECONDARY = mpjpe_l;
   mpjpe_g reported with a survivor-bias warning (it is ANTI-correlated
   with survival — the succeeding release checkpoint scores mpjpe_g 120.9
   vs our failing baseline's 60.7); success_rate is 0.0 for both arms by
   construction at this scale (full-clip all-or-nothing over 2002 frames).

## Protocol

- Both arms START from the overnight baseline's `model_step_002000.pt`
  (10k-iter stock run, seed 42) — the live region of the eval curve
  (progress 0.003 @ 2k → 0.050 @ 4k on the baseline). A fresh start would
  leave the eval scoreboard degenerate-zero for the whole run.
- 6 segments × 50 iters × 256 envs per arm, seed=42 pinned, identical base
  config. Segment N+1 resumes from segment N's snapshot.
- **Base knobs passed explicitly** (`--base-knobs`): the stock strict
  values read from the control run's saved config.yaml
  (anchor_pos 0.15, ee_body_pos 0.15, foot_pos_xyz 0.2). These seed the
  registry belief so notch arithmetic starts from the run's REAL values —
  see "Bugs caught" — and, as identical-value Hydra overrides, leave the
  resolved config equivalent to control (prefix identity held, measured).
- Manager arm: `TrainSideBandPolicy` (band len_low=20, sustain=2,
  binding-axis selection), eval-side tripwire. Control: no changes ever.

## Measured results (journals: `{control,manager}_journal_v3.json`; scored: `comparison_v3_scored.json`)

| Seg | control len / prog / mpjpe_l | manager len / prog / mpjpe_l | manager decision after segment |
|---|---|---|---|
| 1 | 18.02 / 0.00325 / 27.41 | 18.02 / 0.00325 / 27.41 | none (sustain unmet) |
| 2 | 15.37 / 0.00350 / 30.24 | 15.37 / 0.00350 / 30.24 | **loosen `foot_pos_xyz` 0.20→0.25** (binding axis; true one-notch from the real stock value; eval-side tripwire armed at baseline prog 0.00350) |
| 3 | 20.47 / 0.00350 / 29.39 | 20.43 / 0.00350 / 29.09 | none — pending gate (observed, not consulted) |
| 4 | 16.41 / 0.00350 / 27.87 | 21.12 / 0.00375 / 27.35 | none (prior change scored `survived`; len entering band) |
| 5 | 16.56 / 0.00375 / 28.35 | 22.14 / 0.00375 / 25.68 | none (len in band) |
| 6 | 20.72 / 0.00375 / 28.08 | 20.23 / 0.00400 / 28.31 | none (len in band) |

- **Prefix identity holds** [measured]: segments 1–2 identical across arms
  in train metrics (len/rew to 5 decimals) AND eval metrics (all printed
  digits) — arms differ only by the decision.
- **One decision, cleanly attributable**: `foot_pos_xyz` 0.20→0.25 applied
  at cumulative iter 100, journaled with `digest_hash=05f41bb302ad` and
  `applied_at_iter=100`; survived its 2-eval watch on the eval-side metric.
- 0 rollbacks, 0 validator rejections, 0 eval failures in 12 eval passes.
- Final segment: manager progress_rate 0.0040 vs control 0.0037 (one
  quantum ahead), mpjpe_l comparable (28.3 vs 28.1), train len 20.2 vs 20.7
  — i.e. after loosening, the manager arm's episodes got *modestly* longer
  in training (21–22 vs 16–21) while its fixed-threshold eval score did
  not degrade and its local pose error stayed level-to-better.

## What this DOES show

1. **The full doc-08 §3 loop, now eval-scored, runs against real SONIC
   training with no human in the segment loop**: digest → decide →
   validate → apply → per-segment eval at fixed thresholds → eval-side
   tripwire watch → outcome scoring → journal with provenance
   (digest_hash, applied_at_iter).
2. **The v2 "self-inflated tripwire" caveat is retired ONLY with the M1
   pin in place.** In this run the leak was live for segments 3–6 (the
   manager's foot_pos_xyz=0.25 reached its own eval config). The affected
   segments were **re-evaluated from their persisted snapshots with the
   pin** [measured]: all four pinned re-evals came back **byte-identical**
   to the leaked ones — the leaked threshold happened not to change any
   episode's termination in these runs, so the reported numbers stand
   unchanged. That is luck of this run's dynamics, NOT a property of the
   design; the pin + structural test are what make the claim durable.
   Training-side reward collapse alone can no longer trip or mask a
   rollback (unit-tested).
3. **Notch discipline now starts from reality**: the applied change was a
   true one-notch move from the run's actual stock value (0.20→0.25), not
   from a stale registry default (see Bugs caught #1).
4. After loosening, eval progress stayed level-to-one-quantum-ahead and
   mpjpe_l level-to-better while training episodes lengthened (21–22 vs
   16–21). Consistent with a harmless loosen at anecdote scale — though
   note the same-seed shared-luck caveat and that the eval-side
   "no regression" is partly guaranteed by the tiny magnitudes involved.

## What this does NOT show (do not cite otherwise)

- **Not evidence the manager improves training.** The progress_rate
  difference (0.0040 vs 0.0037) is ONE quantization step —
  progress moves in units of 1/(2·2002) ≈ 0.00025 with 2 motions — over
  300 iterations. Direction is right; magnitude is sub-noise. Multi-seed,
  longer runs, and the real library are still required for any value claim.
- **The tripwire never faced a real test**: at these progress values
  (~0.0035) the absolute-floor guard (0.002) makes rollback effectively
  unreachable — that is intended noise protection at tiny magnitudes, but
  it means "0 rollbacks" here is a weak statement. The rollback path is
  exercised by unit tests (fake adapter), not by this live run.
- **Both motions are in the training set**: the eval pass breaks the
  threshold-inflation coupling, NOT the train-on-eval-keys coupling. The
  doc-08 held-out protected metric remains unexercised until bones-seed
  access lands.
- 2 motions, 1 seed, 6 segments: mechanism-scale, not curriculum-value
  evidence.

## Bugs caught (v3 round — the method keeps working)

1. **Registry-default vs actual-config drift [measured].** registry.yaml
   knob defaults (anchor 0.30 / ee 0.30 / foot 0.35) describe the
   Stage-2-patch context, but the stock SONIC config these runs train on
   uses strict 0.15/0.15/0.2. `Registry.current_of()` falls back to the
   registry default when no change has been applied, so in the v2 smoke
   the "one-notch loosen 0.30→0.35" of `ee_body_pos` was ACTUALLY
   0.15→0.35 — a 2.3-notch jump that silently violated max_step semantics
   (the validator compares against the believed value, not the real one).
   v3 fix: the driver seeds `state.current_values` from `--base-knobs`
   (values read from the run's saved config.yaml). Consequence for v2:
   its mechanism conclusions stand, but its applied deltas were larger
   than journaled — flagged retroactively here. Residual: the registry
   should ultimately verify its belief against the run's resolved
   config.yaml itself (queued).
2. **Stock config's adaptive thresholds.** In the STOCK config (unlike the
   Stage-2 patch context) `anchor_pos` and `ee_body_pos` have
   `threshold_adaptive: true` (down_threshold 0.75), so loosening their
   static `threshold` may be partially non-binding — the v1 lesson in a
   new place. `foot_pos_xyz` (no adaptive path, static 0.2) is the
   reliably binding axis; the binding-axis rule steered the live decision
   exactly there [measured: decision rationale cites its windowed
   termination fraction].
3. **Eval scoreboard leak (adversarial review, MAJOR M1).**
   `eval_agent_trl.py:79–112` loads the checkpoint-sibling config.yaml
   (containing manager-applied overrides) and merges the eval config on
   top; `tracking/eval.yaml` names only anchor_pos/anchor_ori_full/
   ee_body_pos, so the manager's foot_pos_xyz=0.25 survived the merge
   into its own s3–s6 evals [measured: reviewer reproduced the resolved
   merge in-container — control s3 eval foot=0.2 vs manager s3/s6
   foot=0.25; all eval logs list 5 active terms incl. foot_pos_xyz].
   Remediation: `build_eval_command` re-pins foot_pos_xyz at stock 0.2
   (+ structural test); manager s3–s6 re-evaluated from persisted
   snapshots with the pin → **byte-identical outputs** [measured:
   `cmp_manager_manager_s{3..6}_eval_pinned/`], so the reported table
   stands. The same merge also invalidated the diagnosis doc's
   "foot_pos_xyz absent at eval" claim — corrected there (its §1/§3).
4. **Stale-eval tripwire hole (adversarial review, MAJOR M3).** With a
   watch armed, a segment whose eval pass FAILED re-read the newest
   (stale, pre-change) eval record — which equals the armed baseline and
   can never breach — so two consecutive eval failures scored a change
   `survived` on zero post-change evidence. Did NOT fire in this run
   (0 eval failures) [measured: journals]. Fixed: `_tripwire_value` now
   requires the record to come from the current segment (`it` match);
   arming likewise refuses a stale baseline; 2 new regression tests.

## Protocol disclosure (review m1)

The two arms ran under different driver revisions: control ran first
(without `--base-knobs`; its journal predates the eval-key extension), and
the manager arm ran after the registry-seeding fix. Equivalence of the
training configuration was established by MEASUREMENT, not identical
invocation: the two s1 config.yamls differ only in the output-dir
timestamp; s1–s2 train metrics are line-identical and s1–s2 eval outputs
byte-identical across arms [measured by the reviewer]. `run_comparison_v3.sh`
as committed reproduces both arms under the current driver.

## Next

1. bones-seed access → real library → held-out split → the doc-08
   protected metric proper (heldout_success_rate / heldout progress).
2. Multi-seed (≥2) × longer segments; then the LLM policy arm (Phase-1
   `LLMPolicy` plugs into the same `propose()` interface — now with the
   eval stream in its digest).
3. Registry verifies believed knob values against the run's resolved
   config.yaml (close Bugs-caught #1 structurally).
4. Remaining harness debt: registry-level pending gate;
   `expected_effect` scoring (outcome is still only `survived`).

## Reproduce

```bash
PY=~/.local/bin/python3.10
$PY -m pytest test_smoke_driver.py -q            # 23 passed (fake adapters)
bash run_comparison_v3.sh control manager        # ~80 min on the A10G
$PY score_comparison_v3.py control_journal_v3.json manager_journal_v3.json
```
Artifacts in-container: `/workspace/wbc-training-logs/cmp_{control,manager}*`
(per-segment train logs, snapshots, eval dirs with metrics_eval.json).
