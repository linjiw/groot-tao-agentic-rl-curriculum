# Manager ON-vs-OFF smoke — first closed-loop manager run on real SONIC training

**Status: mechanism DEMONSTRATED end-to-end (2026-07-01; v2 after
adversarial review). 13/13 driver tests; 2 live arms × 6 segments × 10
iters × 64 envs on the A10G, pinned seed [measured]. This is a MECHANISM
smoke, not evidence the manager improves training — see "What this does
NOT show".**

This experiment went through a project-aware adversarial review between v1
and v2; the review found two MAJOR issues in v1 (both fixed and re-run) —
see "Bugs caught" below. Review agent's findings are reflected throughout.

## Setup

`smoke_driver.py` — first composition of all validated pieces against real
training: `sonic-job-adapter` segments → console-log parse →
`sonic-run-digest` → policy propose → `sonic-knob-registry` validate →
apply-as-next-segment-overrides (or tripwire rollback) → journal.

- Arms: `control` (no knob changes ever) vs `manager`
  (`TrainSideBandPolicy`: episode-length bands, PBHC-style loose-first).
- Both arms: fresh start, **seed=42 pinned in the launch command**, identical
  base config, 6 segments × 10 iters × 64 envs; segment N+1 resumes from
  segment N's snapshot.
- Policy targets the **binding termination axis** by windowed mean
  (`termination_terms_mean_recent`, added to the digest builder), and the
  driver enforces a **pending-decision gate**: while a change is under
  tripwire watch (2 segments), no new change — the policy isn't even
  consulted. Changes that survive the watch are scored `survived` (tripwire survival only — expected_effect satisfaction is NOT checked); rollbacks mark
  the originating decision `failed_rolled_back`.

## Measured results (v2 journals: `{control,manager}_journal_v2.json`)

| Segment | control len / rew | manager len / rew | manager decision after segment |
|---|---|---|---|
| 1 | 11.57 / 0.909 | 11.57 / 0.909 | none (sustain unmet) |
| 2 | 13.90 / 1.113 | 13.90 / 1.113 | **loosen `ee_body_pos` 0.30→0.35** (binding: windowed mean 0.55, cited in rationale) |
| 3 | 12.58 / 0.955 | 14.20 / 1.067 | none — pending gate (change under tripwire watch) |
| 4 | 12.73 / 0.989 | 14.95 / 1.083 | **loosen `foot_pos_xyz` 0.35→0.40** (binding: 0.84; prior change scored `survived`) |
| 5 | 11.86 / 0.905 | 22.43 / 1.484 | none — pending gate |
| 6 | 11.94 / 0.937 | 21.07 / 1.329 | none (len entering band; prior change `survived`) |

- Segments 1–2 identical across arms to 5 decimals (pinned seed) — arms
  differ only by decisions.
- 0 rollbacks, 0 validator rejections; both applied decisions scored `survived`
  after surviving their 2-segment tripwire watch.
- **Attribution caveat:** only the FIRST change (seg 2→3 divergence) is
  cleanly attributable. Segments 4–6 reflect the joint effect of both
  changes; per-decision credit beyond the first is confounded.

## Bugs caught (two rounds — why we run live AND review adversarially)

1. **Binding-axis (caught by the live run, v1):** the first manager run
   hardcoded `anchor_pos` (as in the Phase-1 toy) and produced
   **byte-identical metrics to control** despite "applying" a change —
   `anchor_pos` terminated ~0 episodes in these runs (its
   `threshold_adaptive` path appears to make the strict value non-binding
   here — hypothesis, not verified in source) while `foot_pos_xyz`/
   `ee_body_pos` did the terminating (fractions read from in-container
   console logs, now cited per-decision in journal rationales). Loosening a
   term that never fires is a no-op. Fixes: digest exposes per-term
   termination fractions (last + windowed mean); policy selects the binding
   axis. The playbook's rows 1–2 were updated to the same rule.
2. **Overlapping changes (caught by the adversarial review, v1):** v1's
   driver let a second change arm while the first was still under tripwire
   watch — overwriting `self.armed`, orphaning the first tripwire, and
   making its rollback point include the second change. v1's tick-3 change
   therefore had an unguarded predecessor, and the playbook's own "pending
   → none" rule was being violated. Fixes: pending-decision gate in the
   driver, survived-watch scoring (`survived`), rollback marks the origin
   decision `failed_rolled_back`, no-baseline changes refused. v1 journals
   preserved (`manager_journal.json`, `manager_journal2.json`) for the
   record; v2 is the clean protocol.

## What this DOES show

1. The doc-08 §3 loop **minus the protected metric** runs against real
   SONIC training with no human in the segment loop: digest → decide →
   validate → apply → tripwire watch → outcome scoring → journal.
2. The first manager decision has a real, cleanly-attributable training
   consequence (identical prefix, divergence exactly at the first applied
   change).
3. Two transferable curriculum lessons: (a) threshold curricula must target
   the *binding* axis; (b) one-change-pending discipline must be enforced
   by the harness, not assumed of the policy.

## What this does NOT show (do not cite otherwise)

- **Not evidence the manager improves training.** Loosening termination
  thresholds mechanically lengthens episodes, and longer episodes
  accumulate more per-episode reward — the manager arm's higher len/rew is
  **partly definitional**, not (necessarily) better tracking. "Helps vs
  hurts" requires tracking-error at FIXED thresholds (eval passes) — not
  yet wired.
- The doc-08 **protected-metric discipline is NOT exercised** (2-motion
  library; bones-seed gated): the tripwire watches training-side reward,
  which the manager's own actions inflate — exactly the coupling the
  held-out metric exists to break. Every rationale carries this label.
- 6 segments, 1 seed, 64 envs, 2 motions: anecdote-scale mechanism demo.
- Same-seed prefix identity also means shared initialization luck; real
  comparisons need multiple seeds.

## Next

1. bones-seed access → real library → held-out watcher wiring → protected
   metric + eval-side MPJPE at fixed thresholds.
2. Per-segment `im_eval` passes so "helps vs hurts" is measurable.
3. Multi-seed; longer segments; then the LLM policy arm (Phase-1
   `LLMPolicy` plugs into the same `propose()` interface).
4. Registry-level enforcement of the pending gate (currently driver-level),
   incl. machine-enforcing playbook hard-rule 4 for Family-B knobs.
5. Feed observations to the policy during gated ticks (observe-but-don't-act)
   — currently gated segments leave holes in the policy's sustain history,
   so a "for N segments" rationale can span non-adjacent segments.
6. Score decisions against their stated `expected_effect` (today's
   `survived` = tripwire survival only), and add `digest_hash` /
   `applied_at_iter` to journal entries (doc-08 §3 step 4).

## Reproduce

```bash
PY=~/.local/bin/python3.10
$PY -m pytest test_smoke_driver.py -q            # 13 passed (fake adapter)
$PY smoke_driver.py --arm control --segments 6 --iters 10 --journal-out control_journal_v2.json
$PY smoke_driver.py --arm manager --segments 6 --iters 10 --journal-out manager_journal_v2.json
```
Artifacts in-container: `/workspace/wbc-training-logs/smoke_{control,manager}/`.
