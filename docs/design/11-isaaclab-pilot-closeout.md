# 11 — IsaacLab pilot closeout (doc 10 I0.2)

**Status: closeout, 2026-07-10.** The IsaacLab agentic-RL pilot (off-repo at
`/workspace/IsaacLab/skills/agentic-rl/`) is paused per doc 10's decision (SONIC
library-scale gate is primary; IsaacLab re-enters at paper time as the generalization
platform if C1 holds). This doc records what the pilot established so its findings don't
rot outside the repo, per the pilot's own `EVAL_FRAMEWORK.md §5` (machinery-validation
claims only; n=1 supports no comparative verdict).

Claim labels: [verified] structural/artifact fact · [measured] recomputed from the
pilot journals · [design] judgment.

## What the pilot is

A port of the SONIC run-manager scaffold to IsaacLab (rl_games/skrl), on a locomotion
velocity task (`Isaac-Velocity-Rough-Anymal-C-v0`, skrl PPO). Three arms —
control / manager / scripted — warm-started from a shared checkpoint, 6 segments ×
150 iters × 1024 envs, held-out protected metric on a salted-hash command-grid split.
It vendored the engine-agnostic `run-manager-core` and implemented one `EngineAdapter`
for IsaacLab. Design + gaps: the pilot's `DESIGN.md`, `EVAL_FRAMEWORK.md`, `GAPS.md`.

## What it established [verified / measured]

### M1 — the scaffold ports and runs end-to-end on a second engine [verified]
The full tick loop (digest → propose → validate → apply → pinned eval → tripwire watch →
journal) runs on IsaacLab with a single new adapter. Manager + scripted arms completed
6/6 segments cleanly; journals carry gated decisions with provenance. This is the
generalization evidence the scaffold needed: it is not SONIC-specific.

### M2 — the manager makes real gated closed-loop decisions [measured]
Manager journal: 3 hardening moves on `command_range_lin_vel_x` (1.0→1.25→1.5→1.75) at
ticks 2/4/6, each correctly gated — ticks 3/5 held `none` citing "one change under
tripwire watch at a time" (the pending-gate), and tick 1 held `none` on hard-rule 4
(no held-out metric yet). Decisions carry rationale, expected_effect, tripwire spec.

### M3 — **the phase-2 E1 null reproduces on IsaacLab at n=1** [measured]
The manager and scripted arms produced a **bit-identical** held-out trajectory —
`heldout_success_rate` = [0.3415, 0.3471, 0.3888, 0.3789, 0.3714, 0.3718] for BOTH,
including identical per-condition breakdowns (final worst-condition
`vx=1.0,wz=-1.0` = 0.235, spread [0.235, 0.483]). Because both arms walk the identical
knob ladder and the scripted arm replays it open-loop, closed-loop adaptivity added
nothing measurable — the same result phase 2's E1 reached on SONIC, now on a second
engine. (n=1, so this is corroboration of the mechanism, not a comparative claim.)

### M4 — the lever barely moves the protected metric (the A0 finding) [measured]
Despite 3 hardening moves, held-out moved 0.34→0.39→0.37 — range ~0.047, within
run-to-run drift. Per the pilot's `GAPS.md A0`: widening the *training* command range
changes the training distribution but is evaluated on a FIXED in-envelope grid, so its
effect on in-envelope held-out is ~0. **This is the doc 10 §0.1 lever–metric coupling
problem, independently reproduced on a different platform the same week** — and the
direct motivation for doc 10's validity gate V5 (lever sensitivity is a precondition).

## What it did NOT establish (n=1, honest limits — EVAL_FRAMEWORK §5)

- No comparative verdict (manager vs scripted vs control): needs the 3-seed campaign
  against a measured IsaacLab noise floor (the pilot's `GAPS.md` [BLOCK] items).
- Control arm: still running at closeout (seg 4/6); its numbers are not in this doc.
  A control ≈ manager ≈ scripted result would be the expected null, but is unclaimed
  until measured.
- No IsaacLab τ (chaos floor) measured — the pilot correctly flags that the SONIC
  τ=3.9e-2 does not transfer (different engine/env-count/horizon).
- The "manager" is a deterministic band-stepper, not an LLM — the LLM arm is unbuilt.

## Why it pauses (doc 10 decision)

The pilot's central problem (M4 lever insensitivity) is the SAME one doc 10 exists to
fix on SONIC: use a lever that grips the training distribution (per-step tier-0 σ-EMA),
gated on a measured noise floor and a lever-sensitivity precondition. Rather than run
two under-powered platforms in parallel on one A10G, doc 10 concentrates GPU on the
SONIC gate (G0→G2). **IsaacLab re-enters as the generalization arm** (doc 10 §2.3): if
σ-EMA clears the SONIC gate (C1), the same tier-0 controller + adapter port becomes the
cross-embodiment breadth evidence for the paper. The scaffold M1 result is what makes
that cheap when the time comes.

## Carry-forward into doc 10

- V5 (lever sensitivity) is vindicated by M4 — keep it as a hard precondition.
- M3 (E1 null on a 2nd engine) strengthens the paper's methodology claim (M): the
  "adaptivity collapses to a schedule on low-entropy deterministic action spaces"
  finding is now two-platform.
- The pilot's three integration bugs + operating-point calibration lessons live in its
  `GAPS.md` §F; port them into the IsaacLab track's DESIGN when it resumes.
