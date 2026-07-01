# RLVR Verifier-Reward Reference Spike — RESULTS

**Status:** ✅ runs today on CPU with synthetic/tiny data. No cluster, no GPU, no model training.

This spike is the **"reward engine"** half of the reverse-transfer plan
(`docs/design/07-review-and-revised-roadmap.md` §5, line 104):

> "a rule/verifier reward over grounded/structured QA (multiple-choice,
> numeric, IoU, referring-expression match) using standard RLVR."

SONIC supplies the RL chassis (rollouts/PPO); the **verifier reward is net-new
for TAO**. This directory is a runnable reference implementation of that
verifier reward, plus a tiny toy optimizer that proves the reward closes an RL
loop.

---

## Files

| File | What it is |
|------|-----------|
| `verifiers.py` | Four pure, deterministic verifiable-reward functions, each returning a scalar in `[0,1]`. |
| `rlvr_demo.py` | Toy REINFORCE policy-improvement demo driven by the verifier rewards. `[measured]`-on-synthetic. |
| `test_verifiers.py` | 24 known-answer pytest cases (also runnable as a stdlib self-test). |
| `RESULTS.md` | This file. |

Run environment: `/workspace/Isaac-GR00T/.venv/bin/python` (Python 3.10.20,
numpy 1.26.4, pytest 9.1.1 — pytest installed into the venv via `uv pip install`).
Dependencies: **numpy + stdlib only**. Seeded, deterministic, < 1 s total.

---

## The four verifiers and their TAO data-skill sources

TAO skill provenance below is **[verified]** by reading each skill's `SKILL.md`
under `skills/data/` (Outputs sections cited).

### 1. `multiple_choice_exact(prediction, gold) -> {0.0, 1.0}`
Exact-match reward for a chosen option. Accepts letters (`"B"`, `"b."`) or
integer indices, with letter↔index equivalence (`"A" == 0`).

- **Fed by:** `tao-generate-video-reasoning-annotations` **[verified]**.
  Its Step-4 output writes `mcq.json` / `bcq.json` in the `tao-vl-reason-v1.0`
  envelope, each `items[]` entry having `{question, answer, reasoning}`
  (SKILL.md Outputs, lines 162–173). The gold `answer` is the choice this
  verifier scores against.

### 2. `numeric_tolerance(prediction, gold, tol, shaped=True) -> [0.0, 1.0]`
`err = |pred - gold|`. Outside `tol` → 0.0. Inside `tol`: shaped reward
`1 - err/tol` (dense gradient) or hard 1.0/0.0 when `shaped=False`. `tol>0`
enforced.

- **Fed by:** `tao-generate-video-reasoning-annotations` **[verified]**.
  The same Step-4 QA output includes open-ended / counting /
  temporal-localization questions whose gold `answer` is numeric (counts,
  durations, timestamps). Tolerance is task-appropriate.

### 3. `iou_reward(pred_bbox, gold_bbox) -> [0.0, 1.0]`
Real intersection-over-union of two axis-aligned `[x1,y1,x2,y2]` boxes.
Inverted corners normalized; degenerate zero-area → 0.0.

- **Fed by:** `tao-generate-image-grounding` **[verified]** — Step-1 output
  stores `expressions[].instances[].bbox = [x1,y1,x2,y2]` in pixel space with a
  `score` (SKILL.md Outputs, line 118). Also `tao-generate-referring-expressions`
  **[verified]** — Step-2 grounding output stores
  `expressions[] = {text, instances:[{bbox:[x1,y1,x2,y2]}]}` (SKILL.md Outputs,
  line 133). The gold bbox is the teacher-annotated box; the policy predicts a box.

### 4. `referring_expression_match(prediction, gold, mode="token_f1") -> [0.0, 1.0]`
Normalized string match. `"exact"` → 1.0 iff normalized token sequences equal;
`"token_f1"` → bag-of-tokens F1 (graded partial credit). Normalization
lowercases, strips punctuation, drops articles {a, an, the}.

- **Fed by:** `tao-generate-referring-expressions` **[verified]** — Step-0
  region expressions (`regions[]` with `description`) and Step-2 grounding
  expressions (`expressions[].text`), refined by the Step-3 double-check
  (SKILL.md Outputs, lines 131–135). The gold phrase is the teacher's referring
  expression; the policy generates a phrase.
- **Note:** BERTScore-F1 is the noisier *dense* fallback flagged in the design
  doc; token-F1 is the cheap deterministic rule used here. A learned reward
  model is explicitly out of scope (higher spike).

---

## Test results `[measured]`

Command: `python -m pytest test_verifiers.py -v`

```
============================= test session starts ==============================
platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0
collected 24 items
... (24 tests) ...
============================== 24 passed in 0.04s ==============================
```

**Summary line: `24 passed in 0.04s`** — every verifier has known-answer
coverage including boundary/edge cases:
- IoU of two known half-overlapping boxes = **1/3** (hand-derived), contained
  box = **0.25**, identical = 1.0, disjoint = 0.0, inverted-corner = 1.0,
  degenerate = 0.0.
- numeric tolerance: exact hit = 1.0, mid-band (`err=0.5,tol=1`) = 0.5,
  boundary (`err==tol`) = 0.0, outside = 0.0, hard pass/fail, `tol<=0` raises.
- MC: letter hit/miss, case+punctuation insensitivity, letter↔index equivalence,
  int↔int.
- ref-exp: exact-after-normalization, partial token-F1 = **0.75** (hand-derived),
  no-overlap = 0.0, empty/empty = 1.0, empty/one-side = 0.0.

---

## Toy policy-improvement demo `[measured]`-on-synthetic

Command: `python rlvr_demo.py`  (exit 0 iff both curves improve)

Two numpy toy policies optimized by vanilla REINFORCE using *only* the verifier
rewards:

- **Demo A — softmax bandit over MC letters**, reward = `multiple_choice_exact`.
  Reward-mean **0.438 → 0.991** over 300 iters; final `P(gold letter 'C') = 0.991`.
- **Demo B — 1-parameter Gaussian numeric guesser**, reward =
  `numeric_tolerance` (shaped, `tol=3`), sigma annealed 6→1.
  Reward-mean **0.090 → 0.728** over 800 iters; final `mu = 7.013` vs `gold = 7.0`.

**Summary line:**
`SUMMARY: MC 0.438 -> 0.991 | numeric 0.090 -> 0.728 | BOTH CURVES IMPROVED (reward engine closes the loop)`

Reward-vs-iteration sparklines (start→end) both rise:
```
MC bandit (REINFORCE):  ▁▁▂▄▆█▆▅▆▇███▆█▆▆███▇████████████▇█▆██▇█▇█████████
numeric guesser      :  ▁▁▁▁▁▂▂▁▁▁▁▂▂▂▂▂▃▄▃▄▃▄▄▄▄▃▃▅▄▄▅▅▅▅▆▅▆▆▇▆▇█▇▇▇▇▇▇▇▇
```

Note on the numeric demo: it needs a wide-then-annealed exploration `sigma` so
the initial guesses overlap the reward band — this is an honest RLVR
exploration constraint (sparse reward gives no gradient), not a tuning trick.

---

## What this IS / what this is NOT

**IS:**
- A reference implementation of the RLVR **verifier-reward interface** over
  structured QA — the "reward engine" the reverse-transfer track identified as
  the missing piece.
- Pure, deterministic, `[0,1]`-bounded, unit-tested reward functions tied to
  TAO's *existing* data skills, so the reward inputs are realistic.
- A tiny mechanism demo proving those rewards are optimizable — reward goes up
  under a standard REINFORCE update.

**IS NOT:**
- ❌ A real VLA/LLM RL run. No GR00T, no cosmos-reason, no PPO on a transformer.
- ❌ Trained TAO/GR00T. No `train.train_policy.type` enum unlock, no Cosmos-RL,
  no cluster, no GPU.
- ❌ A quality claim. The policies are numpy toys (a 4-way bandit and a
  1-parameter guesser); the tasks are synthetic constants, not real images/videos.
- ❌ A learned reward model or BERTScore integration (both explicitly out of
  scope; noted as higher spikes in the design doc).

All numbers above are `[measured]` from the real runs in this directory.
Remaining forward work (unlocking the SFT-only enum, wiring these rewards into
Cosmos-RL rollouts) is `[speculative]` and outside this spike.
