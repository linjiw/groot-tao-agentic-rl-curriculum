---
name: tao-curriculum-rl
description: >-
  Reverse-transfer workflow skill folding GR00T/SONIC-style curriculum learning
  and RL structure INTO TAO agentic post-training. Documents (A) curriculum-over-SFT
  that is feasible TODAY via real TAO/GR00T seams (data.ds_weights_alpha annealing,
  dataset mix_ratio scheduling, per-sample loss weighting at Gr00tTrainer.compute_loss,
  stage transitions gated on eval_fn), and (B) the gated path to true RL, which is
  BLOCKED until the SFT-locked schema enum is unlocked and a reward verifier is
  supplied. Use when the user asks about adding RL to TAO, curriculum learning over
  SFT, difficulty scheduling, staged/curriculum fine-tuning, reward verifiers,
  ds_weights_alpha, loss reweighting, or SONIC/GR00T RL transfer into TAO.
license: Apache-2.0
compatibility: >-
  Design + scaffold skill. Curriculum-over-SFT paths reference GR00T/cosmos-reason
  training seams that require docker + nvidia-container-toolkit to actually run.
  RL mode is not runnable today (see RL mode section).
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- tao
- curriculum
- rl
- workflow
- agentic
- sft
- scheduling
---

# TAO Curriculum + RL (reverse-transfer)

Every technical claim below is tagged **[verified]** (re-checked against pinned
source this session), **[design-doc]** (proposed in
`docs/design/07-review-and-revised-roadmap.md` §5, not yet built), or
**[speculative]**.

## Overview — the reverse-transfer premise [verified]

The forward track imports TAO skills into GR00T/SONIC. This skill is the
**reverse** track: fold SONIC-style curriculum + RL structure *into* TAO
agentic post-training.

The core asymmetry (the whole point): **SONIC hands TAO a complete, verified RL
*chassis* — but not the *engine*, i.e. the reward.** SONIC's reward is a dense
analytic tracking kernel; the TAO/VLM equivalent is a sparse or learned
**verifier**, which is net-new work. So:

- The RL *chassis* (GAE/PPO surrogate patterns, KL-adaptive LR controller,
  anti-over-concentration prompt sampler, `schedule_dict` serialization) is
  transferable as code patterns / new components. [design-doc]
- The RL *engine* (the reward/verifier) **does not transfer and must be built**.
  This is the pivot that gates the whole RL path. [verified]

What *is* feasible today is the **curriculum** half, which needs no reward: two
clean seams already exist in the GR00T training stack, and TAO's own eval hooks
can drive stage transitions.

## Curriculum-over-SFT (feasible now) [verified]

This is real SFT with a schedule layered on top. No reward, no rollout, no
schema change — it stays inside `train_policy.type == "sft"`.

Three composable knob families:

### 1. Anneal `data.ds_weights_alpha` across stages

The single-scalar data-mixture curriculum knob.

**Seam:** `external/Isaac-GR00T/gr00t/data/dataset/factory.py:78`

```python
# factory.py:78–85 (verified this session)
alpha = self.config.data.ds_weights_alpha
if alpha is not None and len(all_datasets) > 1:
    ds_lengths = np.array([len(dataset) for dataset in all_datasets], dtype=np.float64)
    all_weights = (np.power(ds_lengths, alpha) / np.power(ds_lengths[0], alpha)).tolist()
    print(
        f"Applied ds_weights_alpha={alpha} across {len(all_datasets)} datasets; "
        "this overrides per-dataset mix_ratio sampling weights."
    )
```

`alpha` reweights each dataset by `length^alpha` (normalized to the first
dataset). At `alpha = 0.0` every dataset gets equal weight (uniform mixture,
ignores size). As `alpha → 1.0` sampling weight tracks dataset length
(size-proportional). Annealing `alpha` across epochs = a **data curriculum**:
start uniform (expose the model evenly to all sub-distributions, including small
hard sets), then shift toward the natural size-proportional mixture.

> Note: `ds_weights_alpha` **overrides** per-dataset `mix_ratio`
> (see the print statement and `factory.py:74`, `weight = relative_length *
> dataset_spec.mix_ratio`). Pick *one* mixture-control lever per stage: alpha
> **or** mix_ratio, not both.

**Worked example — alpha schedule 0.0 → 1.0 over N=4 stages:**

Each stage is a short SFT run resuming from the prior stage's checkpoint, with
only `data.ds_weights_alpha` changed in the training spec:

```text
stage 0:  ds_weights_alpha = 0.0    # uniform mixture — warm all sub-datasets
stage 1:  ds_weights_alpha = 0.33   # begin tilting toward larger sets
stage 2:  ds_weights_alpha = 0.66
stage 3:  ds_weights_alpha = 1.0    # size-proportional (natural) mixture
```

Where it plugs in: `data.ds_weights_alpha` is a normal training-spec field.
Between stages, the orchestrator (this skill's driver, layered on
`tao-launch-workflow`) rewrites that one field, relaunches SFT from the previous
stage's output checkpoint, and evaluates before advancing (see §"stage
transitions"). On the cosmos-reason VLM half there is no `ds_weights_alpha`; the
equivalent lever is per-dataset `mix_ratio` scheduling, which is a
schema-extension task rather than an existing knob. [design-doc]

### 2. Ramp SFT hyperparameters via a schedule spec

Adopt SONIC's `schedule_dict` serialization format **as-is** (linear/segment,
`@`-path) [design-doc] to drive an *open-loop* ramp of standard SFT
hyperparameters across stages — e.g. LR, warmup, sequence length, or
augmentation strength. This is a scheduler over the existing SFT spec fields; it
does not touch the training loop's math.

### 3. Stage transitions gated on `eval_fn`

Rather than open-loop `common_step_counter` milestones, gate each stage
transition on an evaluation callback: run `eval_fn` at the end of a stage, read
a metric, and only advance (or mutate the schedule) when a threshold/plateau
condition is met. This is the **closed-loop** curriculum. See the caveats
section — the default GR00T/cosmos-reason eval scripts do **not** emit
`success_rate`, so the gating metric must be an existing metric (accuracy,
BERTScore-F1) or a bespoke verifier.

## Loss-weighting seam [verified]

The exact seam to inject a **per-sample** difficulty / regret / verifier loss
weight (a finer-grained curriculum than the dataset-level `alpha`).

**Seam:** `external/Isaac-GR00T/gr00t/experiment/trainer.py:254`

```python
# trainer.py:254–275 (verified this session), Gr00tTrainer.compute_loss override
def compute_loss(
    self,
    model,
    inputs,
    return_outputs: bool = False,
    num_items_in_batch: int | None = None,
):  # type: ignore[override]
    ...
    loss, outputs = super().compute_loss(
        model,
        inputs,
        return_outputs=True,
        num_items_in_batch=num_items_in_batch,
    )
    # <-- inject here: scale/reweight `loss` (or per-sample loss terms in
    #     `outputs`) by a difficulty/verifier weight derived from `inputs`
    #     before returning.
```

Because `Gr00tTrainer` already overrides `compute_loss` (to log token accuracy),
this is the natural place to multiply the loss by a per-sample weight — hard or
low-confidence samples up-weighted, easy ones down-weighted. The weight source
can be a precomputed difficulty score attached to each sample, or (later) a
verifier signal. Injecting a weight here is a curriculum mechanism and **does
not** by itself constitute RL — there is still no reward-driven action sampling.

> Constraint: `external/` is a pinned, read-only submodule. This skill documents
> *where* the seam is; it does not patch GR00T. A real integration ships the
> weighting either via a supported config hook or in TAO-side code that
> subclasses the trainer — never by editing the pinned submodule.

## RL mode (BLOCKED, gated) [verified]

**RL is not available today.** State this plainly to any user who asks to "run
RL in TAO": it cannot be launched with the current skill bank.

**The lock is at the schema layer.**
`skills/models/tao-finetune-cosmos-reason/schemas/train.schema.json:1139–1144`:

```json
"type": {
  "default": "sft",
  "description": "Type of policy.",
  "enum": [
    "sft"
  ],
```

`train.train_policy.type` is an enum with **exactly one** allowed value:
`"sft"`. There is no `"rl"`/`"ppo"`/`"grpo"` option to select. RL is *latent* in
Cosmos-RL (e.g. `dp_shard`/`dp_replicate` sharding knobs and a `val/reward_avg`
metric are listed but inert) — "listed-but-latent," not runnable.

**Turning RL on requires all three, none of which exist today:**

1. **Unlock the enum** — add an RL policy type to
   `train.train_policy.type` (a schema change on the TAO side).
2. **Supply a reward verifier** — the missing *engine*. Most concrete feasible
   spike: a rule/verifier reward over grounded/structured QA (multiple-choice,
   numeric match, IoU, referring-expression match) built from TAO's existing
   data skills (`tao-generate-referring-expressions`,
   `tao-generate-image-grounding`,
   `tao-generate-video-reasoning-annotations`) — i.e. standard RLVR.
   BERTScore-F1 is a noisier dense fallback; a learned reward model is a higher
   research spike. [design-doc]
3. **A rollout loop** — sample → score → update, the RL control flow that SFT
   does not have.

Until (1)+(2)+(3) land, this skill exposes RL only as a documented, blocked
target — never as a launchable action. Do not imply RL works today.

## Agentic loop mapping [design-doc]

*Design-doc level, from `docs/design/07-review-and-revised-roadmap.md` §5.* Maps
the curriculum onto TAO's `autoresearch` AutoML machinery so stage progression
is agent-driven rather than open-loop:

- **`SpecPrescreener`** → the "no-PPO pre-screen": cheaply reject bad specs
  before spending a training stage on them.
- **`LLMAnalyzer(narrow_ranges=True)`** → tightens the search space every ~5
  recommendations — the mechanism that *narrows* the curriculum as it
  progresses.
- **`ResearchProgram` phases with `carry_forward="best"`** → literally encode a
  staged curriculum: each phase carries the best config forward as the next
  stage's starting point.
- **`eval_fn → success_rate` drives stage transitions** (plateau detection)
  instead of open-loop `common_step_counter` milestones.

This reuses the AutoML/autoresearch stack (see
`skills/applications/tao-run-automl/SKILL.md`) as the curriculum controller. It
is a proposal, not implemented, and inherits the `success_rate` caveat below.

## Honest caveats (non-negotiable)

- **No `success_rate` in default eval.** No runnable GR00T/cosmos-reason eval
  script emits `success_rate` by default. Any `eval_fn`-gated stage transition
  or agentic `success_rate` signal must use an existing metric (accuracy,
  BERTScore-F1) or a **bespoke verifier** that you build and validate. Do not
  assume `success_rate` exists.
- **Compute dominates.** A single SONIC-scale PPO trial is thousands of
  GPU-hours. The claim that agentic curriculum overhead beats well-tuned static
  defaults is **unquantified** and needs a proxy-scale correlation study before
  any large spend. Curriculum-over-SFT is not free either — each stage is a full
  SFT run.
- **Two distinct curricula — keep them separate.** SONIC's *online sampler*
  (a data curriculum: per-example difficulty reweighting during sampling) and
  the *open-loop scheduler* (`schedule_dict` over hyperparameters) are different
  knob families with different failure modes. Expose them as separate controls;
  do not conflate "reweight the data" with "ramp the LR."
- **Eval/train threshold decoupling.** An agent may tune strict *training*
  thresholds while `eval_fn` measures relaxed ones — a partly-decoupled
  objective. Watch for this when interpreting stage-transition decisions.
- **`external/` is read-only.** All GR00T seams cited here live in a pinned
  submodule. Integrations attach via TAO-side config/subclassing, never by
  editing pinned source.

## References

- `references/seams.md` — exact verified line cites for every source above.
- `docs/design/07-review-and-revised-roadmap.md` §5 — originating design.
- `skills/core/tao-launch-workflow/SKILL.md` — launch gate this skill's driver
  would layer on.
- `skills/applications/tao-run-automl/SKILL.md` — autoresearch/AutoML stack for
  the agentic-loop mapping.
