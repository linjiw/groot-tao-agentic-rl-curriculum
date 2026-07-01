# Verified seams — exact line cites

All line numbers re-opened and confirmed this session against pinned source.
`external/` is a read-only submodule; cites are for reference, not for editing.

## Curriculum seam #1 — data-mixture alpha [verified]

- **File:** `external/Isaac-GR00T/gr00t/data/dataset/factory.py`
- **Line 78:** `alpha = self.config.data.ds_weights_alpha`
- **Lines 78–85:** guarded block reweighting datasets by `length^alpha`
  (normalized to `ds_lengths[0]`); prints that it **overrides** per-dataset
  `mix_ratio`.
- **Line 74 (context):** `weight = relative_length * dataset_spec.mix_ratio`
  — the mix_ratio path that alpha supersedes.
- **Use:** anneal `alpha` across stages (e.g. `0.0 → 1.0`) for a data
  curriculum. Single-scalar knob. GR00T VLA half only; cosmos-reason VLM half
  needs a `mix_ratio` schema extension (design-doc).

## Curriculum seam #2 — per-sample loss weighting [verified]

- **File:** `external/Isaac-GR00T/gr00t/experiment/trainer.py`
- **Line 254:** `def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):`
  — `Gr00tTrainer.compute_loss` override.
- **Lines 254–275:** delegates to `super().compute_loss(..., return_outputs=True)`
  then logs token accuracy. Inject a per-sample difficulty/regret/verifier loss
  weight after the `super()` call, before returning.
- **Use:** finer-grained (per-sample) curriculum than the dataset-level alpha.
  Still SFT — not RL.

## SFT lock — schema enum [verified]

- **File:** `skills/models/tao-finetune-cosmos-reason/schemas/train.schema.json`
- **Lines 1139–1144:** `train.train_policy.type` is an enum with exactly one
  value:

  ```json
  "type": {
    "default": "sft",
    "description": "Type of policy.",
    "enum": [
      "sft"
    ],
  ```

- **Meaning:** RL cannot be selected. "Add RL to TAO" = (1) unlock this enum +
  (2) supply a reward verifier + (3) add a rollout loop. RL is otherwise latent
  in Cosmos-RL (`dp_shard`/`dp_replicate`, `val/reward_avg` listed-but-inert).

## Design origin [design-doc]

- **File:** `docs/design/07-review-and-revised-roadmap.md`
- **§5, lines 79–104:** "GR00T → TAO: incorporating RL + curriculum *into* TAO
  agentic skills" — the transfer ledger, the two curriculum seams, the SFT-lock
  finding, and the proposed `tao-curriculum-rl` workflow skill.

## House-style reference [verified]

- **File:** `skills/core/tao-launch-workflow/SKILL.md` (frontmatter shape:
  `name`, `description`, `license: Apache-2.0`, `metadata.author`,
  `metadata.version`, `allowed-tools`, `tags`).
- **File:** `skills/applications/tao-run-automl/SKILL.md` (workflow skill with
  `metadata.author: NVIDIA Corporation`, `version: "0.1.0"`, tags incl.
  `workflow`).
