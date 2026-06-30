# GR00T × TAO — Agentic RL & Curriculum Learning for Whole-Body Control

**A research design plan bridging NVIDIA TAO's agentic post-training stack and GR00T whole-body humanoid control.**

This repository is a fork of [`NVIDIA-TAO/tao-skills-bank`](https://github.com/NVIDIA-TAO/tao-skills-bank) combined (via git submodules under [`external/`](../../external)) with [`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl) and [`NVIDIA/Isaac-GR00T`](https://github.com/NVIDIA/Isaac-GR00T). It adds a detailed, file-grounded design plan for curriculum-learning and "agentic RL skills" on the whole-body motion-tracking RL stack.

> **Provenance & honesty.** Every claim below was derived by reading the actual source in the pinned submodule commits (WBC `0e35637`, Isaac-GR00T `ab88b50`) and the TAO skill bank, then passed through an adversarial critique that caught and corrected several hallucinated line-numbers and one fabricated metric. Where a mechanism is **unverified** or **infeasible today**, it is labeled as such — those labels are load-bearing, not hedging.

## The three repos and how they connect

| Repo | Role | Key fact |
|---|---|---|
| **Isaac-GR00T** (N1.7 VLA) | High-level "brain" | VLM backbone = `Qwen3VLForConditionalGeneration.from_pretrained("nvidia/Cosmos-Reason2-2B")` → flow-matching DiT action head. Emits `motion_token`s under the `UNITREE_G1_SONIC` embodiment tag. |
| **GR00T-WholeBodyControl** (GEAR-SONIC) | Low-level controller | `TRLPPOTrainer(PPOTrainer)` — **subclasses HuggingFace TRL** (the LLM-RLHF library) — trains a motion-tracking PPO policy in IsaacLab. Decodes GR00T's `motion_token`s into 29-DoF joint commands. |
| **tao-skills-bank** (`tao-finetune-cosmos-reason`) | Agentic post-training tooling | LoRA-SFTs the **same Cosmos-Reason / Qwen3-VL family** (verified base: Cosmos3-Nano). Ships an agentic AutoML layer (`autoresearch`/`pbt`/`eval_fn`) and a failure-driven DEFT data loop. |

**The hierarchical coupling:** GR00T (Cosmos-Reason VLM) emits latent motion tokens → SONIC's TRL-PPO controller decodes/tracks them into whole-body joint commands. A single policy yields language-conditioned, coordinated locomotion + manipulation.

## Two verified anchor facts (the whole plan rests on these)

1. **GR00T N1.7's VLM half *is* `nvidia/Cosmos-Reason2-2B` (Qwen3-VL)** — `external/Isaac-GR00T/gr00t/model/modules/qwen3_backbone.py`. This is the same model family TAO's `tao-finetune-cosmos-reason` skill post-trains (its verified base is `Cosmos3-Nano`).
2. **SONIC's RL core subclasses HuggingFace TRL's `PPOTrainer`** (`trl==0.28.0`) — `external/GR00T-WholeBodyControl/gear_sonic/trl/trainer/ppo_trainer.py:321`. The LLM-RL and robot-RL toolchains literally converge on the same library.

## Document map

- [`01-how-groot-works.md`](01-how-groot-works.md) — technical report on SONIC WBC + GR00T VLA.
- [`02-curriculum-rl-plan.md`](02-curriculum-rl-plan.md) — the 6-stage curriculum-learning plan for whole-body motion-tracking RL.
- [`03-agentic-rl-plan.md`](03-agentic-rl-plan.md) — the two-track "agentic RL skills" plan.
- [`04-tao-cross-pollination.md`](04-tao-cross-pollination.md) — what transfers between TAO's LoRA/Cosmos/agentic stack and GR00T (both directions).
- [`05-flagship-experiment.md`](05-flagship-experiment.md) — the concrete first experiment, runnable on this stack.
- [`06-risks-and-open-questions.md`](06-risks-and-open-questions.md) — verified risks, gaps, and open questions.

## Licensing & attribution

This repo inherits the TAO skill bank's Apache-2.0 license. The GR00T repos are included **by reference** as submodules (not copied), preserving NVIDIA's Apache-2.0 code license and NVIDIA Open Model License on weights. No model checkpoints or LFS assets are vendored here.
