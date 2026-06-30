# GR00T × TAO — Agentic RL & Curriculum Learning for Whole-Body Control

[![License](https://img.shields.io/badge/License-Apache%202.0-76B900.svg)](LICENSE)

A research-design repository that **combines** NVIDIA's [TAO skill bank](https://github.com/NVIDIA-TAO/tao-skills-bank) (this repo's fork base) with NVIDIA's GR00T whole-body-control stack, and adds a detailed, file-grounded plan for **curriculum learning** and **"agentic RL skills"** on humanoid whole-body motion-tracking RL.

> This is a personal research fork by [@linjiw](https://github.com/linjiw). It is **not** an official NVIDIA project. It vendors no model weights and no NVIDIA source — the GR00T repos are included by reference as git submodules.

## What's here

> 📌 **Picking up where we left off?** Start at [`docs/progress/NEXT.md`](docs/progress/NEXT.md) for the rolling next-step plan, and [`docs/progress/`](docs/progress) for dated logs.

```
.
├── docs/progress/            ← daily logs + NEXT.md (resume here)
├── docs/design/              ← THE DESIGN PLAN (start at 00-overview.md)
│   ├── 00-overview.md
│   ├── 01-how-groot-works.md
│   ├── 02-curriculum-rl-plan.md
│   ├── 03-agentic-rl-plan.md
│   ├── 04-tao-cross-pollination.md
│   ├── 05-flagship-experiment.md
│   └── 06-risks-and-open-questions.md
├── external/                 ← GR00T repos (git submodules, by reference)
│   ├── GR00T-WholeBodyControl/   (NVlabs, pinned @ 0e35637)
│   └── Isaac-GR00T/              (NVIDIA, pinned @ ab88b50)
├── skills/ …                 ← inherited TAO skill bank (the fork base)
└── README.tao-upstream.md    ← original TAO skill bank README
```

## The idea in one paragraph

GR00T N1.7's VLM backbone **is `nvidia/Cosmos-Reason2-2B` (Qwen3-VL)** — the same model family TAO's `tao-finetune-cosmos-reason` skill LoRA-SFTs. GR00T's whole-body controller, **GEAR-SONIC**, trains motion-tracking with a `PPOTrainer` that **subclasses HuggingFace TRL** — the same library used for LLM RLHF. So the LLM-post-training toolchain and the humanoid-RL toolchain literally share a substrate. This repo maps that bridge and lays out (1) a 6-stage curriculum plan that formalizes SONIC's already-present-but-disabled adaptive sampling and curriculum hooks, (2) a two-track "agentic RL" plan (VLA-as-agent + LLM-as-curriculum/reward-designer), and (3) what transfers between TAO's LoRA/Cosmos/agentic-AutoML/DEFT tooling and GR00T — in both directions.

## Quick start

```bash
git clone --recurse-submodules git@github.com:linjiw/groot-tao-agentic-rl-curriculum.git
# or, if already cloned:
git submodule update --init --recursive

# read the plan
$EDITOR docs/design/00-overview.md
```

> Submodules are pinned to the exact commits the design analysis ran against, so every `file:line` citation stays valid. The GR00T submodules require Git LFS + Isaac Lab to actually *run*; see their own READMEs.

## Flagship first experiment

[`docs/design/05-flagship-experiment.md`](docs/design/05-flagship-experiment.md) — a **progressive termination-threshold curriculum** for SONIC: pure config + a small `CurriculumCfg` extension, no new training infra, no LLM/AutoML adapter, runnable with the stock `accelerate launch gear_sonic/train_agent_trl.py +exp=...sonic_release` invocation.

## Honesty note

Every claim was derived by reading the pinned source and then passed through an adversarial critique that corrected hallucinated line-numbers and one fabricated metric. Sections flagged **unverified** or **not feasible today** (e.g. continuous-vs-FSQ token mismatch, MotionBricks targeted generation, whether the TAO skill accepts the Cosmos-Reason2-2B base) carry those labels deliberately — see [`docs/design/06-risks-and-open-questions.md`](docs/design/06-risks-and-open-questions.md).

## Attribution & license

- Fork base: [`NVIDIA-TAO/tao-skills-bank`](https://github.com/NVIDIA-TAO/tao-skills-bank) (Apache-2.0) — license retained in [`LICENSE`](LICENSE).
- Submodules: [`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl) and [`NVIDIA/Isaac-GR00T`](https://github.com/NVIDIA/Isaac-GR00T) — Apache-2.0 code; model weights under the NVIDIA Open Model License. Included by reference only.
- Design documents under `docs/design/` © 2026 @linjiw, Apache-2.0.
