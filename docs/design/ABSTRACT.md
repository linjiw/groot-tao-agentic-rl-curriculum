# Abstract & Introduction

## Abstract

Humanoid whole-body control and vision-language-action (VLA) modeling are usually
studied as separate problems with separate toolchains. We show that two NVIDIA
stacks — the **Isaac-GR00T N1.7 VLA** and the **GR00T-WholeBodyControl (GEAR-SONIC)**
motion-tracking controller — in fact converge on a shared substrate, and we exploit
that convergence to design a curriculum-learning and "agentic RL" program for
whole-body humanoid control. Two facts, both verified by reading pinned source,
anchor the work: (1) GR00T N1.7's VLM backbone *is* `nvidia/Cosmos-Reason2-2B`
(Qwen3-VL) — the same model family that NVIDIA TAO's `tao-finetune-cosmos-reason`
skill post-trains; and (2) SONIC's reinforcement-learning core *subclasses
HuggingFace TRL's `PPOTrainer`* (`trl==0.28.0`), so the LLM-RLHF and robot-RL
toolchains literally share a library. Building on these, we contribute a
file-grounded technical report of the SONIC + GR00T stack; a six-stage curriculum
plan that *formalizes and extends* the roughly 80% of an automatic curriculum SONIC
already implements but ships disabled; a two-track agentic-RL plan (a VLA composing
quantized skill tokens, and an LLM agent that designs the RL loop itself via TAO's
AutoML layer); and a bidirectional TAO↔GR00T cross-pollination analysis. We
instantiate the plan in a flagship experiment — a progressive termination-threshold
curriculum — chosen because it is fully runnable on the existing stack with no new
training infrastructure, no LLM/AutoML adapter, and no representation-mismatch risk.
Its enabling mechanism is verified in source: IsaacLab's `TerminationManager` reads
term parameters fresh every step, so a scheduled `modify_term_cfg` write propagates
at episode boundaries. Throughout, we label unverified mechanisms and
infeasible-today components explicitly — most notably the continuous-vs-quantized
representation mismatch between GR00T's flow-matching action tokens and SONIC's
FSQ-quantized decoder, and the substantial adapter work required to drive SONIC
through TAO's SDK-bound AutoML loop. The flagship experiment is coded, statically
validated, and launch-ready; empirical training results await a multi-GPU
Isaac-Lab host.

## 1. Introduction

Recent humanoid robots are increasingly driven by two cooperating models: a
high-level **vision-language-action (VLA)** policy that maps instructions and
observations to abstract action intents, and a low-level **whole-body controller**
that turns those intents into coordinated joint commands for locomotion and
manipulation. NVIDIA's GR00T program ships both halves. **Isaac-GR00T N1.7** is the
VLA "brain": a Cosmos-Reason vision-language backbone feeding a flow-matching
diffusion-transformer action head that, under the `UNITREE_G1_SONIC` embodiment tag,
emits latent `motion_token`s. **GR00T-WholeBodyControl**, via its **GEAR-SONIC**
controller, is the low-level half: a single motion-tracking foundation policy —
trained with PPO in IsaacLab — that decodes those tokens into 29-DoF joint commands
spanning walking, crawling, and teleoperation.

Separately, **NVIDIA TAO** provides an agentic post-training stack. Its
`tao-finetune-cosmos-reason` skill LoRA-SFTs the Cosmos-Reason / Qwen3-VL family,
and its AutoML layer (`autoresearch`, `pbt`, an LLM analyzer, and a failure-driven
DEFT data loop) automates hyperparameter search and data curation. At first glance
these three repositories — a VLA, a robot RL controller, and an LLM post-training
toolkit — occupy different worlds.

**Our central observation is that they do not.** Reading the pinned source rather
than relying on documentation, we establish two anchor facts on which the entire
project rests. First, GR00T N1.7's VLM half is instantiated directly from
`nvidia/Cosmos-Reason2-2B` (Qwen3-VL) in `qwen3_backbone.py` — the same family TAO's
post-training skill targets (its own verified base is `Cosmos3-Nano`). Second,
SONIC's RL trainer is `class TRLPPOTrainer(PPOTrainer)`, subclassing the *real*
HuggingFace TRL package used for LLM RLHF. The high-level brain and the low-level
controller are therefore not merely compatible in spirit; they share a model family
and an RL library. This convergence is what makes cross-pollination between TAO's
LLM-centric agentic tooling and GR00T's robot-centric control concrete rather than
aspirational.

From this foundation we pursue two research directions. The first is **curriculum
learning for whole-body motion-tracking RL**. Our analysis of SONIC shows it already
contains most of the machinery for an automatic curriculum — a PHC-style
failure-weighted motion sampler, adaptive termination thresholds, staged event
presets, and IsaacLab's `CurriculumManager` — but the default experiment config
disables most of these hooks (it selects `curriculum/empty.yaml`). Rather than build
greenfield, we *formalize and extend* this existing 80% into a six-stage plan, every
stage gated on SONIC's shipped `im_eval` metrics. The second direction is **agentic
RL**, split into two non-competing tracks that share the FSQ universal-token
interface and the TRL-PPO substrate: a VLA-as-agent that composes a frozen library
of learned motor skills, and an agent-as-designer that ports TAO's AutoML loop to
design the RL itself — proposing motion subsets, reward weightings, and schedules,
then closing the loop on `success_rate`.

We are deliberate about honesty. Every mechanism here was derived by reading source
at pinned commits (WBC `0e35637`, Isaac-GR00T `ab88b50`) and subjected to an
adversarial critique that caught hallucinated line numbers and one fabricated
metric. Where a mechanism is unverified or infeasible today, we label it as such,
and those labels are load-bearing. The most important is a **representation
mismatch**: GR00T regresses *continuous* motion tokens via flow matching, while
SONIC's decoder was trained on *FSQ-quantized* tokens on a discrete lattice, and the
existing magnitude guard rejects outliers, not off-lattice points. We also flag that
TAO's `AutoMLRunner` is bound to its SDK job path and is *not* reusable as-is for
SONIC's `accelerate launch` + IsaacSim workflow, and that the practical dominant
constraint is compute — a single PPO trial is thousands of GPU-hours.

To make progress tractable and high-signal, we designate a **flagship experiment**:
a progressive termination-threshold curriculum. It is the cheapest fully runnable
test on the existing stack — pure configuration plus a small dataclass extension,
with no new training infrastructure, no LLM/AutoML adapter, and no
representation-mismatch exposure. Its enabling assumption was the first thing we
verified: IsaacLab's `TerminationManager.compute()` reads term parameters fresh on
every call, so a scheduled `modify_term_cfg` write takes effect at episode
boundaries. The hypothesis is that tightening tracking thresholds on a schedule lets
the policy first learn coarse tracking on long, survivable episodes and then refine
to strict precision — reaching the same final precision as a fixed-strict baseline
but with faster, more stable early learning. The experiment is coded, statically
validated, and launch-ready; the sole remaining blocker to empirical results is
access to a multi-GPU Isaac-Lab host.

**Contributions.**
1. A file-grounded technical report of the GR00T VLA + SONIC WBC stack and the two
   verified facts (shared Cosmos-Reason backbone; shared TRL-PPO core) that bridge
   it to TAO's agentic post-training tooling.
2. A six-stage curriculum-learning plan that formalizes and extends SONIC's existing
   but largely disabled auto-curriculum machinery, each stage gated on shipped
   `im_eval` metrics.
3. A two-track agentic-RL plan (VLA-as-agent skill composition; agent-as-designer
   AutoML/curriculum/reward search) with an honest scoping of the adapter work TAO
   integration requires.
4. A bidirectional TAO↔GR00T cross-pollination analysis, including SONIC as a
   working reference for TAO's latent RL mode and IsaacLab's curriculum manager as a
   template for a first-class TAO curriculum skill.
5. A launch-ready flagship experiment (progressive termination-threshold curriculum)
   with its enabling mechanism verified in source, plus an explicit catalog of
   risks, unverified assumptions, and infeasible-today components.
