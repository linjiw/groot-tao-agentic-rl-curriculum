# Next-Step Plan (start here)

Last updated: 2026-07-01 (revised after the multi-agent review — see [`../design/07-review-and-revised-roadmap.md`](../design/07-review-and-revised-roadmap.md)).

**State:** Stage-2 curriculum is coded + statically validated (patch **verified to apply clean**, 6/6 checks reproduce). Nothing has been trained. This box is a **single A10G (23 GB), no Isaac Lab** — so the flagship (needs 64+ GPUs) cannot run here.

**Key decision from the review:** *Do NOT make Stage-2 the first experiment.* It optimizes for a cluster we don't have over the two cheapest **project-killing** questions — both of which run on the A10G today. **Reordered below.**

## 0. Resume checklist (5 min)
```bash
cd /home/ec2-user/work/groot-tao-agentic-rl-curriculum
git pull
git submodule update --init --recursive    # if submodule trees aren't populated
git log --oneline -3                        # expect 13dacc7 (or later) at top
git submodule status                        # expect WBC 0e35637, Isaac-GR00T ab88b50
```
Then re-read this file + `../design/07-review-and-revised-roadmap.md`.

---

## PRIORITY ORDER (revised)

The next two experiments run **on this A10G** and answer the questions that could invalidate whole tracks. Stage-2 is demoted to "when a cluster exists."

### ① Off-lattice FSQ decode-error measurement  *[A10G, ~hours]* — DO FIRST
Quantifies the repo's own **#1 risk** (continuous-vs-FSQ mismatch, `06:7`) and answers Open-Q #4 (`06:38`). No PPO, no 4096 envs, no IsaacLab.
- Load the SONIC FSQ decoder (flat-64 continuous ONNX deploy path, `03:14`).
- Feed (a) teacher FSQ-**quantized** tokens and (b) perturbed/**continuous** off-lattice tokens.
- Measure tracking/action degradation vs a **snap-to-lattice** projection.
- Files: `gear_sonic/scripts/run_vla_inference.py` (the 1.25 magnitude guard), `universal_token_modules.py`.
- **Why first:** if degradation is catastrophic and snap-to-lattice doesn't recover it, the whole VLA-as-agent track needs quantization-aware BC *before anything else* — a roadmap-changing result, learned cheaply.

### ② Cosmos-Reason2-2B ↔ GR00T strict-loader + LoRA-attach spike  *[A10G]* — DO SECOND
Prerequisite spike for the **entire** TAO→GR00T bridge (settles Open-Q #2/#5). 2B bf16 (~4–5 GB) fits in 23 GB for load/validate.
1. Run `prepare_cosmos3_vlm_checkpoint.py` **overriding** its default `--vlm-model-name Qwen/Qwen3-VL-8B-Instruct` → the 2B target. Check output `config.model_type==qwen3_vl` + shard/tokenizer completeness (script `:36–79`).
2. Load converted (+`merge_and_unload`ed) dir through GR00T `setup.py`; confirm **zero** missing/unexpected/mismatched keys (loader raises otherwise, `:106–120`; only `action_head.mask_token` whitelisted).
3. Attach `LoraConfig(r=8, target_modules=['q_proj','v_proj'])` on the kept 12 layers **after** the truncation pop (`qwen3_backbone.py:159`); test the `merge_and_unload`→re-validate path.
- Mirror TAO's shipped LoRA hyperparams (`spec_template_train.yaml:69–80`).
- **Why second:** a "no" here is load-bearing for C1/P3/P4 and is cheap to get.

### ③ Stage-2 curriculum vs baseline  *[ONLY once a 64+-GPU IsaacLab cluster exists]*
Still the flagship — demoted to "when cluster exists." **Before** burning GPU:
- **(F3)** Re-scale `threshold_tighten.yaml` milestones — `common_step_counter` counts across all envs, so `[0,30000,80000]` fires almost instantly at 4096 envs. Set as fractions of total env-steps (`num_learning_iterations × num_steps_per_env × num_envs`).
- **Cheap mechanism check first** (`RUN.md:23–28`): log the live `anchor_pos` threshold stepping 0.30 → 0.22 → 0.15 at milestones. If it never changes, the address shorthand is wrong — fix before spending GPU.
- Apply: `bash experiments/stage2-termination-curriculum/apply_and_validate.sh external/GR00T-WholeBodyControl` (idempotent).
- Reduced-scale smoke (`num_envs=256`) curriculum (`manager_env/curriculum=threshold_tighten`) vs baseline (`=empty`) before full scale.
- Read `im_eval`: longer early episodes + faster/smoother early MPJPE drop, with **final `success_rate` matching** the fixed-strict baseline. Record in a new `docs/progress/<date>.md` + `experiments/.../RESULTS.md`.

---

## NEW DESIGN — Curriculum-Manager Agent (2026-07-01)
`../design/08-curriculum-manager-agent.md` — the project's organizing design: an LLM agent
supervising a live SONIC run at checkpoint cadence, tuning meta-parameters of SONIC's
existing adaptive controllers (sampler floor/cap, competence-gated threshold stepping,
DR ramp, `schedule_dict`) via bounded schema-validated deltas, with a held-out protected
metric + rollback tripwires. Literature-grounded (no published system does mid-run LLM
supervision — the gap is real). Stage-1 PLR is demoted (no legged-robot evidence for
regret/value-loss signals; LP-ACRL beat PLR); Stage-2/3/4 mechanisms are absorbed as
the agent's Family-B/A knobs.

**Phase 0: ✅ DONE (2026-07-01, same session).** 50/50 new tests, repo CPU suite 93/93
[measured]. Built + verified:
- `skills/agentic/sonic-knob-registry/` — action space as data (`registry.yaml`, every
  knob source-cited against WBC 0e35637) + static decision validator (whitelist, hard
  range, max step, per-knob cooldown, one-atomic-change, required tripwire). 22 tests.
- `skills/agentic/sonic-run-digest/` — JSONL (train/eval/sampler, SONIC's own metric
  names) → trend-annotated `digest.json`; failure-vector entropy + cap-saturation use
  the sampler's exact cap semantics (`motion_lib_base.py:2570–2577`). 13 tests.
- `experiments/curriculum-manager-phase0/` — replay harness running the full tick loop
  (digest→decide→validate→apply→tripwire→journal) with `BandStepperPolicy` (deterministic
  playbook core) as LLM stand-in. Acceptance behaviors ALL verified: thrash→0 actions,
  healthy→1 bounded tighten, plateau→sampler floor (never threshold), regression→tripwire
  auto-rollback, no-heldout-metric→no action, rogue policy→fully rejected. 15 tests.
  See `experiments/curriculum-manager-phase0/RESULTS.md`.

**Playbook + watcher + Phase 1: ✅ ALL DONE (2026-07-01, same session).** Repo CPU suite
120/120 [measured].
- `skills/agentic/sonic-curriculum-manager/` — the LLM-facing playbook: hard rules, tick
  procedure, priority-ordered decision table, digest traps, exact decision format.
- `skills/agentic/sonic-heldout-watcher/` — protected-metric producer: salted hash split
  (stable under library growth), integrity-checked manifest the manager never reads,
  `metrics_eval.json` → `heldout_success_rate` records with foreign-key refusal; live
  wiring documented against the verified `filter_motion_keys` seam
  (`eval_agent_trl.py:316–318`). 13 tests.
- `experiments/curriculum-manager-phase1/` — **real LLM in the closed loop**: knob-
  responsive toy tracking run (SONIC-shaped sampler floor/cap, threshold pressure,
  true held-out subset) + `LLMPolicy` shelling to `claude -p` with the playbook.
  Measured: LLM held `none` ×6 on a low-band-but-rising run (correctly citing the
  sustain + trend rules) and executed the full tighten→cooldown-hold→tighten sequence
  on an in-band run; 0 validator rejections needed. Phase 1 also caught a real playbook
  bug (row-1 contraction precondition) — fixed. 14 tests. See its RESULTS.md.

**Phase 2 (needs IsaacLab cluster):** `sonic-job-adapter` skill (launch/checkpoint/
rollback/eval-watcher lifecycle around `accelerate launch train_agent_trl.py`), live
digest streams from wandb/stdout, held-out watcher wired via `filter_motion_keys`,
then the 256-env smoke: manager ON vs OFF vs hand-schedule (doc 08 §8).

## PARALLEL BUILD TRACK (launch-ready patches for a future cluster)
Lower priority than ①②, but keeps the box productive. Same proven loop: **verify mechanism in source → write patch+config → CPU static-validate → keep submodule pinned.**
1. **Stage 3 — progressive domain randomization.** Verify the `force_push_linear_curriculum` slot wiring (`modular_tracking_env_cfg.py`, `push_robot.yaml`); build a `push_scale_curriculum` modify_fn (0.3→1.0) + `dr_ramp.yaml`. Static-validate like Stage 2.
2. **Stage 1 — PLR/regret sampler.** Verify per-step bin-id registration in `RolloutStorage` (`register_key`); build a `MotionLibBase` subclass swapping score to `(1-ρ)·regret + ρ·staleness` + a unit test of the scoring math.

## NEW TRACK — reverse transfer (GR00T → TAO), see §5 of the review doc
Feasible-now, no cluster needed. **Status: first three pieces DONE + verified this session** (Exp③ token-lattice was NO-GO — see below — which is *why* this track was prioritized).
1. ✅ **`tao-curriculum-rl` workflow skill (scaffold)** — landed at `skills/agentic/tao-curriculum-rl/SKILL.md` (+ `references/seams.md`). Curriculum-over-SFT using verified seams `data.ds_weights_alpha` (`factory.py:78`) + `Gr00tTrainer.compute_loss` override (`trainer.py:254`); RL mode reported *blocked* until a verifier reward exists + `train.train_policy.type` enum unlocked (`train.schema.json:1139–1144`, enum==`["sft"]` **[verified]**).
2. ✅ **KL-adaptive-LR + `schedule_dict` port** — `experiments/reverse-transfer-lr-curriculum/` (`kl_adaptive_lr.py`, `curriculum_schedule.py`, tests). Faithful port of SONIC `_adjust_learning_rate_based_on_kl` (`ppo_trainer.py:2142–2166`) + `update_scheduled_params` (`gear_sonic/trl/utils/scheduler.py:296`). **19/19 pytest passing [measured, re-run by parent].**
3. ✅ **VLM verifier spike (RLVR)** — `experiments/rlvr-verifier-reward/` (`verifiers.py`: MC/numeric/IoU/ref-exp; `rlvr_demo.py` toy REINFORCE; tests). **24/24 pytest passing; demo reward curves improve** (MC 0.438→0.991, numeric 0.090→0.728) **[measured, re-run by parent]**. This was the one genuinely missing piece — the reward — now shown concretely implementable on CPU.

### Exp③ verdict — trained-VLA token-lattice distance: **NO-GO (external dep)**
`experiments/vla-token-lattice-distance/` + `experiments/sonic-teacher-token-reference/`. The core measurement is **blocked locally**: no sonic-finetuned GR00T VLA exists on this box (only generic `GR00T-N1.7-3B` with a 132-d flow-matching head — `unitree_g1_sonic` absent from all 4 checkpoint surfaces; network gated 401). Teacher tokens confirmed **exactly on-lattice** (0.0 steps, 3,606 real tokens). Combined with Exp① (snap recovers within ±½ step), the open question — *does a trained sonic VLA's emission drift exceed ½ step?* — is answerable **only with a sonic-finetuned checkpoint we don't have.** Recorded as an external dependency, not a local task.

---

## Housekeeping (do now, ~15 min)
- **(F1) Pin IsaacLab.** The Step-0 proof cites `external/IsaacLab/.../termination_manager.py` but that path **doesn't exist** (real copy `/workspace/IsaacLab`, captured nowhere). Pin it (submodule or recorded commit) + fix the citation, or the load-bearing proof silently rots.
- **(F2) Fix `select_layer` wording.** Source default is `-1` (`qwen3_backbone.py:110`); 12 is set in `gr00t_n1d7.py:47`. Docs say "kept 12" as if intrinsic — reword to "config sets 12."
- **CI:** add a GitHub Action running `apply_and_validate.sh` against the pinned submodule (only real rot risk is upstream WBC drift).
- **Doc map:** `ABSTRACT.md` + `07-review-and-revised-roadmap.md` are now wired into `00-overview.md`.
- **Repo visibility/fork decision:** still open — keep-public-fork vs private vs detach. `gh repo edit linjiw/groot-tao-agentic-rl-curriculum --visibility private` if going private.

## Quick reference — the two facts the whole project rests on
1. GR00T N1.7 VLM backbone **is `nvidia/Cosmos-Reason2-2B` (Qwen3-VL)** — `qwen3_backbone.py:107/151`. Same family TAO's `tao-finetune-cosmos-reason` LoRA-SFTs (verified base `Cosmos3-Nano`; conversion helper defaults to `Qwen3-VL-8B-Instruct` — 2B compat = spike ②).
2. SONIC's RL trainer **subclasses HuggingFace TRL `PPOTrainer`** — `ppo_trainer.py:321`, `trl==0.28.0`.

## Known traps (don't relearn these)
- Don't edit/commit the submodule tree — keep it pinned; ship changes as patches in our repo.
- `modify_term_cfg` is **IsaacLab's**, not `gear_sonic`'s; address shorthand `terminations.` → `termination_manager.cfg.`.
- `open_loop_eval.py` reports **MSE/MAE**, not `success_rate` (that's `im_eval` / manual robot trials). **No `success_rate` in any runnable GR00T/cosmos-reason eval by default.**
- GR00T tokens are **continuous (flow-matching)**; SONIC decoder expects **FSQ-quantized** — representation mismatch, not distribution shift. This is experiment ①.
- `num_steps` milestones are in **env-steps across all envs** — re-scale for `num_envs` (F3).
- TAO ships a **first-class LoRA path already** (`spec_template_train.yaml:69–80`, `enable_lora` across actions) — C1 is "adapt a working stack," not "port an idea." But it's SFT-locked at the schema; RL is latent in Cosmos-RL.
- SONIC gives TAO the RL **chassis** (GAE/PPO/KL-LR/sampler/`schedule_dict`) but **not the reward** — that's net-new (a verifier).
