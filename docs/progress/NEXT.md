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

## PARALLEL BUILD TRACK (launch-ready patches for a future cluster)
Lower priority than ①②, but keeps the box productive. Same proven loop: **verify mechanism in source → write patch+config → CPU static-validate → keep submodule pinned.**
1. **Stage 3 — progressive domain randomization.** Verify the `force_push_linear_curriculum` slot wiring (`modular_tracking_env_cfg.py`, `push_robot.yaml`); build a `push_scale_curriculum` modify_fn (0.3→1.0) + `dr_ramp.yaml`. Static-validate like Stage 2.
2. **Stage 1 — PLR/regret sampler.** Verify per-step bin-id registration in `RolloutStorage` (`register_key`); build a `MotionLibBase` subclass swapping score to `(1-ρ)·regret + ρ·staleness` + a unit test of the scoring math.

## NEW TRACK — reverse transfer (GR00T → TAO), see §5 of the review doc
Feasible-now, no cluster needed:
1. **`tao-curriculum-rl` workflow skill (scaffold).** Curriculum-over-SFT using verified seams `data.ds_weights_alpha` (`factory.py:78–85`) + `Gr00tTrainer.compute_loss` override (`trainer.py:254–275`); adopt SONIC's `schedule_dict` `@`-path format; stage transitions via `autoresearch`/`eval_fn`. RL mode reported *blocked* until a verifier reward exists + `train.train_policy.type` enum is unlocked (`train.schema.json:1139–1147`).
2. **VLM verifier spike (RLVR).** Rule verifier over grounded/structured QA using existing TAO data skills (`tao-generate-referring-expressions`, `-image-grounding`, `-video-reasoning-annotations`). This is the one genuinely missing piece — the reward.

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
