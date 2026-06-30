# Next-Step Plan (start here tomorrow)

Last updated: 2026-06-30 (see [`2026-06-30.md`](2026-06-30.md) for full context).

**State:** Stage-2 curriculum is coded + statically validated + launch-ready. Nothing has been
trained. The gating question is **do we have an Isaac-Lab multi-GPU host?** The plan branches on that.

## 0. Resume checklist (5 min)
```bash
cd /home/ec2-user/work/groot-tao-agentic-rl-curriculum
git pull
git submodule update --init --recursive    # if submodule trees aren't populated
git log --oneline -3                        # expect 918719d at top
git submodule status                        # expect WBC 0e35637, Isaac-GR00T ab88b50
```
Then re-read this file + `2026-06-30.md`.

---

## TRACK A — if a multi-GPU Isaac-Lab host IS available (preferred)

Goal: get the **first real training signal** (Stage-2 curriculum vs baseline).

1. **Stand up the host** (Isaac Lab 2.3.2 + Isaac Sim; `pip install -e "gear_sonic/[training]"`;
   `python download_from_hf.py --training`; convert+filter Bones-SEED motion lib). Verify:
   `python -c "import isaaclab, gear_sonic"`.
2. **Apply Stage-2** into the WBC checkout:
   `bash experiments/stage2-termination-curriculum/apply_and_validate.sh external/GR00T-WholeBodyControl`
   (idempotent; runs CPU static checks again on the host).
3. **Cheap mechanism check FIRST** (RUN.md Step 2): log the live `anchor_pos` threshold every N steps for
   a few hundred steps; confirm it steps 0.30 → 0.22 → 0.15 at the milestones. If it never changes, the
   address/shorthand is wrong — fix before spending real GPU.
4. **Reduced-scale smoke run** before full scale: `num_envs=256`, short
   `++algo.config.num_learning_iterations`, a small motion subset. Run **curriculum** vs **baseline**
   (`manager_env/curriculum=threshold_tighten` vs `=empty`), everything else identical.
   - **Re-scale the milestones**: `common_step_counter` counts across all envs, so `[0,30000,80000]` is
     hit almost instantly at scale. Set milestones as fractions of total env-steps
     (≈ `num_learning_iterations × num_steps_per_env × num_envs`). Put the chosen numbers in the config.
5. **Read the proxy** (RUN.md Step 4 eval, `im_eval`): does curriculum show longer early episode length
   and a faster/smoother early MPJPE drop, with **final `success_rate` matching** the fixed-strict
   baseline? Record numbers in a new `docs/progress/<date>.md` and an
   `experiments/stage2-termination-curriculum/RESULTS.md`.
6. **Only if the proxy holds**, queue a full-scale run (`num_envs=4096`). Budget GPU-hours first.

**Definition of done (Track A, day 1):** mechanism confirmed firing + one reduced-scale curriculum-vs-
baseline comparison with eval numbers written down.

---

## TRACK B — if NO training host yet (stay productive on the A10G/CPU)

Goal: make the **next stages launch-ready** the same way Stage 2 is, so a future GPU host can run a whole
batch. Each follows the proven loop: **verify mechanism in source → write patch+config → CPU static-validate
→ keep submodule pinned**.

1. **Stage 3 — progressive domain randomization** (lowest risk after Stage 2).
   - Verify: does the `force_push_linear_curriculum` slot already wire to `linear_curriculum`? Read the
     injection block in `modular_tracking_env_cfg.py` and `push_robot` event
     (`config/manager_env/events/terms/push_robot.yaml`).
   - Build: a `push_scale_curriculum` modify_fn (dict-valued range scaling 0.3→1.0) + a `dr_ramp.yaml`
     curriculum config. Static-validate like Stage 2.
2. **Stage 1 — PLR/regret sampler** (higher value, more code).
   - Verify: can a per-step bin-id be registered in `RolloutStorage` (`register_key`)? Where is
     `update_adaptive_sampling_probabilities` and what does it operate over (active-bin subset vs full set)?
   - Build: a `MotionLibBase` subclass swapping the score to `(1-ρ)·regret + ρ·staleness`; document the
     post-GAE bincount plumbing. (No CPU run possible, but the code + a unit test of the scoring math are.)
3. **De-risk the agentic track (B7/C1) — LoRA on the GR00T VLM half** (pure code/spike, no sim).
   - Verify: `qwen3_backbone.set_trainable_parameters` (boolean gates, no `peft`), `select_layer=12`
     truncation, and `setup.py` strict-key loader.
   - Spike: prototype a `LoraConfig` branch + a `merge_and_unload`-then-validate path; write a failing/passing
     test against the strict key validator. **Open Q to settle:** does the TAO `tao-finetune-cosmos-reason`
     skill accept `Cosmos-Reason2-2B` as a base (its verified base is `Cosmos3-Nano`)? Try the conversion
     helper on the GR00T backbone dir and see if the validator passes.

**Definition of done (Track B, day 1):** at least Stage 3 fully launch-ready (patch + config + static
checks green), committed + pushed.

---

## Housekeeping (either track, ~10 min)
- **Repo visibility/fork decision:** confirm keep-public-fork vs private vs detach-from-fork. If detaching:
  `gh api -X POST repos/linjiw/groot-tao-agentic-rl-curriculum/... ` (or recreate as non-fork). If going
  private: `gh repo edit linjiw/groot-tao-agentic-rl-curriculum --visibility private`.
- **CI idea:** add a tiny GitHub Action running `apply_and_validate.sh` against the pinned submodule so the
  patch can't silently rot against upstream drift.
- Add `experiments/stage2-termination-curriculum/RESULTS.md` once any run happens (even a smoke run).

## Quick reference — the two facts the whole project rests on
1. GR00T N1.7 VLM backbone **is `nvidia/Cosmos-Reason2-2B` (Qwen3-VL)** — `qwen3_backbone.py`. Same family
   TAO's `tao-finetune-cosmos-reason` LoRA-SFTs.
2. SONIC's RL trainer **subclasses HuggingFace TRL `PPOTrainer`** — `ppo_trainer.py:321`. LLM-RL and
   robot-RL toolchains converge.

## Known traps (don't relearn these)
- Don't `bind-mount`/edit the submodule and commit it — keep it pinned; ship changes as patches in our repo.
- `modify_term_cfg` is **IsaacLab's**, not `gear_sonic`'s; address shorthand `terminations.` →
  `termination_manager.cfg.`.
- `open_loop_eval.py` reports **MSE/MAE**, not `success_rate` (that's `im_eval` / manual robot trials).
- GR00T tokens are **continuous (flow-matching)**; SONIC decoder expects **FSQ-quantized** — representation
  mismatch, not distribution shift (Track-B item 3 and `06-risks-and-open-questions.md`).
- `num_steps` milestones are in **env-steps across all envs** — re-scale for `num_envs`.
