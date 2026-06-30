# 06 — Risks, Gaps, and Open Questions

These survived an adversarial critique pass against the pinned source. They are load-bearing constraints, not boilerplate caveats.

## Top risks & gaps

1. **Continuous-vs-quantized representation mismatch (biggest unaddressed risk).** GR00T's flow-matching DiT regresses **continuous** 64-dim motion_tokens, but SONIC's decoder was trained on **FSQ-quantized** tokens on a discrete 32-level-per-dim lattice. Predicted tokens need not lie on the lattice; the 1.25 magnitude guard (`run_vla_inference.py:291`) rejects outliers, not off-lattice points, and a `post_quantization` residual (B2) does not fix it. Requires snap-to-lattice / quantization-aware BC and an explicit off-lattice decode-error metric.

2. **TAO `AutoMLRunner` is NOT reusable as-is for SONIC.** It is bound to the TAO-SDK job path (`skill_info.yaml`, TAO container, `data_sources` mapped to train/evaluate/inference actions, platform job submission), while SONIC is `accelerate launch train_agent_trl.py` needing IsaacSim/IsaacLab. B3/B4/B5 require a substantial new adapter; "reuse, not greenfield" is overstated.

3. **Compute budget is the dominant practical constraint and is largely unquantified.** One PPO trial is `num_envs=4096` on 64+ GPUs up to 1e5 iterations; `autoresearch` (10–50 recs) or `pbt` (population × such runs) plus the 6–12-run curriculum ablation matrix are very expensive. Proxy-to-full-scale correlation is asserted, not validated.

4. **GR00T has NO LoRA/PEFT today** (`qwen3_backbone.py` boolean gates only). Adding it must contend with `select_layer=12` truncation (adapters on dropped layers / visual tower discarded) and `setup.py` strict-key validation, which raises on any missing/unexpected/mismatched key.

5. **TAO cosmos-reason skill's verified base is `nvidia/Cosmos3-Nano`.** That it accepts `nvidia/Cosmos-Reason2-2B` (and that the converted dir passes GR00T's strict-key validator after `merge_and_unload`) is **unverified** and is a prerequisite spike, not an assumption.

6. **C4 RL-mode transfer to TAO does NOT deliver the hardest piece (the reward).** SONIC's reward is a dense analytic tracking kernel; a VLM RL reward is a sparse/learned verifier. Only scaffolding (GAE, KL-LR, anti-concentration sampler) transfers. Also, Cosmos-RL is a separate FSDP framework, not built on HF TRL — the "shared TRL substrate" is a code-pattern analogy.

7. **MotionBricks-based targeted data generation is not feasible today.** Text-conditioning is disabled in all shipped configs, training uses synthetic random tensors, the full pipeline is roadmap, and there is a 4096-d (text slot) vs 2048-d (Cosmos-Reason hidden) dimensional mismatch plus a G1Skeleton34 ↔ SONIC-29DoF skeleton/representation gap with no conversion pipeline.

8. **Stage-1 PLR regret needs per-bin keying of returns that does not exist in `RolloutStorage` today.** A new per-step `motion_time_step`/bin-id key must be registered and bincounted post-GAE; accumulators must span the full bin set but index into the GPU-resident active subset.

9. **Stage-2 load-bearing assumption** (terminations read `params.threshold` from live cfg each call) **and B5's pbt checkpoint-state co-copy** (sampler `state_dict` + `running_mean_std` persist via different paths than weights; TAO pbt mutates spec params) are both unverified mechanisms.

10. **Eval/train threshold decoupling.** Training uses strict adaptive thresholds while `im_eval` `success_rate` is computed under fixed relaxed eval thresholds, so any agent tuning training thresholds optimizes a partly-decoupled objective.

11. **Stage-4 mastery oracle plumbing.** Per-bin MPJPE is produced in a separate (possibly different-process) eval pass, not in the training rollout, so feeding it back to the training `MotionLibRobot` at cadence is unspecified.

12. **Multi-venv + C++ deploy boundary.** SONIC spans `.venv_sim`/`.venv_teleop`/`.venv_data_collection`/`.venv_inference` plus a C++/TensorRT build; any AutoML/DEFT orchestrator must cross these, and the `FileLock`-guarded IsaacSim launch and wandb-dir parsing are brittle plumbing.

13. **B1↔B3 circular coupling.** Improving the SONIC controller (B3) invalidates the GR00T BC token dataset (B1), requiring re-export; needs an interface-version freeze (FSQ `levels=32`, `max_num_tokens=2`) and a defined iteration cadence.

## Open questions

1. Does `terminations.py` read `params.threshold` from the live `TerminationManager` term cfg on every call (so `modify_term_cfg` schedules take effect at episode boundaries), or is it cached at setup (making Stage 2 inert and forcing a `schedule_dict`-based alternative)?
2. Can the `tao-finetune-cosmos-reason` skill (and its Cosmos3-Nano→Qwen3-VL conversion helper) actually take `nvidia/Cosmos-Reason2-2B` as a base, and does the converted/merged `state_dict` pass GR00T `setup.py` strict-key validation?
3. Given GR00T truncates to `select_layer=12` and discards adapters on dropped layers and (unless explicitly targeted) the visual tower, does `q_proj`/`v_proj` LoRA on only the kept 12 LLM layers provide enough capacity to be worthwhile vs `tune_top_llm_layers`?
4. What is the off-lattice decode degradation when GR00T's continuous flow-matching motion_tokens drive SONIC's FSQ-trained decoder, and does a snap-to-lattice / nearest-codebook projection or quantization-aware BC recover teacher-level tracking?
5. Does a low-budget SONIC proxy (fewer envs/iters or a motion subset) correlate with full-scale 4096-env `success_rate` well enough to make AutoML/curriculum sweeps trustworthy, and what is the GPU-hour cost per `autoresearch` run?
6. By how much (if at all) does an agentic sampler/HPO loop beat the already-well-tuned `sonic_release` failure-weighted defaults — is the LLM-in-the-loop + multi-trial-PPO overhead justified?
7. How do per-bin MPJPE / `failed_keys` from the separate (possibly cross-process) eval pass get back into the training-process `MotionLibRobot` at training cadence to drive Stage-4 retirement and B6 active sampling?
8. How is a 4096-d MotionBricks text embedding produced (which LLM, since Cosmos-Reason2 hidden size is 2048), and what skeleton-conversion + physical-feasibility-filter pipeline would turn G1Skeleton34 MotionBricks output into SONIC 29-DoF motion_lib clips?
9. What concrete mechanism makes TAO `pbt` copy SONIC's sampler `state_dict` + `running_mean_std` alongside model weights, given pbt mutates TAO spec params rather than arbitrary Python state?
10. For the C4 reverse transfer, what reward signal would unlock Cosmos-RL's RL mode for a VLM — since SONIC supplies only RL scaffolding, not the dense analytic reward that does not generalize to a VLM verifier?

## Verification provenance
This plan was produced by reading the pinned submodule source (WBC `0e35637`, Isaac-GR00T `ab88b50`) and the TAO skill bank, then subjecting each section to an adversarial critique that flagged hallucinated line-numbers and a fabricated `success_rate` in `open_loop_eval.py` (corrected: it outputs Action MSE/MAE only). Remaining line-number citations are best-effort; verify against the pinned commits before relying on an exact line.
