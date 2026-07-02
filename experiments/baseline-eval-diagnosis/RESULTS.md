# Baseline eval diagnosis — why success_rate = 0.0

Date: 2026-07-02. Box: A10G (23 GB), container `isaac-lab-base`, WBC pinned at
`0e35637` (citations valid in `external/GR00T-WholeBodyControl` and in
`/workspace/GR00T-WholeBodyControl`). Raw outputs of this session:
`/workspace/wbc-training-logs/diagnosis/` (container).

Labels: **[measured]** = number read from an artifact produced on this box;
**[source: file:line]** = code citation; **[hypothesis]** = plausible, not verified.

---

## 1. What success_rate actually is

**Success = the environment plays its motion clip from frame 0 to the final
frame without ever triggering a non-timeout termination.** For our 2 motions
that is 2002 frames at 50 Hz control ≈ 40 s of uninterrupted tracking per clip.

Mechanics, all in `gear_sonic/trl/callbacks/im_eval_callback.py`:

- Per step, `dones` from the env are taken and **timeout dones are masked out**
  (`died[time_outs] = False`) [source: im_eval_callback.py:363–364]. A
  termination only counts as failure if it happens strictly before the last
  motion frame: `termination_state = (curr_steps <= motion_num_steps - 1) AND died`,
  OR-accumulated into a sticky per-env `terminate_state`
  [source: im_eval_callback.py:366–370].
- `progress_state` increments each step only for never-terminated envs
  [source: im_eval_callback.py:372]; at batch end it is divided by the full
  motion length → per-motion `progress` [source: im_eval_callback.py:419–426].
- `success_rate = 1 − mean(terminated)` over the unique motions
  [source: im_eval_callback.py:428–433 and 747]; `progress_rate =
  mean(progress)` [source: im_eval_callback.py:434–436 and 748]; survivors are
  set to progress 1.0 [source: im_eval_callback.py:696].
- The "motion completed" timeout itself is `tracking_time_out` (elapsed ≥ motion
  length, flagged `time_out: true` so it is *not* a failure)
  [source: gear_sonic/envs/manager_env/mdp/terminations.py:245–261,
  config/manager_env/terminations/terms/motion_time_out.yaml].

Eval-time termination terms (`+manager_env/terminations=tracking/eval`)
[source: gear_sonic/config/manager_env/terminations/tracking/eval.yaml]:

| term | fires when | threshold (eval) | training value (our run) |
|---|---|---|---|
| `anchor_pos` | anchor Z-height error | > 0.25 m | 0.15 m (adaptive down 0.75) |
| `anchor_ori_full` | squared quat error | > 1.0 rad² (=1 rad) | 0.2 rad² (~0.45 rad) |
| `ee_body_pos` | any ankle/wrist Z error | > 0.25 m | 0.15 m (adaptive) |
| `time_out` | motion fully played | — (success path) | same |

**Correction (adversarial review M2):** `foot_pos_xyz` is NOT absent at
eval. `eval_agent_trl.py:79–112` loads the checkpoint-sibling
`config.yaml` and OmegaConf-merges the eval override on top; since
`eval.yaml` does not name `foot_pos_xyz`, the training term survives the
merge **at its training threshold (0.2 m)** — confirmed in the eval logs,
which list 5 active termination terms including `foot_pos_xyz`
[measured: eval_baseline/step_002000/eval.log, step_010000/eval.log].
Termination functions: `exceeded_anchor_height`
[source: terminations.py:97–127], `exceeded_anchor_ori`
[source: terminations.py:160–179], `exceeded_body_height`
[source: terminations.py:206–236].

`Start: 0` in the eval log is `env.start_idx` — the motion-library batch offset
(we have 2 motions, 64 envs, one eval loop), *not* a start frame; every episode
starts at frame 0 and success requires ~2002 consecutive good steps
[source: im_eval_callback.py:265–276, 877].

So success_rate=0.0 means: **in no episode did the policy stay within 0.25 m /
1 rad of the reference for the full 40 s clip.** It is a very sparse,
all-or-nothing metric.

## 2. Pipeline sanity check: release checkpoint on the same 2 motions

Command: the verified eval invocation, checkpoint
`/workspace/GR00T-WholeBodyControl/sonic_release/last.pt` (NVIDIA full-scale
training, global step 41,550 — [measured] from the `model_step_041550.pt` copy
the eval script writes). Note: the release checkpoint ships without a sibling
`config.yaml`; we downloaded `sonic_release/config.yaml` from HF
(`nvidia/GEAR-SONIC`) into the same dir — without it `eval_agent_trl.py`
errors (`experiment_dir` not in struct) [source: eval_agent_trl.py:79–112].

Output: `/workspace/wbc-training-logs/diagnosis/eval_release/metrics_eval.json`.

**Release result [measured]: success_rate = 1.0, progress_rate = 1.0.**
Terminated counter stayed 0 through all 2002 steps for all 64 envs
[measured: eval_release/eval.log]. **The eval pipeline, motion data, and
success criterion are fine. Our baseline is simply far from completing the
motions.**

### Release vs baseline table [measured]

| checkpoint | success_rate | progress_rate | mpjpe_g (mm) | mpjpe_l (mm) | per-motion progress (A001_M, A001) |
|---|---|---|---|---|---|
| baseline step 2k | 0.0 | 0.0032 | 36.3 | 29.3 | 0.0030, 0.0035 |
| baseline step 4k | 0.0 | 0.0502 | 56.1 | 36.0 | 0.0500, 0.0504 |
| baseline step 6k | 0.0 | 0.0477 | 62.1 | 43.4 | 0.0539, 0.0415 |
| baseline step 8k | 0.0 | 0.0442 | 60.2 | 45.6 | 0.0554, 0.0330 |
| baseline step 10k | 0.0 | 0.0410 | 60.7 | 32.2 | 0.0549, 0.0270 |
| **sonic_release** | **1.0** | **1.0** | **120.9** | **18.0** | 1.0, 1.0 |

Release per-motion mpjpe_g: 196.5 mm (A001_M), 45.2 mm (A001); mpjpe_l 20.0 /
16.0 mm; mpjpe_pa 11.8 mm [measured].

### The mpjpe_g inversion (survivor bias, confirmed)

The *succeeding* release model shows **2× the global mpjpe** of our failing
baseline (120.9 vs 60.7 mm). Reason: per-motion mpjpe is averaged only over
frames actually executed inside the eval loop, which ends when all counted envs
have terminated [source: im_eval_callback.py:398–413 (loop bound = max steps of
non-terminated envs), 455–460 (mean over collected frames)]. Our baseline only
ever executes the first ~110 of 2002 frames — the easy opening of the walk,
before global drift accumulates — while the release model is scored over the
full 40 s including drift (its mpjpe_l stays 18 mm; the 120 mm is mostly global
root drift on the mirrored clip). **mpjpe_g is therefore anti-correlated with
survival and must not be used as a "lower is better" score across runs with
different survival lengths.** This is visible inside our own curve too: 2k→4k
progress 0.003→0.050 while mpjpe_g 36→56.

Additional caveat [hypothesis, code-based]: envs that terminate mid-loop are
auto-reset by the manager env and keep contributing frames until the loop ends;
mpjpe collection does not mask them [source: im_eval_callback.py:344–354 —
appended for all envs every step], so a terminated motion's per-motion mpjpe
mixes frames from more than one partial attempt.

## 3. Why progress_rate plateaus at ~0.05

- **What kills the episodes.** Eval logs only expose the aggregate
  `Terminated: N` counter (no per-term breakdown). At step 2k, 59/64 envs
  died at step ~7; at 10k the fleet dies between steps ~60–110 and the loop
  ends at step 110 [measured: eval_baseline/step_*/eval.log]. Given the §1
  correction (foot_pos_xyz IS active at eval, at its strict training 0.2 m,
  via the config merge) and the training-side termination mix — foot_pos_xyz
  72%, ee_body_pos 14%, anchor_ori_full 11%, anchor_pos ~0% [measured:
  baseline_10k.log tail] — **the lead suspect is `foot_pos_xyz` at 0.2**
  [hypothesis; needs per-term logging or threshold ablation to confirm].
  This also better explains why eval survival ≈ training survival despite
  nominally "relaxed" eval thresholds: the binding term was never relaxed.
- **Consistency check.** Training mean episode length at strict thresholds:
  ~55–100 steps (last iterations: 55.5–93.0) [measured: baseline_10k.log].
  Eval survival at relaxed thresholds: ~60–110 steps [measured]. Same regime —
  the policy reliably survives ~1–2 s of a 40 s clip, then derails. progress
  0.05 × 2002 ≈ 100 frames. Nothing about eval is inconsistent with training;
  the policy is just early in learning.
- **Scale arithmetic.** Ours: 10,000 iters × 16 steps/env × 256 envs =
  **41.0 M env-steps** [measured: config.yaml:266 `num_steps_per_env: 16`,
  :271 `num_learning_iterations: 10000`, 256 envs]. Release recipe:
  `num_steps_per_env: 24`, `num_envs: 4096`, target
  `num_learning_iterations: 100000` [source: sonic_release/config.yaml:6,238,243];
  the shipped checkpoint is at global step 41,550 → 41,550 × 24 × 4096 ≈
  **4.08 B env-steps per process — almost exactly 100× ours** — and the README
  recommends 64+ GPUs for this config, so the true total is likely another
  1–2 orders of magnitude higher [hypothesis for the multi-GPU factor;
  source: README.md:218–224]. A 100–10,000× data gap fully accounts for the
  performance gap; there is no evidence of an eval-side bug.

## 4. Videos — DONE via the MuJoCo fallback (RTX path remains broken)

**Videos exist** (rendered 2026-07-02, after the comparison freed the GPU):

- `videos/baseline_step_010000/000000.mp4`, `000001.mp4` — our 10k
  baseline, 50 frames @ 25 fps (~2 s: the policy's full survival window;
  render_frame_skip=2 × ~110 executed steps).
- `videos/sonic_release/000000.mp4`, `000001.mp4` — the release
  checkpoint, 1001 frames @ 25 fps (the full 40 s clip, both motions).

Path taken (fallback C, since the RTX probe failed again — see below):
TrajectoryRecorder pkl dump (no cameras, proven headless path; verified
`+manager_env.recorders.trajectory.*` struct-add works) → offline
kinematic replay in MuJoCo (`render_trajectory_mujoco.py`, joint mapping
by NAME from IsaacLab breadth-first order `joint_utils.py:11–40` to MJCF
qpos order, `MUJOCO_GL=egl`, offscreen framebuffer raised via MjSpec) →
mp4 via imageio. Visual spot-check [measured]: release frame at t≈16 s
shows a natural mid-stride walk; baseline frame at t≈1.2 s shows the
robot leaning back, arms flailing — the balance-loss derail the metrics
predicted (§5 verdict (a) confirmed visually).

Caveat: these are KINEMATIC replays of recorded joint/root states on a
checkerboard MuJoCo scene, not Isaac renders — good for judging motion
quality, not for judging contacts/shadows/scene fidelity.

### The original RTX render path (still broken)

- The metrics eval path (`headless`, no cameras) works. The render path
  (`render_results=True` → `enable_cameras=True` → experience file
  `isaaclab.python.headless.rendering.kit`) **segfaults during app startup** in
  `librtx.scenedb.plugin.so` (carb crashreporter, crash inside
  viewport/hydra-engine init: py-spy shows
  `AppLauncher.__init__ → create_new_stage → __enable_hydra_engine`)
  [measured: /workspace/wbc-training-logs/diagnosis/render_baseline/render.log,
  /isaac-sim/kit/data/Kit/Isaac-Sim/5.1/crash_2026-07-02_18-53-41_26602.txt].
- **The crash is infra-level, not our command**: a minimal repro (plain
  `AppLauncher(headless=True, enable_cameras=True)`, no eval code, no
  checkpoint) crashes identically [measured, this session]. Isaac Sim 5.1 +
  driver 595.71.05 + A10G in this container has never had the rendering path
  verified (infra guide only verified training).
- Two attempts + repro crashed at 2 and 0 envs' worth of load, so it is not
  VRAM (GPU memory was 0 MiB at crash; host RAM 27 GB free) [measured].
- One command-level trap found and fixed on the way: the saved training config
  already defines `recorders`, so the render override must be
  `manager_env/recorders=render` (no leading `+`), otherwise Hydra errors with
  "Multiple values for manager_env/recorders" [measured].
- The `--rendering_mode performance` probe (RENDER_TODO.md step 0) was run
  after the GPU freed: **still segfaults identically** [measured
  2026-07-02 20:40]. RTX rendering on this box is dead pending a
  driver/Isaac-Sim fix; the MuJoCo fallback above is the working video
  path and is scriptable for future runs.

## 5. Tracking-quality verdict

**(a)-leaning: the baseline tracks the motion coarsely for the first ~1–2
seconds and then derails/falls; it is NOT reward-hacking survival, and it is
not failing outright from step 0.** Evidence:

- While alive, pose tracking is real: mpjpe_l 29–46 mm across checkpoints
  (release reference while succeeding: 18 mm) and mpjpe_pa 12–25 mm
  [measured]. Random or fallen poses would be hundreds of mm.
- Survival is short (~0.03–0.055 of the clip), so "surviving without tracking"
  (b) is ruled out — there is no long survival to hack [measured].
- The improvement 2k→4k (progress ×15) then plateau 4k→10k with slightly
  degrading per-motion progress on A001 (0.050→0.027) suggests the policy is
  still deep in the learning curve at 41 M env-steps, ~100× (or more) short of
  the release recipe [measured + arithmetic above].
- Visual confirmation [measured, videos §4]: the release policy walks
  naturally through the full clip; our baseline holds a walking stance
  briefly, then leans back with arms flailing and derails within ~2 s —
  exactly the coarse-track-then-balance-loss picture the metrics implied.

## 6. Implications for scoring the ON-vs-OFF manager comparison

1. **Do not use success_rate** at this training scale — it will be 0.0 for
   both arms (needs 2002 flawless steps; the policy manages ~110).
2. **Do not use mpjpe_g (or any all-frames mpjpe) jointly with progress_rate
   as "both should improve".** mpjpe_g mechanically *worsens* as survival
   lengthens (survivor bias, §2). A manager that genuinely improves tracking
   will look worse on mpjpe_g.
3. Recommended primary: **progress_rate** (eval-side, deterministic —
   [measured] same ckpt → identical metrics). Secondary: **mpjpe_l** (local
   pose error, much less drift-sensitive: our curve 29→32 mm is roughly flat
   while progress ×13). If an mpjpe-style tiebreaker is wanted, compute it
   over a **fixed early window** (e.g. first 100 frames) so both arms are
   scored on the same frames, or compare only at matched progress.
4. progress_rate saturates at 1.0 for survivors [source:
   im_eval_callback.py:696] — fine at this scale, matters later.
5. Per-motion `all_metrics_dict.progress` is worth logging per arm: the two
   motions already diverge (0.055 vs 0.027 at 10k) and adaptive sampling acts
   on exactly that asymmetry.

## Open questions

- Which eval termination term actually fires (ee_body_pos vs anchor_ori_full)?
  Needs a one-line per-term counter in the eval loop or an eval with terms
  ablated one at a time (cheap: ~2 min/run when GPU free).
- The RTX render segfault root cause (driver 595.71.05 vs Isaac Sim 5.1 RTX
  scenedb; container was never render-verified). See RENDER_TODO.md for the
  candidate fixes; if none work, the trajectory-recorder + offline MuJoCo
  render fallback avoids RTX entirely.
- Release config's actual world size (how many GPUs × 4096 envs) — bounds the
  true data gap between 100× and ~10,000×.
- Whether per-motion mpjpe for terminated envs mixes multiple partial attempts
  (post-reset contamination) — code reading says yes [hypothesis]; would bias
  per-motion tables, not the headline conclusions.

## Artifact index

- Release eval (this session): `/workspace/wbc-training-logs/diagnosis/eval_release/{metrics_eval.json,eval.log}`
- Baseline evals (pre-existing): `/workspace/wbc-training-logs/eval_baseline/step_{002000..010000}/`, `/workspace/wbc-training-logs/eval_smoke/`
- Render crash logs: `/workspace/wbc-training-logs/diagnosis/render_baseline/render.log`, `/isaac-sim/kit/data/Kit/Isaac-Sim/5.1/crash_2026-07-02_18-*.txt`
- Release config now at: `/workspace/GR00T-WholeBodyControl/sonic_release/config.yaml` (downloaded from HF this session)
- Videos: none yet — see `RENDER_TODO.md`; target host dir `experiments/baseline-eval-diagnosis/videos/`
