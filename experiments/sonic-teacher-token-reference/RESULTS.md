# Reference — SONIC teacher-token distribution vs FSQ lattice

**Companion/reference for Experiment ③** (`experiments/vla-token-lattice-distance/`)
and follow-up to Exp① (`experiments/offlattice-fsq-decode-error/`).

Exp③'s core measurement (a trained GR00T VLA's emitted-token distance to the FSQ
lattice) hit a **NO-GO feasibility gate** — the cached generic `GR00T-N1.7-3B`
does not emit the `unitree_g1_sonic` motion token. This document delivers the
other half that *is* runnable today: **where does the genuine SONIC *teacher*
token distribution sit relative to the FSQ lattice?** It calibrates any future
VLA-emitted distance measurement against a verified baseline.

---

## Verdict [measured] — teacher tokens are EXACTLY on-lattice, input-invariant

The real cached SONIC encoder (`model_encoder.onnx`) emits tokens that sit
**exactly** on the FSQ lattice — `max_dist = 0.0 steps` — for **every** input
tested (real motion + four synthetic distributions). This is not a property of
the input; it is baked into the exported graph.

| Run | n values | mean | median | p90 | p95 | p99 | **max** | frac > ½-step |
|---|---|---|---|---|---|---|:--:|:--:|
| real_motion_enc0 (g1) | 76,928 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 |
| real_motion_enc1 (teleop) | 76,928 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 |
| real_motion_enc2 (smpl) | 76,928 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 |
| **real_motion_all** (3,606 tokens) | 230,784 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 |
| synthetic_zeros | 16,384 | 0.0 | — | — | — | — | **0.0** | 0.0 |
| synthetic_randn×1 | 16,384 | 0.0 | — | — | — | — | **0.0** | 0.0 |
| synthetic_randn×3 | 16,384 | 0.0 | — | — | — | — | **0.0** | 0.0 |
| synthetic_uniform±5 | 16,384 | 0.0 | — | — | — | — | **0.0** | 0.0 |

Distances are in **step units** (1 step = 0.0625 normalized; ½-step = 0.03125).
All values verified by re-running the script independently (parent re-run).

---

## Why it's on-lattice [verified — source]

`gear_sonic/trl/modules/universal_token_modules.py` :: `UniversalTokenModule.encode()`
runs the FSQ quantizer **inside** the graph:

```python
quantized_codes, _ = self.quantizer(latent)   # FSQ(levels=[32]*32)
encoded_tokens = quantized_codes.contiguous()
```

The exported `model_encoder.onnx` therefore returns **post-quantization** codes.
The SONIC decoder consumes these on-lattice tokens as the leading 64 dims of its
994-d `obs_dict` input (the decoder is byte-identical — `md5 1d4391ad…` — to the
one used in Exp①, so results transfer directly).

**Architecture** (config `all_mlp_v1.yaml`): `num_fsq_levels=32`,
`fsq_level_list=32`, `max_num_tokens=2` → flat 64-d token; **3 encoder branches**
(g1 / teleop / smpl). ONNX probe confirms `obs_dict[1,1762] → encoded_tokens[1,64]`
with **dim0 = encoder-index scalar** (0=g1, 1=teleop, 2=smpl; index ≥3 → ScatterND
error, confirming exactly 3 encoders). [measured]

---

## FSQ cross-check [measured]

Feeding genuine **continuous pre-quant latents** (`5·N(0,1)`, shape `[200,2,32]`)
through the real `FSQ(levels=[32]*32)`:

- `quantized_continuous_max_dist_steps = 0.0` (lands exactly on lattice)
- range `[-1.0, 0.9375]` — matches lattice geometry (step 0.0625)
- `encoder_output_on_lattice = True` (`encoder_output_on_lattice_max_dist_steps = 0.0`)

> **Note (corrected artifact):** an earlier draft re-fed *already-quantized*
> values through `FSQ.forward()` and saw a spurious 0.0625 delta. That is **not**
> an idempotence check — `FSQ.forward()` applies an internal bound to its *input*,
> so it must be fed raw continuous latents. The corrected check (above) feeds
> genuine continuous latents and confirms 0.0 residual.

---

## Real vs baseline inputs [verified honesty note]

Real motion source: `walk_forward_amateur_001__A001` — `smpl_filtered/…A001.pkl`
+ `robot_filtered/210531/…A001.pkl`, **1,202 frames**, feature_dim 273 assembled
from REAL motion fields (SMPL joints root-relative, root 6D, robot 29-DOF, pose_aa,
root quat) at REAL magnitudes.

**Limitation:** the exact per-term byte offsets of the 1762-d `obs_dict` require a
live IsaacLab `command_manager` (body-local transforms vs. the robot's runtime sim
pose) and are **not reconstructable from the pkl alone**. Real values were used at
real magnitudes but placed layout-approximately. **This does not affect the
conclusion** — the on-lattice result is proven **input-invariant** by the synthetic
battery (zeros → uniform±5 all give `max_dist=0.0`), because FSQ is baked in-graph.

---

## Interpretation for the Exp③ VLA measurement

The genuine SONIC teacher distribution is **exactly on-lattice — zero residual**.
So when a *sonic-finetuned* GR00T VLA becomes available and its emitted tokens are
measured (the Exp③ core, currently NO-GO for lack of such a checkpoint), those
distances must be read against a **zero baseline**: any nonzero distance is pure
**model-emission drift**, not inherent to the teacher representation. Combined with
Exp①'s result (snap recovers perfectly within ±½ step), the open question is
strictly *"does the VLA's emission drift exceed ½ step?"* — answerable only with a
sonic checkpoint.

---

## Reproduce

```bash
cd experiments/sonic-teacher-token-reference
/workspace/GR00T-WholeBodyControl/.venv_sim/bin/python run_teacher_reference.py
# writes reference_distribution.json; all runs max_steps == 0.0
```

- **Venv:** `/workspace/GR00T-WholeBodyControl/.venv_sim/bin/python` (torch+CUDA;
  onnxruntime 1.23.2 CPU build, `vector_quantize_pytorch`, `joblib`, `einops`).
- **Encoder:** `/workspace/hf-cache/wbc-checkpoints/policy/release/model_encoder.onnx`
  (CPUExecutionProvider; encoder is small, CPU is fine).
- **Source pin:** `gear_sonic` @ `0e35637` (== submodule `external/GR00T-WholeBodyControl`).
  No submodule/source tree modified. No git commit by subagent (parent reviews).

### Label key
`[verified]` = read directly from source / ONNX graph structure.
`[measured]` = numeric result from running the encoder / FSQ in this experiment
(independently re-run by the parent). `[speculative]` = reasoned implication.
