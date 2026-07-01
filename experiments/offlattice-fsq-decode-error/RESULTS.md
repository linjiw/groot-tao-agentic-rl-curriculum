# Experiment ① — Off-lattice FSQ decode-error for the SONIC universal-token decoder

**Question (repo risk #1, `docs/design/06-risks-and-open-questions.md`):** GR00T emits
*continuous* flow-matching tokens; SONIC's decoder was trained on *FSQ-quantized* lattice
tokens. If a continuous (off-lattice) token is fed to the deployed decoder, how badly does
the decoded action degrade, and does snapping the token to the nearest lattice point recover it?

**Bottom line:** Off-lattice decode degradation is **small and smoothly bounded, NOT
catastrophic** [measured], and **snap-to-lattice recovers it essentially perfectly for any
perturbation smaller than half a quantization step** [measured]. This *downgrades* risk #1: the
VLA-as-agent track does **not** require quantization-aware BC as a hard blocker before anything
else, though a cheap snap-to-lattice guard at deploy is advisable [speculative].

---

## 1. Verified lattice geometry

Source: `gear_sonic/trl/modules/universal_token_modules.py` and the hydra configs it consumes.

| Quantity | Value | Source (verified) |
|---|---|---|
| Quantizer class | `vector_quantize_pytorch.FSQ` | `gear_sonic/config/actor_critic/quantizers/fsq.yaml:1` (`_target_: vector_quantize_pytorch.FSQ`) |
| `num_fsq_levels` (= `token_dim`) | **32** | `config/actor_critic/universal_token/all_mlp_v1.yaml:34`; `universal_token_modules.py:231` (`self.token_dim = self.num_fsq_levels`) |
| `fsq_level_list` (per-dim codebook size) | **32** (int → broadcast to `[32]*32`) | `all_mlp_v1.yaml:35`; `universal_token_modules.py:219-220` |
| `max_num_tokens` | **2** | `all_mlp_v1.yaml:36`; `universal_token_modules.py:233-234` |
| `token_total_dim` | **64** (= 32 × 2) | `universal_token_modules.py:237` (`token_dim * max_num_tokens`) |
| FSQ codes per dim | 32 discrete values | `FSQ(levels=[32]*32)` (instantiated live) |
| Normalized lattice values | k/16, k ∈ [−16, 15] → **min −1.0, max 0.9375** | `FSQ.bound`: `round_ste(bounded_z)/half_width`, `half_width = levels//2 = 16` (live-inspected) |
| Inter-lattice spacing (quantization step) | **1/16 = 0.0625** | derived from `half_width=16` (live-inspected) |

**Verdict on the docs' "2 tokens × 32 levels = 64-dim":** ✅ **[verified] correct.** The
config-time values (`num_fsq_levels=32`, `max_num_tokens=2`) match the code defaults path and
produce a flat 64-dim token. (The *module defaults* in the `__init__` signature are different —
`num_fsq_levels=5`, `fsq_level_list=16`, `universal_token_modules.py:76-77` — but the deployed
SONIC config overrides them to 32/32/2. The earlier review note about "config-time vs code
default" numbers applies here: trust the config, which is what ships.)

### The deployed decoder & the "1.25 magnitude guard"

- The **deployed decoder is an ONNX graph**, not the PyTorch `UniversalTokenModule`:
  `~/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/snapshots/5e22ddc.../model_decoder.onnx`
  [verified by loading it]. Signature: **input `obs_dict` shape `[1, 994]` (float32) → output
  `action` shape `[1, 29]`** (29 = G1 body joints).
- The doc's "flat continuous (B,64)" refers to the **`token_state` sub-vector** = the first
  **64** dims of the 994-dim decoder input (the flattened 2×32 motion token). Confirmed by
  `observation_config.yaml` (`token_state` is the first enabled observation; header comment
  documents a 64-dim token block). ✅ **[verified]** — with the correction that the decoder's
  *full* input is 994-dim, of which the token is the leading 64.
- **The 1.25 guard** (`scripts/run_vla_inference.py:291-297` [verified]): rejects an action chunk
  when `np.abs(action[motion_key]).max() > 1.25`, printing "Exceeds action bound, skipping" and
  returning `None`. It gates the **magnitude of the emitted motion token** (a blow-up / outlier
  rejector). It does **NOT** snap off-lattice points to the lattice and does **NOT** check
  lattice membership. Note the valid lattice max is 0.9375 (< 1.25), so a token can be fully
  off-lattice yet still pass the guard. ✅ **[verified]** — the guard confirms/rejects gross
  outliers only, exactly as `docs/design/06` states.

---

## 2. Method

Standalone script: `experiments/offlattice-fsq-decode-error/measure_decode_error.py`
(run with `/workspace/GR00T-WholeBodyControl/.venv_sim/bin/python`).

1. Instantiate the **real** `FSQ(levels=[32]*32)`.
2. Sample B=512 continuous latents `z ~ N(0,1)` of shape `(B, 2, 32)`.
3. **Clean / teacher (on-lattice):** `t_clean = FSQ.quantize(z)` → values on the k/16 lattice.
4. **Off-lattice:** `t_off = t_clean + U(−a, a)`, with `a = frac × step`, `step = 0.0625`,
   for `frac ∈ {0, 0.1, 0.25, 0.5, 1.0}`.
5. **Snap:** nearest-codebook projection by direct rounding to k/16, clamped to [−16, 15]/16.
   (We deliberately do **NOT** re-call `FSQ.quantize()` for the snap: `quantize()` applies a
   `tanh` bound and is **not idempotent** on already-normalized lattice values — re-quantizing a
   clean token perturbs it by ~1.35 in action space. Direct rounding is the true nearest-lattice
   projection for the deployed normalized token space; verified `snap==clean` for `frac ≤ 0.5`.)
6. Decode `t_clean`, `t_off`, `t_snap` through the **real ONNX decoder**, holding the other 930
   obs dims at a fixed baseline so only the token perturbation drives the measured deltas.
   Compare the 29-dim action outputs (L2, per-dim MSE, max-abs).

**Hardware note:** FSQ ran on the **A10G GPU** (`torch.cuda.is_available()==True`). The ONNX
decoder ran on **CPUExecutionProvider** — the installed `onnxruntime` wheel (1.23.2) is the
CPU build and did not expose CUDA EP in this venv; the decoder is a small MLP so CPU is fine and
this does not affect the numerical result. `onnxruntime`, `vector_quantize_pytorch`, and
`einops` were **not** present in `.venv_sim` and were installed via `uv pip` into that venv to
run the real FSQ + real ONNX decoder (no source was modified).

Reference scale: the clean decode has **mean action L2 ≈ 7.44** (std ≈ 1.42) over the batch, so
`raw_rel` below = raw-off L2 ÷ 7.44 expresses degradation as a fraction of the action's own norm.

---

## 3. Measured results  [measured]

B=512, seed=0. `step = 0.0625`. Errors are in raw 29-dim action units unless noted.

| frac·step | noise amp | token off-dist (L2, 64-dim) | **raw-offlattice decode L2** | raw rel-to-clean | **snapped decode L2** | recovery frac | snap==clean token? |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 0.00 | 0.0000 | 0.0000 | 0.00000 | 0.0000 | 0.00000 | — | ✅ yes |
| 0.10 | 0.0063 | 0.0287 | **0.04936** | 0.0066 | **0.00000** | **1.0000** | ✅ yes |
| 0.25 | 0.0156 | 0.0721 | **0.12548** | 0.0169 | **0.00000** | **1.0000** | ✅ yes |
| 0.50 | 0.0312 | 0.1438 | **0.24628** | 0.0331 | **0.00000** | **1.0000** | ✅ yes |
| 1.00 | 0.0625 | 0.2885 | **0.49906** | 0.0671 | **0.60673** | −0.2157 | ❌ no |

Supporting per-magnitude detail (from `results.json`):

- raw-off **max-abs** action error: 0.0 → 0.075 → 0.19 → 0.376 → **0.904** across the fracs.
- raw-off **per-dim MSE**: 0 → 1.4e-4 → 8.9e-4 → 2.3e-3 → **9.5e-3**.
- snapped **max-abs** error: 0 for frac ≤ 0.5; **1.176** at frac = 1.0.

### Reading the table

- **Off-lattice degradation is smooth, monotonic, and modest.** Even at a full ¼-step of
  uniform noise the decode moves only ~0.13 L2 (≈1.7% of the action norm); at a *half-step*
  (`frac 0.5`) it's ~0.25 L2 (≈3.3%). No cliff, no blow-up — the ONNX decoder is a locally
  Lipschitz MLP and interpolates gracefully off the lattice. **[measured]**
- **Snap-to-lattice recovers 100%** for any perturbation up to **half a quantization step**
  (`frac ≤ 0.5`): the snapped token is bit-identical to the clean token, so the decode error is
  exactly 0. **[measured]**
- **At a full step (`frac 1.0`) snapping can OVER-correct** into the *neighbouring* lattice cell
  (recovery goes negative, snap L2 0.61 > raw-off 0.50). This is expected: with noise up to ±1
  full step, a sizeable fraction of dims cross the ±½-step decision boundary, so rounding lands
  on the wrong codeword. Snapping is only a faithful recovery operator when the continuous token
  is within ±½ step of its intended lattice point. **[measured]**

---

## 4. VERDICT

**Is off-lattice decode degradation catastrophic?** **No.** **[measured]** Feeding continuous
(off-lattice) tokens to the real deployed SONIC ONNX decoder produces small, smooth, bounded
action error — ≤ ~3.3% of the action norm out to a half-step perturbation, worst single-joint
error ≤ ~0.9 rad only at a *full-step* perturbation. The decoder does not fall off a cliff when
its input leaves the FSQ lattice.

**Does snap-to-lattice recover it?** **Yes, perfectly, within the basin of attraction.**
**[measured]** For any continuous token within ±½ quantization step of its intended lattice
point, direct nearest-codebook rounding reproduces the clean token exactly → zero decode error.
Beyond ±½ step, rounding can select the wrong neighbour and no longer recovers — so snap is a
reliable operator only when the VLA's continuous output stays within half a step of a valid code.

**Implication for the VLA-as-agent track:** **[speculative, grounded in the measurements]**
- Risk #1 is **downgraded from "catastrophic, blocks everything" to "manageable."** A GR00T VLA
  emitting continuous tokens near the lattice will decode to near-teacher actions; the decoder's
  local smoothness means small flow-matching drift ≈ small action drift.
- **Quantization-aware BC is NOT a hard prerequisite** for starting the VLA-as-agent track. The
  earlier roadmap assumption (that a catastrophic-and-unrecoverable result would force QA-BC
  first) is **not supported** by these numbers.
- **Recommended cheap mitigation instead:** add a **snap-to-lattice projection** on the VLA's
  emitted motion token at deploy (round to k/16, clamp to [−16,15]/16) *before* the decoder, and
  keep the existing 1.25 magnitude guard for gross outliers. This is a few lines, needs no
  retraining, and buys exact-teacher decoding whenever the VLA stays within ½-step.
- **Open follow-up (out of scope here):** the untested failure mode is *systematic* VLA bias that
  pushes tokens > ½ step off-lattice consistently (where snap flips to the wrong code). Worth a
  short measurement of the trained GR00T VLA's actual token-to-nearest-lattice distance
  distribution before relying on snap in production. If that distance is routinely > ½ step,
  QA-BC (or a straight-through/soft-quantization head) becomes worthwhile — but as a *tuning*
  step, not a gate.

---

## 5. Reproduce

```bash
/workspace/GR00T-WholeBodyControl/.venv_sim/bin/python \
  experiments/offlattice-fsq-decode-error/measure_decode_error.py
# writes experiments/offlattice-fsq-decode-error/results.json
```

Requires (installed into `.venv_sim` via `uv pip`, no submodule edits):
`vector_quantize_pytorch==1.29.1`, `onnxruntime==1.23.2`, `einops==0.8.2`.
GEAR-SONIC decoder from the HF cache `models--nvidia--GEAR-SONIC` snapshot `5e22ddc`.

### Label key
`[verified]` = read directly from source / live object. `[measured]` = produced by running the
real FSQ + real ONNX decoder in this experiment. `[speculative]` = reasoned implication.
