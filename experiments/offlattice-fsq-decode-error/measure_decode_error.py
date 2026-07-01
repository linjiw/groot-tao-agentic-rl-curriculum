#!/usr/bin/env python
"""Experiment (1): Off-lattice FSQ decode-error for the SONIC universal-token decoder.

Measures how much the DEPLOYED SONIC decoder degrades when it receives CONTINUOUS
(off-FSQ-lattice) motion tokens -- the situation created when a GR00T flow-matching
VLA emits continuous tokens into a decoder that was trained on FSQ-quantized ones.

Uses the REAL components:
  * FSQ quantizer  : vector_quantize_pytorch.FSQ(levels=[32]*32)  (the class the
                     gear_sonic hydra config `quantizers/fsq.yaml` instantiates,
                     with num_fsq_levels=32 / fsq_level_list=32 / max_num_tokens=2).
  * Deployed decoder: model_decoder.onnx from the nvidia/GEAR-SONIC HF cache
                     (obs_dict[1,994] -> action[1,29]); the first 64 dims of
                     obs_dict are `token_state` = the flattened 2x32 motion token.

Method
------
1. Sample continuous latents z ~ N(0,1) of shape (B, 2, 32).
2. Clean/teacher path : t_clean = FSQ.quantize(z)  (values on lattice = k/16).
3. Off-lattice path   : t_off  = t_clean + noise, noise ~ U(-a,a) with
                        a = frac * step, step = 1/16 = 0.0625 (verified spacing).
4. Snap path          : t_snap = FSQ.quantize(t_off) (nearest-codebook projection).
5. Decode all three through the real ONNX decoder (token_state slice varied,
   the other 930 obs dims held at a fixed realistic baseline) and compare the
   29-dim action outputs.

Reports per magnitude:
  raw_off_err   = ||dec(t_off)  - dec(t_clean)||     (degradation from continuous)
  snap_err      = ||dec(t_snap) - dec(t_clean)||     (residual after snapping)
  recovery      = 1 - snap_err/raw_off_err            (fraction snap recovers)
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from vector_quantize_pytorch import FSQ
import onnxruntime as ort

HF = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/"
    "snapshots/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f"
)
DECODER = os.path.join(HF, "model_decoder.onnx")

NUM_TOKENS = 2
TOKEN_DIM = 32
TOKEN_TOTAL = NUM_TOKENS * TOKEN_DIM  # 64
OBS_DIM = 994
STEP = 1.0 / (TOKEN_DIM // 2)  # FSQ spacing = 1/half_width = 1/16 = 0.0625
FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]
B = 512
SEED = 0


HALF_WIDTH = TOKEN_DIM // 2  # 16


def build_fsq(device):
    return FSQ(levels=[TOKEN_DIM] * TOKEN_DIM).eval().to(device)


def snap_to_lattice(v):
    """Nearest-codebook projection for normalized FSQ tokens.

    FSQ lattice values are k/HALF_WIDTH with k in [-HALF_WIDTH, HALF_WIDTH-1]
    (verified: min=-1.0, max=0.9375, step=1/16). NOTE: we do NOT re-run
    fsq.quantize() here -- quantize() applies a tanh bound and is therefore
    NOT idempotent on already-normalized lattice values (it re-compresses),
    so re-quantizing a clean token does not return the clean token. Direct
    rounding is the true nearest-lattice snap for the deployed token space.
    """
    k = torch.round(v * HALF_WIDTH)
    k = torch.clamp(k, -HALF_WIDTH, HALF_WIDTH - 1)
    return k / HALF_WIDTH


def decode_batch(sess, in_name, token_states, base_obs):
    """Decode a batch of (B,64) token_state vectors through the ONNX decoder.

    token_states : np.ndarray (B,64) float32
    base_obs     : np.ndarray (994,) float32 -- fixed baseline for non-token dims
    returns      : np.ndarray (B,29) actions
    """
    outs = []
    for i in range(token_states.shape[0]):
        obs = base_obs.copy()
        obs[:TOKEN_TOTAL] = token_states[i]
        a = sess.run(None, {in_name: obs[None, :].astype(np.float32)})[0]
        outs.append(a[0])
    return np.stack(outs, axis=0)


def l2(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean())


def per_dim_mse(a, b):
    return float(((a - b) ** 2).mean())


def max_abs(a, b):
    return float(np.abs(a - b).max())


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fsq = build_fsq(device)

    # ONNX decoder (CPU build of onnxruntime -> CPUExecutionProvider).
    sess = ort.InferenceSession(DECODER, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    prov = sess.get_providers()

    # --- sample continuous latents & the clean teacher (on-lattice) tokens ---
    z = torch.randn(B, NUM_TOKENS, TOKEN_DIM, device=device)
    t_clean = fsq.quantize(z)  # on lattice, values in [-1, 0.9375], step 0.0625
    lattice_min = float(t_clean.min())
    lattice_max = float(t_clean.max())

    t_clean_flat = t_clean.reshape(B, TOKEN_TOTAL).detach().cpu().numpy()

    # Fixed realistic baseline for the 930 non-token obs dims. These come from
    # robot proprioception at deploy; we hold them constant (small gaussian) so
    # that ONLY the token off-lattice perturbation drives the measured deltas.
    rng = np.random.default_rng(SEED)
    base_obs = np.zeros(OBS_DIM, dtype=np.float32)
    base_obs[TOKEN_TOTAL:] = rng.normal(0.0, 0.1, size=OBS_DIM - TOKEN_TOTAL).astype(np.float32)

    dec_clean = decode_batch(sess, in_name, t_clean_flat, base_obs)

    # sanity: dynamic range of the clean decode itself (action units)
    clean_norm = float(np.linalg.norm(dec_clean, axis=-1).mean())
    clean_std = float(dec_clean.std())

    rows = []
    for frac in FRACS:
        a = frac * STEP
        if a == 0.0:
            t_off = t_clean.clone()
        else:
            noise = (torch.rand_like(t_clean) * 2 - 1) * a  # U(-a, a)
            t_off = t_clean + noise
        t_snap = snap_to_lattice(t_off)  # nearest-codebook projection (direct round)

        t_off_flat = t_off.reshape(B, TOKEN_TOTAL).detach().cpu().numpy()
        t_snap_flat = t_snap.reshape(B, TOKEN_TOTAL).detach().cpu().numpy()

        # how far off-lattice the token itself is, and does snap restore clean?
        tok_off_dist = float(
            np.linalg.norm((t_off_flat - t_clean_flat), axis=-1).mean()
        )
        snap_restores_token = bool(np.allclose(t_snap_flat, t_clean_flat, atol=1e-4))

        dec_off = decode_batch(sess, in_name, t_off_flat, base_obs)
        dec_snap = decode_batch(sess, in_name, t_snap_flat, base_obs)

        raw_l2 = l2(dec_off, dec_clean)
        snap_l2 = l2(dec_snap, dec_clean)
        recovery = (1.0 - snap_l2 / raw_l2) if raw_l2 > 1e-9 else float("nan")

        rows.append(
            dict(
                frac=frac,
                noise_amp=a,
                token_off_dist=tok_off_dist,
                snap_restores_clean_token=snap_restores_token,
                raw_off_l2=raw_l2,
                raw_off_perdim_mse=per_dim_mse(dec_off, dec_clean),
                raw_off_maxabs=max_abs(dec_off, dec_clean),
                snap_l2=snap_l2,
                snap_perdim_mse=per_dim_mse(dec_snap, dec_clean),
                snap_maxabs=max_abs(dec_snap, dec_clean),
                recovery_frac=recovery,
                raw_off_l2_rel_clean=raw_l2 / clean_norm if clean_norm > 0 else float("nan"),
            )
        )

    result = dict(
        meta=dict(
            device_fsq=device,
            onnx_providers=prov,
            batch=B,
            seed=SEED,
            num_tokens=NUM_TOKENS,
            token_dim=TOKEN_DIM,
            token_total=TOKEN_TOTAL,
            obs_dim=OBS_DIM,
            fsq_step=STEP,
            lattice_min=lattice_min,
            lattice_max=lattice_max,
            magnitude_guard=1.25,
            clean_decode_meanL2=clean_norm,
            clean_decode_std=clean_std,
            decoder_path=DECODER,
        ),
        table=rows,
    )
    print(json.dumps(result, indent=2))

    # pretty table
    print("\n" + "=" * 100)
    print(
        f"{'frac_step':>9} {'noise_amp':>9} {'tok_off':>8} "
        f"{'raw_L2':>9} {'raw_rel':>8} {'snap_L2':>9} {'recovery':>9} {'snap==clean':>11}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r['frac']:>9.2f} {r['noise_amp']:>9.4f} {r['token_off_dist']:>8.4f} "
            f"{r['raw_off_l2']:>9.5f} {r['raw_off_l2_rel_clean']:>8.4f} "
            f"{r['snap_l2']:>9.5f} {r['recovery_frac']:>9.4f} "
            f"{str(r['snap_restores_clean_token']):>11}"
        )
    print("=" * 100)
    print(f"clean decode mean-L2 (action-norm scale) = {clean_norm:.5f}, std = {clean_std:.5f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
