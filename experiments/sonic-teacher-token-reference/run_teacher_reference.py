#!/usr/bin/env python3
"""SONIC-teacher token distribution reference.

Runs the REAL cached SONIC encoder (model_encoder.onnx, obs_dict[1,1762] ->
encoded_tokens[1,64]) on REAL cached sample motion data and characterizes where
the encoded 64-d tokens sit relative to the FSQ(levels=[32]*32) lattice.

KEY ARCHITECTURAL FACT (verified by reading gear_sonic source):
  gear_sonic/trl/modules/universal_token_modules.py :: UniversalTokenModule.encode()
  returns (encoded_tokens, latent) where
      quantized_codes, _ = self.quantizer(latent)   # FSQ(levels=[32]*32)
      encoded_tokens = quantized_codes.contiguous()
  i.e. the FSQ quantizer runs INSIDE the exported ONNX graph, so the encoder's
  'encoded_tokens' output is POST-quantization -> it lands exactly on the FSQ
  lattice for ANY input. This script confirms that empirically on real motion.

Run with the sim venv:
  /workspace/GR00T-WholeBodyControl/.venv_sim/bin/python run_teacher_reference.py
"""
import json
import os
import glob
import numpy as np

ENC_PATH = "/workspace/hf-cache/wbc-checkpoints/policy/release/model_encoder.onnx"
SAMPLE_ROOT = "/workspace/hf-cache/wbc-checkpoints/sample_data"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# FSQ lattice geometry [verified Exp1 offlattice-fsq-decode-error]:
#   FSQ(levels=[32]*32); code value = k/16 for integer k in [-16, 15]
#   min=-1.0, max=0.9375, STEP=1/16=0.0625, half-step=0.03125
STEP = 1.0 / 16.0
HALF_STEP = STEP / 2.0
K_MIN, K_MAX = -16, 15


def nearest_lattice(v):
    """Round to nearest FSQ lattice point (k/16 clamped to k in [-16,15])."""
    k = np.round(v * 16.0)
    k = np.clip(k, K_MIN, K_MAX)
    return k / 16.0


def dist_in_steps(v):
    """|v - nearest_lattice(v)| expressed in STEP units."""
    return np.abs(v - nearest_lattice(v)) / STEP


def summarize(dists_steps, label):
    d = dists_steps.reshape(-1)
    return {
        "label": label,
        "n_values": int(d.size),
        "mean_steps": float(d.mean()),
        "median_steps": float(np.median(d)),
        "p90_steps": float(np.percentile(d, 90)),
        "p95_steps": float(np.percentile(d, 95)),
        "p99_steps": float(np.percentile(d, 99)),
        "max_steps": float(d.max()),
        "frac_beyond_half_step": float((d > 0.5).mean()),
    }


def load_all_motions():
    """Load every real sample_data pkl via joblib; return dict of arrays."""
    import joblib
    motions = {}
    for pkl in sorted(glob.glob(os.path.join(SAMPLE_ROOT, "**", "*.pkl"), recursive=True)):
        rel = os.path.relpath(pkl, SAMPLE_ROOT)
        d = joblib.load(pkl)
        # each pkl is {motion_name: {field: array}}
        for mname, mdata in d.items():
            if isinstance(mdata, dict):
                motions[f"{rel}::{mname}"] = {
                    k: np.asarray(v) for k, v in mdata.items()
                    if hasattr(v, "shape")
                }
    return motions


def build_real_feature_matrix(smpl_pkl, robot_pkl):
    """Assemble a per-frame REAL motion feature matrix [T, F].

    We concatenate the real motion fields at their real magnitudes:
      - SMPL joints, root-relative (smpl encoder territory)  [24*3=72]
      - SMPL root orientation as 6D rotation                 [6]
      - robot DOF (29 joints)                                [29]
      - robot pose_aa flattened                              [30*3=90]
      - robot root_rot quat                                  [4]
      - smpl_joints from robot pkl, root-relative            [24*3=72]
    NOTE: these are REAL motion values but NOT placed at the exact per-term byte
    offsets of the 1762-d obs_dict -- that mapping requires a live IsaacLab
    command_manager (body-local transforms vs. the robot's runtime sim pose) and
    is not reconstructable from the pkl alone (see RESULTS.md 'Real vs baseline').
    The output-geometry conclusion (on-lattice) is input-invariant, so real
    magnitudes are what matter here.
    """
    import joblib
    smpl = joblib.load(smpl_pkl)
    robot = joblib.load(robot_pkl)

    def unwrap(d):
        # robot/soma pkls nest fields under a motion-name key; smpl pkls are flat.
        if "smpl_joints" in d or "dof" in d:
            return d
        return list(d.values())[0]
    smpl_m = unwrap(smpl)
    robot_m = unwrap(robot)

    T = min(smpl_m["smpl_joints"].shape[0], robot_m["dof"].shape[0])

    def root_relative(joints, root_transl):
        # joints: [T,J,3]; subtract root (joint 0) to canonicalize translation
        return joints[:T] - joints[:T, 0:1, :]

    sj = root_relative(smpl_m["smpl_joints"], None).reshape(T, -1)          # 72
    # 6D rot proxy from smpl root quat (wxyz) -> first two columns of rot matrix
    q = smpl_m["pose_aa"][:T].reshape(T, -1, 3)[:, 0, :]  # root axis-angle [T,3]
    # build a crude 6D from axis-angle magnitude (real values, real magnitude)
    sixd = np.concatenate([np.cos(q), np.sin(q)], axis=-1)                  # 6
    dof = robot_m["dof"][:T]                                                # 29
    pose_aa = robot_m["pose_aa"][:T].reshape(T, -1)                         # 90
    root_rot = robot_m["root_rot"][:T]                                      # 4
    rsj = root_relative(robot_m["smpl_joints"], None).reshape(T, -1)        # 72

    feats = np.concatenate([sj, sixd, dof, pose_aa, root_rot, rsj], axis=-1).astype(np.float32)
    return feats, T


def make_obs_1762(feat_row, encoder_index):
    """Place a real feature row into a 1762-d obs vector.

    dim0 = encoder_index (scalar; 0=g1, 1=teleop, 2=smpl -- measured from ONNX
    ScatterND valid-index probe). Real motion features are tiled to fill dims
    [1:1762]; unused tail is zero-padded. This is a REAL-magnitude input, layout
    -approximate (see build_real_feature_matrix docstring).
    """
    obs = np.zeros((1, 1762), np.float32)
    obs[0, 0] = float(encoder_index)
    body = feat_row.reshape(-1)
    n = min(body.size, 1761)
    obs[0, 1:1 + n] = body[:n]
    return obs


def main():
    import onnxruntime as ort
    from vector_quantize_pytorch import FSQ
    import torch

    sess = ort.InferenceSession(ENC_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    assert sess.get_inputs()[0].shape == [1, 1762]
    assert sess.get_outputs()[0].shape == [1, 64]

    results = {
        "encoder_path": ENC_PATH,
        "encoder_io": {"input": [in_name, [1, 1762]], "output": ["encoded_tokens", [1, 64]]},
        "lattice": {"levels": "[32]*32", "step": STEP, "half_step": HALF_STEP,
                    "min": K_MIN / 16.0, "max": K_MAX / 16.0},
        "runs": {},
    }

    # ---- 1. REAL MOTION run --------------------------------------------------
    smpl_pkl = glob.glob(os.path.join(
        SAMPLE_ROOT, "smpl_filtered", "**", "walk_forward_amateur_001__A001.pkl"),
        recursive=True)[0]
    robot_pkl = glob.glob(os.path.join(
        SAMPLE_ROOT, "robot_filtered", "**", "walk_forward_amateur_001__A001.pkl"),
        recursive=True)[0]
    feats, T = build_real_feature_matrix(smpl_pkl, robot_pkl)
    results["real_motion_input"] = {
        "smpl_pkl": smpl_pkl, "robot_pkl": robot_pkl,
        "n_frames_available": int(T), "feature_dim": int(feats.shape[1]),
    }

    all_tokens = []
    for enc_idx in (0, 1, 2):  # g1, teleop, smpl encoder branches
        toks = []
        for t in range(T):
            obs = make_obs_1762(feats[t], enc_idx)
            y = sess.run(None, {in_name: obs})[0]  # [1,64]
            toks.append(y[0])
        toks = np.asarray(toks)  # [T,64]
        all_tokens.append(toks)
        d = dist_in_steps(toks)
        results["runs"][f"real_motion_enc{enc_idx}"] = summarize(d, f"real_motion_enc{enc_idx}")
        results["runs"][f"real_motion_enc{enc_idx}"].update({
            "token_min": float(toks.min()), "token_max": float(toks.max()),
        })
    real_stack = np.concatenate(all_tokens, axis=0)  # [3T,64]
    results["runs"]["real_motion_all"] = summarize(dist_in_steps(real_stack), "real_motion_all")
    results["runs"]["real_motion_all"].update({
        "token_min": float(real_stack.min()), "token_max": float(real_stack.max()),
        "n_tokens": int(real_stack.shape[0]),
    })

    # ---- 2. INPUT-INVARIANCE battery (synthetic distributions) ---------------
    rng = np.random.default_rng(0)
    battery = {
        "zeros": np.zeros((256, 1762), np.float32),
        "randn_1": rng.standard_normal((256, 1762)).astype(np.float32),
        "randn_3": (3.0 * rng.standard_normal((256, 1762))).astype(np.float32),
        "uniform_pm5": rng.uniform(-5, 5, (256, 1762)).astype(np.float32),
    }
    for name, X in battery.items():
        X[:, 0] = 2.0  # valid encoder_index (smpl); keep scatter index in-range
        toks = np.asarray([sess.run(None, {in_name: X[i:i+1]})[0][0] for i in range(X.shape[0])])
        results["runs"][f"synthetic_{name}"] = summarize(dist_in_steps(toks), f"synthetic_{name}")
        results["runs"][f"synthetic_{name}"].update({
            "token_min": float(toks.min()), "token_max": float(toks.max()),
        })

    # ---- 3. Cross-check against the REAL FSQ(levels=[32]*32) -----------------
    # (a) The decisive test that the encoder output is on-lattice: round each
    #     value to nearest k/16 (k in [-16,15]) and measure residual. Done above
    #     for every run (max_steps == 0.0). Recorded here as the headline check.
    # (b) Confirm the REAL FSQ maps genuine CONTINUOUS pre-quant latents exactly
    #     onto the same lattice. FSQ.forward() applies an internal bound to its
    #     INPUT, so it must be fed raw continuous latents (NOT values already in
    #     [-1,0.9375]); re-feeding quantized values is not an idempotence check.
    fsq = FSQ(levels=[32] * 32)
    rng2 = np.random.default_rng(7)
    cont = (5.0 * rng2.standard_normal((200, 2, 32))).astype(np.float32)
    qc, _ = fsq(torch.from_numpy(cont))
    qc = qc.detach().numpy()
    results["fsq_cross_check"] = {
        "note": ("real FSQ(levels=[32]*32).quantize applied to genuine continuous "
                 "pre-quant latents lands exactly on lattice; this is the same "
                 "operation baked into the ONNX encoder graph"),
        "quantized_continuous_max_dist_steps": float(dist_in_steps(qc).max()),
        "quantized_continuous_min": float(qc.min()),
        "quantized_continuous_max": float(qc.max()),
        "encoder_output_on_lattice_max_dist_steps":
            float(dist_in_steps(real_stack).max()),
        "encoder_output_on_lattice":
            bool(dist_in_steps(real_stack).max() < 1e-9),
    }

    with open(os.path.join(OUT_DIR, "reference_distribution.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
