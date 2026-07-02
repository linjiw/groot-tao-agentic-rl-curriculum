# SPDX-License-Identifier: Apache-2.0
"""Offline CPU render of a SONIC TrajectoryRecorder .pkl with MuJoCo.

Fallback path C from RENDER_TODO.md: the box's Isaac Sim RTX render path
segfaults at app startup (infra-level, see RESULTS.md §4), so videos are
produced by kinematic replay instead — no RTX, no GPU:

  TrajectoryRecorder pkl (dof_pos in IsaacLab order, root pose wxyz)
    → reorder joints by NAME to the MJCF's qpos order
    → mj_forward per frame (kinematics only, no dynamics)
    → offscreen mujoco.Renderer (MUJOCO_GL=osmesa|egl) → mp4 via imageio.

Joint-order note: dof_pos comes from `robot.data.joint_pos` (recorders.py:
273-278), i.e. IsaacLab's breadth-first order — G1_ISAACLab_ORDER in
gear_sonic/envs/env_utils/joint_utils.py:11-40 [verified: programmatic
compare against the pinned submodule, 29/29 names match]. The MJCF
(g1_29dof_rev_1_0.xml) lays out qpos in depth-first tree order. Mapping is
by joint NAME, never by index.

Run inside the isaac-lab-base container with the kit python
(/isaac-sim/kit/python/bin/python3 — has mujoco 3.10 + imageio installed):

  MUJOCO_GL=osmesa /isaac-sim/kit/python/bin/python3 render_trajectory_mujoco.py \
      --traj-dir /workspace/wbc-training-logs/diagnosis/traj_baseline \
      --mjcf /workspace/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml \
      --out-dir /workspace/wbc-training-logs/diagnosis/videos_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

# IsaacLab breadth-first joint order for the pkl's dof_pos columns
# [verified: gear_sonic/envs/env_utils/joint_utils.py:10-40]
G1_ISAACLAB_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def render_one(traj_path: str, model, renderer, mujoco, fps_override=None,
               camera_distance=3.0):
    with open(traj_path, "rb") as f:
        d = pickle.load(f)
    dof = np.asarray(d["dof_pos"])            # (T, 29) IsaacLab order
    root_pos = np.asarray(d["root_pos_w"])    # (T, 3)
    root_quat = np.asarray(d["root_quat_w"])  # (T, 4) wxyz
    fps = float(fps_override or d.get("fps", 25.0))
    quat_format = str(d.get("quat_format", "wxyz"))
    if quat_format != "wxyz":
        raise ValueError(f"unexpected quat_format {quat_format!r}")

    # name-based column mapping pkl -> mjcf qpos
    jmap = []
    for name in G1_ISAACLAB_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not in MJCF")
        jmap.append(model.jnt_qposadr[jid])
    jmap = np.asarray(jmap)

    data = mujoco.MjData(model)
    frames = []
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = camera_distance
    cam.elevation = -15.0
    cam.azimuth = 135.0

    for t in range(dof.shape[0]):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_quat[t]  # mujoco free joint is wxyz too
        data.qpos[jmap] = dof[t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = root_pos[t]    # follow the root
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    return frames, fps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="offline MuJoCo render of trajectory pkls")
    p.add_argument("--traj-dir", required=True)
    p.add_argument("--mjcf", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-frames", type=int, default=0,
                   help="0 = all frames")
    args = p.parse_args(argv)

    import imageio
    import mujoco

    os.makedirs(args.out_dir, exist_ok=True)
    # raise the offscreen framebuffer via spec (the shipped MJCF has no
    # <visual><global> clause, default 640x480 < our render size)
    spec = mujoco.MjSpec.from_file(args.mjcf)
    spec.visual.global_.offwidth = max(args.width, 640)
    spec.visual.global_.offheight = max(args.height, 480)
    model = spec.compile()
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("MJCF root joint is not free — qpos layout assumption broken")
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    meta_path = os.path.join(args.traj_dir, "scene_metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    rendered = []
    for fname in sorted(os.listdir(args.traj_dir)):
        if not fname.endswith(".trajectory.pkl"):
            continue
        frames, fps = render_one(os.path.join(args.traj_dir, fname),
                                 model, renderer, mujoco)
        if args.max_frames:
            frames = frames[: args.max_frames]
        out = os.path.join(args.out_dir, fname.replace(".trajectory.pkl", ".mp4"))
        imageio.mimwrite(out, frames, fps=fps, quality=7)
        rendered.append((fname, len(frames), out))
        print(f"{fname}: {len(frames)} frames @ {fps} fps -> {out}")

    if not rendered:
        print(f"no .trajectory.pkl files in {args.traj_dir}", file=sys.stderr)
        return 1
    if meta:
        print("scene_metadata:", json.dumps({k: v.get("num_frames") for k, v in meta.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
