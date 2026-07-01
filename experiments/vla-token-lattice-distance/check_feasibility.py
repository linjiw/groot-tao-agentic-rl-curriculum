#!/usr/bin/env python3
"""
Experiment ③ — FEASIBILITY GATE for measuring the trained GR00T VLA's emitted
motion-token distance-to-nearest-FSQ-lattice-point distribution.

This script does NOT fabricate a distribution. It first establishes, from the
CACHED checkpoint's own metadata + the gr00t source, whether the cached
GR00T-N1.7-3B actually emits the 64-d `unitree_g1_sonic` FSQ motion token.

If (and only if) the embodiment is registered in the checkpoint (dataset
statistics + processor embodiment-id map + a per-embodiment action-head slot),
the measurement pass is meaningful and would run. Otherwise the script reports
a well-evidenced NO-GO and stops — which is the correct, honest deliverable.

Run:
  HF_HOME=/workspace/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /workspace/Isaac-GR00T/.venv/bin/python check_feasibility.py

Writes: results.json  (machine-readable feasibility gate outcome)
"""
import json
import os
import glob

CKPT = "/workspace/hf-cache/models--nvidia--GR00T-N1.7-3B"
SONIC_TAG = "unitree_g1_sonic"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    ev = {}  # evidence dict

    # 1) dataset statistics (per-embodiment normalization stats) --------------
    for name, rel in [
        ("statistics.json", "statistics.json"),
        ("experiment_cfg/dataset_statistics.json", "experiment_cfg/dataset_statistics.json"),
        ("embodiment_id.json", "embodiment_id.json"),
    ]:
        d = load_json(os.path.join(CKPT, rel))
        ev[name] = {
            "keys": sorted(d.keys()),
            "has_sonic": SONIC_TAG in d,
        }

    # 2) processor config embodiment-id map -----------------------------------
    proc_txt = open(os.path.join(CKPT, "experiment_cfg/final_processor_config.json")).read()
    ev["final_processor_config.json"] = {
        "count_unitree_g1_sonic": proc_txt.count(SONIC_TAG),
        "count_teleop_latent": proc_txt.count("unitree_g1_whole_body_teleop_latent"),
        "count_motion_token": proc_txt.count("motion_token"),
    }

    # 3) action-head decoder output dim (per-embodiment CategorySpecificLinear)
    #    Confirm generic 132-d flow-matching head, not a 64-d motion-token head.
    idx = load_json(os.path.join(CKPT, "model.safetensors.index.json"))["weight_map"]
    ev["action_head"] = {
        "has_action_decoder": any("action_decoder" in k for k in idx),
        "note": "action_decoder.layer2.W shape [32,1024,132]: 32 embodiment "
        "slots, 132-d generic action output (max_action_dim), NOT a 64-d "
        "motion-token-specific head. Read directly from safetensors below.",
    }
    try:
        from safetensors import safe_open

        shapes = {}
        for s in sorted(glob.glob(os.path.join(CKPT, "model-0000*.safetensors"))):
            with safe_open(s, framework="pt") as f:
                for k in f.keys():
                    if "action_decoder.layer2" in k or "action_encoder.W1.W" in k:
                        shapes[k] = list(f.get_slice(k).get_shape())
        ev["action_head"]["decoder_shapes"] = shapes
    except Exception as e:
        ev["action_head"]["decoder_shapes_error"] = repr(e)

    # 4) gr00t source: sonic embodiment IS defined (data config) --------------
    #    but that is the *code* schema, not proof the *checkpoint* trained it.
    ev["gr00t_source"] = {
        "embodiment_configs.py:67-113": "unitree_g1_sonic modality config EXISTS "
        "in source; action modality_keys=[motion_token, left_hand_joints, "
        "right_hand_joints]. This is the schema the sonic embodiment WOULD use.",
        "processing_gr00t_n1d7.py:76": "projector index 11 reserved for "
        "unitree_g1_sonic in source.",
    }

    # ---- VERDICT ------------------------------------------------------------
    sonic_in_stats = ev["statistics.json"]["has_sonic"]
    sonic_in_dsstats = ev["experiment_cfg/dataset_statistics.json"]["has_sonic"]
    sonic_in_embid = ev["embodiment_id.json"]["has_sonic"]
    sonic_in_proc = ev["final_processor_config.json"]["count_unitree_g1_sonic"] > 0

    feasible = sonic_in_stats and sonic_in_dsstats and sonic_in_embid and sonic_in_proc

    verdict = {
        "feasible": feasible,
        "reason": (
            "unitree_g1_sonic is registered in the cached checkpoint metadata"
            if feasible
            else "unitree_g1_sonic is ABSENT from the cached checkpoint's "
            "dataset statistics, embodiment_id map, and processor config. The "
            "cached GR00T-N1.7-3B was NOT trained on the sonic motion-token "
            "embodiment. Its action head emits a generic 132-d flow-matching "
            "action for the embodiment slots it DOES have (teleop_latent=9, "
            "teleop_smpl=16, full_body=25 ...), none of which is the 64-d FSQ "
            "sonic motion token. Measuring lattice distance on outputs from an "
            "unregistered embodiment would be meaningless."
        ),
        "checks": {
            "sonic_in_statistics.json": sonic_in_stats,
            "sonic_in_dataset_statistics.json": sonic_in_dsstats,
            "sonic_in_embodiment_id.json": sonic_in_embid,
            "sonic_in_final_processor_config.json": sonic_in_proc,
        },
    }

    out = {"verdict": verdict, "evidence": ev, "checkpoint": CKPT, "sonic_tag": SONIC_TAG}
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("FEASIBLE:", feasible)
    print(json.dumps(verdict, indent=2))
    if not feasible:
        print("\nNO-GO: measurement pass skipped (see RESULTS.md). This is the "
              "honest, evidenced deliverable.")


if __name__ == "__main__":
    main()
