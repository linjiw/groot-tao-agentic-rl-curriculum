# Experiment ③ — Trained-VLA motion-token distance-to-FSQ-lattice distribution

**Open follow-up from Exp① (`experiments/offlattice-fsq-decode-error/RESULTS.md`).**
Exp① showed snap-to-lattice recovers the teacher token *perfectly* within ±½
quantization step (±0.03125 normalized), and flips to the wrong codeword beyond
it. This experiment was to MEASURE where a real trained GR00T VLA's emitted
motion tokens actually land relative to that ±½-step recovery basin — to decide
whether snap-to-lattice is safe in production or whether quantization-aware BC /
a soft-quantization head is warranted.

---

## 0. FEASIBILITY GATE — reported FIRST  →  **NO-GO** [verified]

**The cached `GR00T-N1.7-3B` does NOT emit the 64-d `unitree_g1_sonic` FSQ
motion token.** The `unitree_g1_sonic` embodiment is **absent from every piece
of this checkpoint's own metadata**. The generic pretrained checkpoint was not
finetuned on the sonic motion-token embodiment, so driving forward passes and
measuring lattice distance on its output would be **measuring an unrelated
132-d generic action**, not the FSQ motion token. Per the task's critical
honesty gate, the measurement pass is **not run** and no distribution is
fabricated. A sonic-specific finetuned checkpoint is required (see §4).

This gate outcome is produced by `check_feasibility.py` and captured in
`results.json`. All four registration checks are **False**:

| Check (in cached `models--nvidia--GR00T-N1.7-3B/`) | `unitree_g1_sonic` present? |
|---|:--:|
| `statistics.json` (per-embodiment normalization stats) | ❌ False |
| `experiment_cfg/dataset_statistics.json` | ❌ False |
| `embodiment_id.json` (tag → embodiment-slot map) | ❌ False |
| `experiment_cfg/final_processor_config.json` (processor embodiment-id map) | ❌ False (substring count = 0) |

---

## 1. Evidence (file:line) [verified]

### 1a. The sonic embodiment *schema* EXISTS in gr00t source (code, not checkpoint)
- `external/Isaac-GR00T` == `/workspace/Isaac-GR00T` @ ab88b50 (submodule pin).
- `gr00t/configs/data/embodiment_configs.py:67-113` — `"unitree_g1_sonic"`
  modality config: `action` modality_keys = `["motion_token",
  "left_hand_joints", "right_hand_joints"]`, `delta_indices=list(range(40))`
  (action_horizon 40). **This is the schema the sonic embodiment WOULD use if a
  checkpoint were trained on it.** [verified]
- `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:76` — projector index **11**
  reserved for `unitree_g1_sonic`. [verified]
- `gr00t/data/embodiment_tags.py:107` — `UNITREE_G1_SONIC = "unitree_g1_sonic"`
  enum member. [verified]

The *existence of the schema in code* does NOT imply the *cached checkpoint*
trained it. It only means gr00t knows how to consume sonic data IF a matching
checkpoint + statistics are provided.

### 1b. The cached checkpoint's embodiment roster — sonic is NOT in it
`models--nvidia--GR00T-N1.7-3B/embodiment_id.json` and
`experiment_cfg/final_processor_config.json` (embodiment-id map) enumerate the
embodiments this checkpoint actually has slots/stats for. The G1-latent-style
tags present are:
- `unitree_g1_whole_body_teleop_latent` → 9
- `unitree_g1_whole_body_teleop_smpl` → 16
- `unitree_g1_full_body_with_waist_height_nav_cmd` → 25

`unitree_g1_sonic` appears **0 times**. `motion_token` appears **0 times** in
the processor config. [verified — `check_feasibility.py`]

`dataset_statistics.json` / `statistics.json` contain per-embodiment
normalization stats for only 8 embodiments (all `xdof_*`, `oxe_droid_*`,
`real_g1_*`, `real_r1_pro_sharpa_*`) — no sonic. Without normalization stats,
gr00t's processor cannot even build the sonic transform. [verified]

### 1c. The action head is a GENERIC 132-d flow-matching decoder, not a 64-d token head
Read directly from the checkpoint safetensors (via `check_feasibility.py`):

| Weight | Shape | Meaning |
|---|---|---|
| `action_head.action_decoder.layer2.W` | `[32, 1024, 132]` | 32 embodiment slots × hidden 1024 → **132-d** generic action |
| `action_head.action_decoder.layer2.b` | `[32, 132]` | per-embodiment 132-d bias |
| `action_head.action_encoder.W1.W` | `[32, 132, 1536]` | encodes a **132-d** action, not a 64-d token |

- `action_head.py` uses `CategorySpecificLinear`
  (`gr00t/model/modules/embodiment_conditioned_mlp.py:59-79`): per-embodiment
  weight slice `self.W[cat_ids]`. Output dim = `config.max_action_dim = 132`
  (`config.json:42`), NOT 64. [verified]
- The decoder emits a **132-d** vector per action step for whichever of its
  registered embodiment slots is selected. For a sonic checkpoint the head's
  output for slot 11 would carry the 64-d motion token (+ hand joints) in the
  first channels — but slot 11 / sonic stats do not exist here. [verified]

### 1d. The deployment path confirms sonic is the intended-but-separate target
- `gear_sonic/scripts/run_vla_inference.py:110` — `embodiment_tag =
  "unitree_g1_sonic"` (the default the *deployed* runner expects). [verified]
- The runner talks to an Isaac-GR00T **PolicyServer** (`run_vla_inference.py`
  header + config lines 64-76) and pulls `action["motion_token"]`
  (`:290, :691-717`), a `[horizon, 64]` chunk. That server must be backed by a
  **sonic-finetuned** checkpoint that has embodiment slot 11 + sonic stats —
  which the cached generic `GR00T-N1.7-3B` is not. [verified]

---

## 2. Why measuring this checkpoint would be meaningless [verified reasoning]

To get a meaningful "emitted motion token", the model must (a) accept the sonic
observation schema, (b) select the sonic embodiment slot, and (c) decode to the
motion-token action space with the sonic normalization applied. This checkpoint
fails (a)–(c): no sonic modality transform (no stats), no sonic embodiment id,
no sonic-trained decoder slot. Forcing a different embodiment id (e.g. 9 or 25)
and reading its 132-d output would produce numbers for a *different* action
space (whole-body teleop / full-body nav), whose leading 64 channels are **not**
FSQ motion tokens and have no relationship to the FSQ lattice. Any lattice
distance computed on them would be an artifact, not a measurement of the
quantity Exp③ asks about. Hence: **no distribution is produced.** [verified]

---

## 3. VERDICT

- **Feasibility gate:** **NO-GO** [verified]. Cached `GR00T-N1.7-3B` does not
  emit the sonic FSQ motion token; sonic embodiment is unregistered in the
  checkpoint (stats + id map + processor all absent).
- **Distance-to-lattice distribution:** **NOT MEASURED** — deliberately not
  fabricated. Reporting the blocker with evidence is the deliverable, per the
  task's honesty gate.
- **Tie-back to Exp① / risk #1:** Exp① already downgraded risk #1 (snap
  recovers perfectly within ±½ step; off-lattice decode is smooth/bounded). This
  experiment **cannot further confirm or refute** where a *trained sonic VLA's*
  tokens land, because the required sonic checkpoint is not available locally.
  Risk #1 therefore **stays at Exp①'s downgraded status** — the "systematic-bias
  > ½-step" failure mode remains **open and unmeasured** [verified], neither
  confirmed nor ruled out. Snap-to-lattice's production safety is **not yet
  empirically validated on real trained-VLA outputs**. [speculative]

---

## 4. What WOULD be required to run the core measurement [verified/speculative]

1. **A sonic-finetuned GR00T checkpoint** with embodiment slot 11
   (`unitree_g1_sonic`) trained, including:
   - `unitree_g1_sonic` present in `embodiment_id.json` +
     `dataset_statistics.json` + processor config (normalization stats for
     `motion_token`, `left_hand_joints`, `right_hand_joints`). [verified need]
   - `action_head.action_decoder` slot 11 trained to emit the motion-token
     action space. [verified need]
   This is the checkpoint the deployed `run_vla_inference.py` PolicyServer
   expects; it is **not** the generic `nvidia/GR00T-N1.7-3B` cached here.
2. Then: load via `Gr00tPolicy` / PolicyServer with `embodiment_tag=
   "unitree_g1_sonic"`, feed the sample state from
   `/workspace/hf-cache/wbc-checkpoints/sample_data/*.pkl`, a fixed prompt
   ("walk forward"), and a zero/black `ego_view` frame if required; run several
   flow-matching passes; collect ≥ a few hundred `[40, 64]` motion-token chunks;
   compute per-dim distance to `round(v*16)/16` clamped to `[-16,15]/16` in step
   units (step 0.0625); report mean/median/p90/p95/p99/max and the fraction of
   dims / tokens beyond ±½ step, plus fraction outside `[-1.0, 0.9375]`. [the
   intended method — ready to run once a sonic checkpoint exists]
3. Alternatively, if only the SONIC **encoder** ONNX
   (`wbc-checkpoints/policy/release/model_encoder.onnx`, the on-lattice teacher)
   is available, one could measure teacher-token on-lattice-ness — but that is
   NOT the VLA's *emitted* distribution Exp③ targets. [speculative]

---

## 5. Reproduce

```bash
cd experiments/vla-token-lattice-distance
HF_HOME=/workspace/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /workspace/Isaac-GR00T/.venv/bin/python check_feasibility.py
# prints FEASIBLE: False and writes results.json
```

- **Venv:** `/workspace/Isaac-GR00T/.venv/bin/python` (torch 2.7.1+cu128,
  transformers 4.57.3, safetensors available; A10G 23GB).
- **Env vars:** `HF_HOME=/workspace/hf-cache` (the `~/.cache` copy is a stub),
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (network is gated HTTP 401).
- **Checkpoint inspected:** `/workspace/hf-cache/models--nvidia--GR00T-N1.7-3B/`
  (real 2-shard safetensors).
- No model weights were loaded onto GPU (the gate resolves from metadata +
  safetensors headers alone — no forward pass was justified once the embodiment
  was proven absent). No submodule/source tree was modified. No git commit.

### Label key
`[verified]` = read directly from checkpoint metadata / gr00t source / live
safetensors inspection in this experiment. `[measured]` = numeric result from a
forward pass (**none produced here — gate is NO-GO**). `[speculative]` =
reasoned implication.
