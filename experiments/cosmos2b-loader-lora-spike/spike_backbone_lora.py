#!/usr/bin/env python3
"""Experiment (2): Cosmos-Reason2-2B backbone load + LoRA attach + strict-loader wall.

Spike A: load GR00T Qwen3Backbone (pulls nvidia/Cosmos-Reason2-2B config), confirm
         select_layer truncation, report kept LLM layer count.
Spike B: attach a LoRA adapter (r=8, alpha=8, target=[q_proj,v_proj]) to the kept
         language_model layers; confirm adapters land ONLY on kept layers; count
         trainable params before/after.
Spike C: exercise GR00T's strict-key validator (gr00t_n1d7/setup.py lines ~97-120)
         against three state_dicts:
           (c0) plain backbone           -> baseline (expect PASS)
           (c1) LoRA-wrapped, NOT merged -> expect FAIL (lora_A/lora_B unexpected + base missing)
           (c2) merge_and_unload()       -> expect PASS iff keys byte-identical to plain

NOTE: nvidia/Cosmos-Reason2-2B has only config+tokenizer cached (no weights), and
network is gated (HTTP 401, no token). The backbone code path (qwen3_backbone.py:154-158)
loads CONFIG ONLY and instantiates random weights; real weights are overwritten later by
GR00T.from_pretrained. This spike is about state_dict KEYS/STRUCTURE, not weight values,
so config-only load is sufficient and faithful. We DO NOT run the full VLA reload (weights
not cached) — instead we replicate setup.py's exact validator logic against the key sets.
"""
import os
import sys
import json

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

RESULTS = {}


def banner(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Spike A: backbone load + select_layer truncation
# ---------------------------------------------------------------------------
banner("SPIKE A: Qwen3Backbone load (nvidia/Cosmos-Reason2-2B config)")

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
SELECT_LAYER = 12  # gr00t/configs/model/gr00t_n1d7.py:47

# Read full-model layer count from the cached config for reference.
from transformers import Qwen3VLForConditionalGeneration
cfg = Qwen3VLForConditionalGeneration.config_class.from_pretrained(MODEL_NAME, local_files_only=True)
full_layers = cfg.text_config.num_hidden_layers if hasattr(cfg, "text_config") else cfg.num_hidden_layers
print(f"[config] full num_hidden_layers (from config.json) = {full_layers}")

backbone = Qwen3Backbone(
    model_name=MODEL_NAME,
    tune_llm=False,
    tune_visual=False,
    select_layer=SELECT_LAYER,
    load_bf16=True,
    transformers_loading_kwargs={"local_files_only": True},
)
lm = backbone.model.language_model
kept = len(lm.layers)
print(f"[measured] select_layer={SELECT_LAYER}  kept language_model.layers = {kept}")
assert kept == SELECT_LAYER, f"expected {SELECT_LAYER} kept layers, got {kept}"
RESULTS["A_full_layers"] = int(full_layers)
RESULTS["A_kept_layers"] = int(kept)
RESULTS["A_pass"] = kept == SELECT_LAYER
print(f"[A] PASS: truncation kept exactly {kept} of {full_layers} layers")


# ---------------------------------------------------------------------------
# Spike B: LoRA attach on kept layers
# ---------------------------------------------------------------------------
banner("SPIKE B: LoRA attach (r=8, alpha=8, target=[q_proj,v_proj])")

from peft import LoraConfig, get_peft_model

def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def count_total(m):
    return sum(p.numel() for p in m.parameters())

# Make the LLM trainable so LoRA has grad-enabled base to wrap (peft freezes base anyway).
lm.requires_grad_(True)
trainable_before = count_trainable(lm)
total_before = count_total(lm)
print(f"[measured] language_model trainable params BEFORE LoRA = {trainable_before:,}")

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights=True,
)
peft_lm = get_peft_model(lm, lora_cfg)

trainable_after = count_trainable(peft_lm)
total_after = count_total(peft_lm)
print(f"[measured] trainable params AFTER LoRA = {trainable_after:,}  (total {total_after:,})")

# Which layer indices got adapters?
adapter_layers = set()
adapter_modules = []
for name, mod in peft_lm.named_modules():
    if name.endswith("lora_A") or ".lora_A." in name + ".":
        pass
for name, _ in peft_lm.named_parameters():
    if "lora_A" in name or "lora_B" in name:
        adapter_modules.append(name)
        # extract layer index: ...layers.<i>.self_attn.q_proj.lora_A...
        parts = name.split(".")
        if "layers" in parts:
            idx = parts[parts.index("layers") + 1]
            adapter_layers.add(int(idx))

max_idx = max(adapter_layers) if adapter_layers else -1
print(f"[measured] adapter-bearing layer indices: {sorted(adapter_layers)}")
print(f"[measured] max adapter layer index = {max_idx}  (kept layers 0..{kept-1})")
print(f"[measured] #lora param tensors = {len(adapter_modules)}")
print("[measured] sample adapter param names:")
for n in adapter_modules[:6]:
    print("   ", n)

only_on_kept = (max_idx <= kept - 1)
RESULTS["B_trainable_before"] = int(trainable_before)
RESULTS["B_trainable_after"] = int(trainable_after)
RESULTS["B_adapter_layer_indices"] = sorted(adapter_layers)
RESULTS["B_adapters_only_on_kept"] = bool(only_on_kept)
RESULTS["B_num_lora_tensors"] = len(adapter_modules)
RESULTS["B_pass"] = only_on_kept and trainable_after < trainable_before
print(f"[B] adapters only on kept layers: {only_on_kept}; "
      f"trainable dropped {trainable_before:,} -> {trainable_after:,}")


# ---------------------------------------------------------------------------
# Spike C: strict-loader wall
# ---------------------------------------------------------------------------
banner("SPIKE C: strict-key validator against plain / LoRA / merged state_dicts")

# Replicate setup.py:97-120 validator logic exactly. GR00T loads the FULL VLA and
# compares loaded checkpoint keys against the freshly-built model's keys. We model
# that by: reference model key set = PLAIN backbone keys (what a from-scratch build
# expects); checkpoint key set = the state_dict we would try to load.

def validate(reference_keys, checkpoint_keys):
    """Mirror gr00t_n1d7/setup.py strict-key check.

    missing_keys  = present in reference model but absent from checkpoint
    unexpected    = present in checkpoint but absent from reference model
    mask_token is the ONLY whitelisted missing key.
    Returns (passed, error_list).
    """
    missing_keys = sorted(reference_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - reference_keys)
    other_missing = [k for k in missing_keys if "mask_token" not in k]
    errors = []
    if other_missing:
        errors.append(f"Missing keys ({len(other_missing)}): {other_missing[:8]}"
                      + ("..." if len(other_missing) > 8 else ""))
    if unexpected_keys:
        errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:8]}"
                      + ("..." if len(unexpected_keys) > 8 else ""))
    return (len(errors) == 0), errors, len(other_missing), len(unexpected_keys)

# ---- Reference: a freshly-built PLAIN backbone's language_model keys ----
plain_backbone = Qwen3Backbone(
    model_name=MODEL_NAME,
    tune_llm=False, tune_visual=False,
    select_layer=SELECT_LAYER, load_bf16=True,
    transformers_loading_kwargs={"local_files_only": True},
)
plain_lm_keys = set(plain_backbone.model.language_model.state_dict().keys())
print(f"[measured] plain language_model state_dict keys = {len(plain_lm_keys)}")

# ---- (c0) plain vs plain -> baseline sanity ----
p0, e0, m0, u0 = validate(plain_lm_keys, set(plain_lm_keys))
print(f"[c0] plain-vs-plain: PASS={p0} missing={m0} unexpected={u0}")

# ---- (c1) LoRA-wrapped, NOT merged ----
lora_keys = set(peft_lm.state_dict().keys())
print(f"[measured] LoRA-wrapped state_dict keys = {len(lora_keys)}")
# peft renames base weights under base_model.model.* ; that alone will explode the diff.
p1, e1, m1, u1 = validate(plain_lm_keys, lora_keys)
print(f"[c1] plain-vs-LoRA(unmerged): PASS={p1} missing={m1} unexpected={u1}")
for e in e1:
    print("     ", e)
# Also show the raw lora_A/lora_B keys that appear as unexpected:
lora_only = sorted(k for k in (lora_keys - plain_lm_keys) if "lora_" in k)
print(f"[measured] lora_A/lora_B unexpected keys (sample): {lora_only[:4]}")

# ---- (c2) merge_and_unload ----
merged_lm = peft_lm.merge_and_unload()
merged_keys = set(merged_lm.state_dict().keys())
print(f"[measured] merged (merge_and_unload) state_dict keys = {len(merged_keys)}")
p2, e2, m2, u2 = validate(plain_lm_keys, merged_keys)
print(f"[c2] plain-vs-merged: PASS={p2} missing={m2} unexpected={u2}")
for e in e2:
    print("     ", e)

# Byte-for-key comparison: are the merged keys IDENTICAL to plain?
keys_identical = (merged_keys == plain_lm_keys)
print(f"[measured] merged keys == plain keys (set equality): {keys_identical}")
if not keys_identical:
    print("   only-in-merged:", sorted(merged_keys - plain_lm_keys)[:6])
    print("   only-in-plain :", sorted(plain_lm_keys - merged_keys)[:6])

# ---- ACTUALLY save merged checkpoint + reload, to prove round-trip on disk ----
import tempfile
from safetensors.torch import save_file, load_file
outdir = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(outdir, exist_ok=True)
merged_path = os.path.join(outdir, "merged_lm.safetensors")
sd = {k: v.contiguous() for k, v in merged_lm.state_dict().items()}
save_file(sd, merged_path)
reloaded = load_file(merged_path)
reloaded_keys = set(reloaded.keys())
p3, e3, m3, u3 = validate(plain_lm_keys, reloaded_keys)
print(f"[measured] saved merged checkpoint -> {merged_path}")
print(f"[c2-disk] plain-vs-reloaded-merged: PASS={p3} missing={m3} unexpected={u3}")

RESULTS["C_plain_keys"] = len(plain_lm_keys)
RESULTS["C_c0_pass"] = bool(p0)
RESULTS["C_c1_lora_unmerged_pass"] = bool(p1)
RESULTS["C_c1_missing"] = m1
RESULTS["C_c1_unexpected"] = u1
RESULTS["C_c1_sample_lora_keys"] = lora_only[:4]
RESULTS["C_c2_merged_pass"] = bool(p2)
RESULTS["C_c2_keys_identical"] = bool(keys_identical)
RESULTS["C_c2_disk_roundtrip_pass"] = bool(p3)

banner("SUMMARY")
print(json.dumps(RESULTS, indent=2))
with open(os.path.join(outdir, "results.json"), "w") as f:
    json.dump(RESULTS, f, indent=2)
print("\nwrote", os.path.join(outdir, "results.json"))
