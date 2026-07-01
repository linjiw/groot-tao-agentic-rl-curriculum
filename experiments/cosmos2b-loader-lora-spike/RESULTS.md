# Experiment ②: Cosmos-Reason2-2B backbone load + LoRA attach + strict-loader wall

**Date:** 2026-07-01 · **HW:** single NVIDIA A10G (23 GB) · **Env:** `/workspace/Isaac-GR00T/.venv`
(torch 2.7.1+cu128, transformers 4.57.3, peft 0.17.1, CUDA available)
**gr00t source:** `/workspace/Isaac-GR00T` @ ab88b50 (= repo submodule pin) — **not modified**
**Script:** `experiments/cosmos2b-loader-lora-spike/spike_backbone_lora.py`
**Raw output:** `experiments/cosmos2b-loader-lora-spike/out/results.json`, `out/merged_lm.safetensors`

---

## 0. Environment reality-check (correction to task premise)

- `[verified]` The cache is at **`HF_HOME=/workspace/hf-cache`**, NOT `~/.cache/huggingface`
  (the `~/.cache` copy is an empty `refs/main` stub).
- `[verified]` `nvidia/Cosmos-Reason2-2B` under `/workspace/hf-cache/hub` has **config + tokenizer
  JSON only — NO weight shards**. `nvidia/GR00T-N1.7-3B` cache is a 12 KB stub (no weights).
  Network to HF is **gated: HTTP 401, no token** → no download possible.
- `[verified]` This does **not** block the spike. The backbone code path
  (`qwen3_backbone.py:154-158`) loads **config only** and instantiates *random* weights;
  real weights are overwritten later by `GR00T.from_pretrained`. Every spike question here is
  about **state_dict KEYS / STRUCTURE**, not weight values, so the config-only instantiation is
  faithful. The one thing weights-absence prevents is running the *full* `AutoModel.from_pretrained`
  VLA reload; per task instructions we instead **replicate `setup.py`'s exact validator logic**
  against the real key sets (see §4).

---

## 1. Confirmed source facts

| Claim | Verdict | Evidence |
|---|---|---|
| Backbone loads `nvidia/Cosmos-Reason2-2B` | `[verified]` | `qwen3_backbone.py:107` default `model_name="nvidia/Cosmos-Reason2-2B"`; config loaded at `:154-158`, model built at `:158` |
| Layers popped until `len == select_layer` | `[verified]` | `qwen3_backbone.py:161-162` `while len(self.model.language_model.layers) > select_layer: ...pop(-1)` |
| **CODE default `select_layer` is `-1`** | `[verified]` | `qwen3_backbone.py:110` `select_layer: int = -1` (default `-1` ⇒ `while len>-1` pops **all** layers) |
| **Config sets `select_layer = 12`** | `[verified]` | `gr00t/configs/model/gr00t_n1d7.py:47` `select_layer: int = 12` (also `model_name` :40, `backbone_embedding_dim=2048` :44) |
| `set_trainable_parameters` uses boolean `requires_grad_` gates only, **no peft** | `[verified]` | `qwen3_backbone.py:177-199` — `requires_grad`, `requires_grad_(False)`, top-layer loop; no `import peft` anywhere in file |
| Strict loader calls `AutoModel.from_pretrained(..., output_loading_info=True)` and RAISES on missing/unexpected/mismatched | `[verified]` | `gr00t/model/gr00t_n1d7/setup.py:82-120` — builds `errors[]`, `raise RuntimeError` at `:117` |
| **Only whitelisted exception is `action_head.mask_token`** | `[verified]` | `setup.py:98-104` (`mask_token_missing` → init), `:108` `other_missing = [k for k in missing_keys if "mask_token" not in k]` |

Net: the two "unverified from memory" line numbers were right in spirit; the precise facts —
**code default `-1`, config override `12`, mask_token-only whitelist** — are all confirmed above.

---

## 2. Spike A — backbone load & `select_layer` truncation `[measured]` PASS

- Full model per cached `config.json`: **`num_hidden_layers = 16`** (hidden_size 2048, `model_type=qwen3_vl`, arch `Qwen3VLForConditionalGeneration`).
- Built `Qwen3Backbone(select_layer=12, load_bf16=True, local_files_only=True)` → loaded clean.
- **Kept `language_model.layers` = 12** (exactly `select_layer`; 4 top layers popped). ✅

## 3. Spike B — LoRA attach on kept layers `[measured]` PASS

`LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj","v_proj"])`
(mirrors `spec_template_train.yaml:69-80`), applied via `get_peft_model` to the **truncated** `language_model`.

- Trainable params: **1,273,811,968 → 688,128** after LoRA (peft freezes base, only adapters trainable).
- **48 LoRA tensors** (12 layers × 2 targets × {lora_A, lora_B}).
- Adapter-bearing layer indices = **[0..11]**, max index **11** ⇒ adapters land **only on the 12 kept layers**, none on popped layers. ✅
- Sample key: `base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight`.

## 4. Spike C — the strict-loader wall `[measured]`

Validator faithfully replicates `setup.py:97-120`: `reference` = freshly-built **plain** backbone
`language_model` keys (what the from-scratch model expects); `checkpoint` = the state_dict we try to load.
`missing = reference − checkpoint`, `unexpected = checkpoint − reference`, mask_token whitelisted.

| Branch | Checkpoint keys | missing | unexpected | Validator verdict |
|---|---|---|---|---|
| **c0** plain vs plain (baseline) | 134 | 0 | 0 | **PASS** ✅ |
| **c1** LoRA-wrapped, **NOT merged** | 182 | **134** | **182** | **FAIL** ❌ → `RuntimeError` |
| **c2** `merge_and_unload()` | 134 | 0 | 0 | **PASS** ✅ |
| **c2-disk** save→reload merged `.safetensors` | 134 | 0 | 0 | **PASS** ✅ |

- `[measured]` **c1 (unmerged) trips the wall two ways at once:** peft re-parents every base weight
  under `base_model.model.*` (so all 134 base keys are reported *missing*) **and** adds 48 `lora_A/lora_B`
  keys plus the renamed 134 (182 total *unexpected*). Neither is whitelisted → `RuntimeError`.
  Sample unexpected: `base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight`.
- `[measured]` **c2 (merged) is byte-key-identical to the plain backbone:** `merge_and_unload()`
  folds `B·A` into `q_proj/v_proj` weights, strips the `base_model.` wrapper and all `lora_*` keys.
  `set(merged_keys) == set(plain_keys)` → **True**; 0 missing, 0 unexpected.
- `[measured]` Round-trip proven on disk: merged `state_dict` saved to
  `out/merged_lm.safetensors`, reloaded via `safetensors.torch.load_file`, re-validated → **PASS**.

### Strict-loader verdict
- `[measured]` **`merge_and_unload()` clears the wall.** A merged LoRA checkpoint presents an
  identical key set to the plain backbone, so GR00T's `setup.py` strict validator passes with
  **zero whitelist relaxation needed**.
- `[measured]` **An un-merged LoRA checkpoint does NOT pass** and would require invasive
  whitelist changes (accept `base_model.` prefix + `lora_*` keys) that GR00T does not implement.
  ⇒ The bridge must **merge before handing the checkpoint to GR00T**, not ship raw adapters.
- `[speculative]` Not exercised here: (a) numerical equivalence of merged weights vs. LoRA-active
  forward pass (only keys checked, weights were random since 2B shards absent); (b) the *full* VLA
  `AutoModel.from_pretrained` reload with real GR00T-3B weights (not cached). Both are
  weight-value concerns, orthogonal to the key-structure wall this spike settled.

---

## 5. Conversion-helper assessment — `prepare_cosmos3_vlm_checkpoint.py` `[verified]`

- Purpose (docstring :4-8): convert a **Cosmos3-Nano *Omni* checkpoint** (`--checkpoint-path`) into a
  **Qwen3-VL safetensors dir** via `cosmos_framework.scripts.convert_model_to_vlm_safetensors` (:137-140),
  run inside `nvcr.io/nvidia/pytorch:25.09-py3` (:22).
- `--vlm-model-name` (default **`Qwen/Qwen3-VL-8B-Instruct`**, :23) is the **target Qwen3-VL architecture
  template** the converter maps Omni weights onto — it is **not** a checkpoint you load directly.
- Success criteria are all **family/format** checks, not size: output `config.model_type == "qwen3_vl"`
  (:54-55, :149-150, :231), `architectures` includes `Qwen3VLForConditionalGeneration` (:56-57),
  present `model.safetensors.index.json` + shards (:64-72), tokenizer files (:74-77).

**Is pointing it at Cosmos-Reason2-2B plausible?**
- `[verified]` Cosmos-Reason2-2B is the **same model family** the helper targets: its `config.json`
  has `model_type=qwen3_vl` and `architectures=["Qwen3VLForConditionalGeneration"]` — exactly what the
  validators demand. So it satisfies the *family/format* contract.
- `[verified]` It is a **different size** from the default: 2B has `hidden_size=2048`, `num_hidden_layers=16`,
  `intermediate_size=11008`, `num_attention_heads=16/8kv`; `Qwen3-VL-8B-Instruct` is materially larger.
  The 2B hidden_size 2048 matches GR00T's `backbone_embedding_dim=2048` (config :44), i.e. the 2B is the
  size GR00T actually expects.
- `[speculative]` Whether `convert_model_to_vlm_safetensors` accepts a 2B-shaped `--vlm-model-name` and
  emits shards whose per-tensor shapes match a 2B Omni source depends on `cosmos-framework` internals
  (not present locally; requires the NGC container + a real Cosmos3-Nano Omni checkpoint). The helper's
  own gate (`model_type==qwen3_vl`) would pass; a **shape** mismatch would only surface at convert time.

**Bottom line:** same family (`qwen3_vl`), different default size. Re-pointing `--vlm-model-name` at a
2B Qwen3-VL is architecturally coherent and format-compatible; the only open risk is shape-plumbing
inside the unvendored converter, which needs the container to prove.

---

## 6. GO / NO-GO — TAO → GR00T LoRA bridge

### 🟢 **GO** (conditional)

`[measured]`/`[verified]` evidence supports proceeding:
1. GR00T's backbone loads the Cosmos-Reason2-2B config and truncates to 12 layers cleanly (A).
2. TAO's shipped LoRA hyperparams (r=8, α=8, q/v_proj) attach cleanly to exactly the kept layers (B).
3. **The strict-key loader wall is passable without touching submodule source** — a
   `merge_and_unload()` checkpoint is key-identical to the plain backbone and clears the validator;
   round-trip proven to disk (C).

### Conditions / guardrails
- **MUST merge adapters before GR00T ingest.** Shipping raw (un-merged) LoRA checkpoints → guaranteed
  `RuntimeError` (134 missing + 182 unexpected keys). The bridge's export step = `merge_and_unload()` → save.
- **Still to validate** before full GO on the training track:
  - `[speculative]` End-to-end **weight-value** equivalence + a real full-VLA `from_pretrained` reload
    once GR00T-N1.7-3B (and 2B) weight shards are actually available (network/token or pre-seeded cache).
  - `[speculative]` `cosmos-framework` converter shape-compatibility when `--vlm-model-name` points at a
    2B target (needs the NGC container + a Cosmos3-Nano Omni source checkpoint).
- No submodule files under `external/` or `/workspace/Isaac-GR00T` were modified. New artifacts live only
  in `experiments/cosmos2b-loader-lora-spike/`.

---

### Artifacts
- `experiments/cosmos2b-loader-lora-spike/spike_backbone_lora.py` — the spike (A+B+C)
- `experiments/cosmos2b-loader-lora-spike/out/results.json` — machine-readable measurements
- `experiments/cosmos2b-loader-lora-spike/out/merged_lm.safetensors` — merged checkpoint used for the disk round-trip
