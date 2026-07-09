# G0 / G1 / G2 pre-registration (doc 10 §4; validation gate V1)

**Committed BEFORE launch.** Anything not specified here is exploratory and cannot
headline (V1). Endpoint, τ, seeds, exclusion rules, and decision tables are fixed now.
Amendments after this commit must be dated and justified; post-hoc endpoint changes
void the confirmatory status of the affected experiment.

Status at prereg time (2026-07-09): tier-0 σ-EMA shim built + CPU/container-verified
(I1, 197 core tests); resolving held-out metric + replicate-collapse built (I2). GPU
blocked on the IsaacLab pilot; G0 launches when it frees.

---

## Shared setup (all three)

- **Engine/platform:** SONIC GR00T-WBC, single A10G (23 GB), `isaac-lab-base` container.
- **Training motion set:** `data/motion_lib_bones_seed/robot_curriculum` (116,924 motions).
- **Standard eval (the scoreboard):** pinned to
  `data/motion_lib_bones_seed/robot_curriculum_eval64` (64 motions), fixed relaxed
  thresholds, every action-space knob re-pinned (V6, existing structural test).
- **Held-out protected split:** `robot_heldout_eval64` (64 motions, disjoint, salted
  manifest v0.1.0). **Protected metric = `heldout_progress_rate`** (per-motion progress,
  the RESOLVING metric — I2.1; NOT success_rate, which is 0.0-everywhere).
- **Primary endpoint (curriculum-side):** mean of the FINAL-2 segments'
  `eval/progress_rate` on `curriculum_eval64` (final-2 damps s8–s10 oscillation).
- **Segment length:** 50 iters × 256 train envs; 64 eval envs. 10 segments (G0/G1/G2).
- **Tier-0 injection:** `job_adapter.build_train_command(env=..., pythonpath=..., extra_overrides=[func-swap])`;
  shim deployed to `/workspace/rmc_tier0` (copy, not symlink — I1 F10).
- **Run-success criterion:** artifact-based (journal events + snapshot presence +
  `_eval` provenance), never exit codes (V7). Disk gate + purge active (E0).
- **Statistical rule:** paired-by-seed; n=3 → per-seed table + range, NO significance
  test, NO CI (V2). Bit-identical replicates collapsed before any spread (I2.2).

---

## G0 — Library-native warm start + insertion bit-identity smoke (prereq)

**Purpose:** produce the shared warm-start checkpoint and PROVE the insertion perturbs
nothing (gate-0). Not a comparison — infrastructure.

**Protocol:**
1. Train ~2,000 iters from scratch on `robot_curriculum` (seed 42), no checkpoint.
   Verify the `curriculum_eval64` eval curve is in the live/rising region (not
   saturated, not floored) — record the curve.
2. From that checkpoint, run TWO 10-iter smoke segments:
   - `stock`: no tier-0 override.
   - `noop`: `SONIC_TIER0_ACTIVE=0` + the `func`-swap override + PYTHONPATH.
3. Gate-0: `core.equivalence.compare_journals(stock, noop)` on `rew_mean_last`,
   `len_mean_last` MUST be `bit_identical`.

**Pass/fail:** bit-identical → insertion is inert, proceed to G1. NOT bit-identical →
STOP: the shim perturbs training via import side-effects or op reordering; debug before
any σ-EMA claim (the no-op returns the stock reward object unchanged, so a failure is a
real defect, not tolerance).

**Cost:** ~2–4 GPU-h. **Kill:** n/a (infra), but gate-0 failure blocks everything.

## G1 — Noise floor + lever sensitivity at the new operating point

**Purpose:** measure τ_G at THIS operating point (equivalence.py's own re-measure rule)
and confirm the σ-EMA lever can move the endpoint at all (V5).

**Protocol:**
1. **Noise floor:** 3 launches of the STOCK config from the G0 checkpoint (seed 42),
   10 segments, + 1 ε-perturbed chaos run (E5b method: perturb a continuously-acting
   path — `entropy_coef` — by an fp32-distinct ε). Collapse bit-identical replicates
   (I2.2 `collapse_by_segment`); compute the endpoint noise band over DISTINCT
   trajectories (`noise_band_from_replicates`). τ_G = `calibrate_tau(floor,
   safety_factor=3)`, sanity-checked to not swallow a 10% effect.
2. **Lever sensitivity:** 1 σ-EMA run (`SONIC_TIER0_ACTIVE=1`, ema_rate=0.01 —
   aggressive-but-sane) from the G0 checkpoint, 10 segments. Endpoint must differ from
   the stock mean by > τ_G (magnitude, either sign) — proving the lever grips.

**Pass/fail (pre-registered):**
- (i) τ_G > 0.10 (swallows a 10% effect) → SURFACE to user before G2 (inherited
  NARROW-GO Phase-A checkpoint). Do not auto-proceed.
- (ii) σ-EMA endpoint within τ_G of stock in the sensitivity probe → the lever cannot
  move the metric at this budget; **report that** and do NOT run G2 as a value test
  (phase-2's mistake was a null on an insensitive lever). Pivot to writeup + IsaacLab
  breadth.
- else → τ_G locked; proceed to G2.

**Cost:** ~6 runs × 10 seg ≈ 18 GPU-h (2 nights).

## G2 — The gate: σ-EMA vs stock

**Purpose:** the go/no-go for the whole GACL line (doc 10 §4-G2).

**Arms:** `{stock}` vs `{σ-EMA}` (`SONIC_TIER0_ACTIVE=1`, ema_rate + sigma_floor fixed
at the G1-informed setting, `SONIC_TIER0_SIDECAR_DIR` set so σ persists across the 10
segment relaunches — I1 F8). Stock adaptive sampling stays ON in BOTH arms (the test is
+σ-EMA, not instead-of; risk 3). Everything else frozen. Identical G0 warm start + seed
per pair.

**Design:** 2 arms × 3 seeds (42, 1337, 7) × 10 segments = 60 segments.

**Endpoint & analysis:**
- Primary: paired (per-seed) Δ = σ-EMA − stock on the final-2-segment mean
  `curriculum_eval64` progress_rate.
- Mandatory per-motion decomposition (V3): any single motion contributing > 50% of a
  seed's Δ triggers the leave-one-out re-analysis; report W/L/T + median paired delta.
- Secondary: `heldout_progress_rate` (protected, resolving), survival-matched mpjpe_l.
- σ trajectory journaled (the trace sidecar) — reported as a mechanism check.

**Decision rule (pre-registered):**
- **PASS** = paired mean Δ > τ_G AND consistent sign across ≥ 2/3 seeds, AFTER the
  single-motion exclusion rule. → proceed to G3 (add bin-LP) then G4 (n=5 confirmatory).
- **FAIL** = Δ ≤ τ_G, or sign flips, or the single-motion exclusion erases it. →
  **project-level NO-GO for the GACL line**: ship M (methodology) + the honest negative
  (doc 08:135 prior), pivot to IsaacLab breadth.

**Cost:** 6 runs × 10 seg ≈ 18 GPU-h (2 nights).

---

## Launch commands (reference — the exact shape, `job_adapter`)

```python
# G0 warm start (from scratch, library motion set)
build_train_command("g0_warm", num_envs=256, iterations=2000,
    extra_overrides=["++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_curriculum"],
    log_path=".../g0_warm.log")

# G0 noop bit-identity arm (10 iters from the warm ckpt)
build_train_command("g0_noop_s1", checkpoint="<g0_warm last.pt>", iterations=10,
    env={"SONIC_TIER0_ACTIVE": "0"}, pythonpath="/workspace/rmc_tier0",
    extra_overrides=["++manager_env.rewards.tracking_anchor_pos.func=adapters.sonic_tier0.sonic_sigma_ema_term:SigmaEMAAnchorPos",
                     "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_curriculum"])

# G2 sigma-EMA arm (active, sidecar-persisted sigma)
build_train_command("g2_sigma_s1", checkpoint="<g0_warm last.pt>", iterations=50,
    env={"SONIC_TIER0_ACTIVE": "1", "SONIC_TIER0_EMA_RATE": "0.01",
         "SONIC_TIER0_SIGMA_FLOOR": "0.1",
         "SONIC_TIER0_SIDECAR_DIR": ".../g2_sigma_s1_sidecar",
         "SONIC_TIER0_TRACE": ".../g2_sigma_s1_sigma_trace.jsonl",
         "SONIC_TIER0_LOG_EVERY": "10"},
    pythonpath="/workspace/rmc_tier0",
    extra_overrides=["++manager_env.rewards.tracking_anchor_pos.func=adapters.sonic_tier0.sonic_sigma_ema_term:SigmaEMAAnchorPos",
                     "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_curriculum"])
```

## Open decisions folded here (not blockers)
- **Which tracking term(s) σ-EMA wraps.** First cut wraps `tracking_anchor_pos` only
  (the largest-weight anchor term). If G1 sensitivity is weak, G1b may widen to the
  5-point/relative-body terms before G2 — that widening is a G1 finding, journaled, not
  a post-hoc G2 change.
- **ema_rate/sigma_floor values** are set from G1, then FIXED for G2 (they are not
  swept in G2 — that would confound the gate).
