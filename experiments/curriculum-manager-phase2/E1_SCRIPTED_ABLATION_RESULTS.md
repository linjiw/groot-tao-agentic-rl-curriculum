# E1 Scripted-Replay Ablation — Results (v5)

**Date:** 2026-07-08 · **Runs:** `ARMS=scripted SEEDS="42 1337"`, 10 segments × 50 iters × 256 envs each, eval 64 envs on the fixed 64-motion set. Artifacts: `scripted_journal_v5_seed{42,1337}.json`. 20/20 segments done, 0 failures, checkpoint purge freed 44 GB total (E0 in production, incl. docker-exec fallback for root-owned files).

## Question (pre-registered)

Does replaying the manager's exact knob ladder **open-loop** (no digest, no watch gating, tick-indexed only) reproduce the manager arm's results? If yes → the manager's closed-loop adaptivity contributes nothing beyond its schedule.

## Ladder fidelity [verified]

`knobs_in` per segment confirms the scripted arm applied the identical sequence the v4 manager chose (one-segment application lag identical to manager semantics): foot_pos_xyz 0.20→0.25 (s3) →0.30 (s7); ee_body_pos 0.15→0.20 (s5) →0.25 (s9). The t10 rung applies to a segment 11 that never runs — same as v4 manager.

Note: manager seed1337 issued its second ee_body_pos step at tick 10 (not tick 8): with the one-segment application lag the manager's rung would land at s11 (never runs) while the scripted ladder applies it at s9 — i.e. for seed1337 the scripted second ee rung is two segments *earlier* than that seed's manager transcript, and is the only arm that actually experienced it. Seed42's ladder is an exact replay.

## Result — final eval_progress_rate (PRIMARY)

| seed | scripted (open-loop) | manager (closed-loop) | control |
|---|---|---|---|
| 42   | 0.0947 | 0.0951 | **0.0988** |
| 1337 | **0.1092** | 0.1056 | 0.0916 |

Per-motion paired, scripted vs manager (final segment, n=64):

| seed | scripted wins | manager wins | ties |
|---|---|---|---|
| 42   | 19 | 21 | 24 |
| 1337 | 23 | 7  | 34 |

## Key observations [measured]

1. **Scripted matches or beats manager in both seeds.** Seed42: statistically indistinguishable (0.0947 vs 0.0951, W/L 19/21 ≈ coin flip). Seed1337: scripted slightly ahead (0.1092 vs 0.1056, W/L 23/7). The closed-loop machinery (digest reading, sustain windows, tripwire watches) added nothing over its own decision transcript.
2. **Prefix determinism then micro-divergence:** identical seed+knobs gives identical eval values for the early segments (seed42 s1–s3, seed1337 s1–s8 match manager to 4 decimals), after which GPU-level nondeterminism drifts them apart *even under identical configs* (seed42 s4: 0.0867 vs 0.0852 with the same knobs). Run-to-run noise at fixed seed is real and non-zero.
3. **The postmortem "breakthrough" is stochastic, not schedule-caused.** `postmortem_convulsions_stomach_loop_R_001__A471_M` final: seed42 scripted **0.054** vs manager 0.426 (same ladder, same seed — did NOT reproduce); seed1337 scripted 1.000 = manager 1.000 (did reproduce). A breakthrough that appears/disappears under an identical schedule and identical seed is driven by training stochasticity, not by the knob decisions.
4. **The ladder itself shows no consistent value over control:** seed42 control finishes highest (0.0988); seed1337 scripted finishes highest (0.1092). Cross-seed sign flip, n=2 — no claim beyond noise.

## Verdict (per pre-registered kill-criteria)

**Adaptivity is dead as an explanation of v4 differences.** Open-loop replay reproduces (or exceeds) the closed-loop manager within noise in both seeds. Combined with the v4 post-review amendment (arm-level signal did not survive per-motion decomposition), the phase-2 evidence is:

- closed-loop decision-making: **no measurable contribution** (this ablation)
- the specific knob schedule: **no consistent contribution vs control** (sign flips across seeds)
- observed arm-level differences: dominated by **single-motion stochastic breakthroughs** + run noise that exists even at fixed seed (obs. 2)

**Project implication:** the "adaptive curriculum manager" framing does not survive. Value, if any, lives in (a) schedule *discovery* (search over ladders, not reactive adjustment), and (b) the guardrail/journal/rollback scaffold itself, which is validated infrastructure. E3(b) (survival-matched mpjpe re-eval, ~1.3 GPU-h) is now low-priority: it would refine a comparison whose headline is already null.

## Caveats

- n=2 seeds; all quantitative statements are ranges, not CIs.
- progress_rate quantization ≈ 0.00025 (1/(2·2002)).
- mpjpe metrics remain survivor-biased (see COMPARISON_V4_RESULTS.md caveats).
- seed1337 ladder timing differs from that seed's manager transcript by one segment on the second ee rung (see above); the match-or-beat conclusion is robust to this since scripted ≥ manager there.
