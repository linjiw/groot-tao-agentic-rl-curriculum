# Adversarial data audit — COMPARISON_V4_RESULTS.md vs raw artifacts

Auditor: independent subagent, 2026-07-07. Sources checked:
`multiseed_v4_report.json`, `{control,manager}_journal_v4_seed{42,1337}.json`,
`{arm}_summary_v4_seed{seed}.json`, `archive_v4_failed/*`, `heldout/*`,
`run_comparison_multiseed.sh`, driver logs. All recomputations done with
python3 against the raw JSON (scripts inline below). Read-only audit; no
source artifact was modified.

## Verdict

The core numbers are honest: **every table cell, every per-segment metric,
all finals/means, the decision streams, prefix identity, rollback/eval-error
counts, and the held-out numbers reproduce exactly** from the report and
journals. However, the audit found **one factual claim contradicted by the
paper's own table (P1, arguably P0)**, **one derived-number range that is
wrong at the low end (P1)**, **one failure-narrative detail unsupported by
any retained artifact (P1)**, and several P2 nits (empty driver logs making
all duration claims unverifiable, loose disk arithmetic, inconsistent
per-seed ordering).

---

## P0 — fabricated / wrong numbers

**None found.** Every printed number that could be traced to an artifact
matched to printed precision. Specifically verified [verified]:

- All 40 cells of the PRIMARY per-segment table == `eval_progress_rate_PRIMARY.per_seed_per_segment` == the per-segment `eval.progress_rate` in all four journals (round to 4 dp, exact match, 0 mismatches across 40 segments × 3 metrics = 120 comparisons).
- Final-segment cross-seed table: control 0.0988/0.0916, mean 0.0952; manager 0.0951/0.1056, mean 0.10035 → printed 0.1004 ✓.
- mpjpe_l finals 43.5514/43.2691 → 43.55/43.27, mean 43.41025 → 43.41; manager 42.5164/45.5009 → 42.52/45.50, mean 44.00865 → 44.01 ✓. s1337 manager s6→s7 mpjpe_l 40.9869→44.8226 matches "41.0→44.8" ✓.
- mpjpe_g: control mean 56.6212 (per-seed 58.1771/55.0653) → "56.62 (55.07/58.18)"; manager 58.4746 (56.5171/60.4321) → "58.47 (56.52/60.43)" ✓ (but see P2-3 on ordering).
- Held-out: `heldout_success_rate` = 0.0 for all 40 entries ✓. `heldout_mpjpe_g` s1→s10 raw: ctrl s42 51.490→50.779 ("51.5→50.8" ✓), ctrl s1337 52.132→52.903 ("52.1→52.9" ✓), mgr s42 51.490→50.831 ("51.5→50.8" ✓), mgr s1337 52.132→52.608 ("52.1→52.6" ✓).
- Train context: len 18.625 vs 13.53 → "18.6 vs 13.5" ✓; rew 1.1557 vs 0.9258 → "1.16 vs 0.93" ✓.
- +5.4% relative: (0.10035−0.0952)/0.0952 = **+5.41%** ✓ (header's "+5%" consistent).
- Run integrity: `eval_sources` = "_eval" × 40 ✓; `rollbacks` all 0 ✓; `eval_errors` all [] ✓; summaries `rejected: 0` for all four runs ✓; `prefix_identity_all_seeds: true` ✓, and empirically re-verified: segment-1 `eval`, `heldout`, and `knobs_in` dicts are byte-identical between arms within each seed ✓.
- Decision streams [verified against both `manager_decisions` and journal `decision`/`applied`/`outcome` events]:
  - s42: t2 foot_pos_xyz 0.20→0.25 (knobs_in confirms from-value 0.2), t4 ee_body_pos 0.15→0.20, t6 foot_pos_xyz 0.25→0.30 @iter 300, t8 ee_body_pos 0.20→0.25, t10 foot_pos_xyz 0.30→0.35 `pending` @iter 500 — exactly as the md states ✓.
  - s1337: t2/t4/t6 identical knobs/values/ticks to s42 ✓; t6 outcome `survived_effect_confirmed` (journal effect: len_mean 20.31 ≥ 20.0, confirmed:true) — the only confirmed outcome in the four kept journals ✓; t10 ee_body_pos →0.25 `pending` ✓; ticks 8–9 were `action:none` (band satisfied), consistent with the md listing only 4 decisions for s1337 ✓.
- s7 jump reproduces [verified]: mgr s42 0.0870→0.0993 (Δ +0.0123), mgr s1337 0.0858→0.1028 (Δ +0.0170); control same transition +0.0030 / +0.0011. The qualitative claim is solid. (But see P1-2 on the stated delta range.)
- Held-out split integrity [verified]: `heldout/curriculum_eval_subset64.json` (64 keys) and `heldout/eval_subset64.json` (64 keys) have **0 overlapping motion keys** — "disjoint 64-motion subset" ✓.
- Archive failure narrative core [verified]: `archive_v4_failed/manager_journal_v4_seed1337.json` tick 9 carries `eval_error`/`heldout_error` (no metrics_eval.json produced) and tick 10 is `{"event":"segment_failed","tracebacks":2}`; the archived summary exists with `rew_series[9]=null` — consistent with bug item 3's "summary JSON existed for the failed run" ✓.

## P1 — misleading framing / claims not supported by data

1. **[verified] "segments 2–6 manager ≤ control in both seeds" is FALSE.**
   At seed 1337, segment 6: manager 0.0858 > control 0.0841 (the md's own
   table shows this). The "early cost in both seeds" claim holds for
   segments 2–5 in s1337 and 2–6 in s42, but the stated range/scope is
   contradicted by the data. Recomputation: all 10 (seed, seg∈2..6) pairs
   checked; 9 satisfy mgr ≤ ctrl, 1 violates (s1337 seg6, +0.0017).
   Arguably P0 since it is a flatly wrong statement about the numbers; it
   does not change the paper's direction (it actually *understates* the
   manager) but it is incorrect as written.

2. **[verified] The "s7 jumps (~0.014–0.017, ~60 quanta)" range is wrong at
   the low end.** Recomputed deltas: s42 = 0.0993−0.0870 = **0.0123**
   (49.2 quanta at 1/(2·2002)), s1337 = 0.1028−0.0858 = **0.0170**
   (68.1 quanta). The correct range is ~0.012–0.017, i.e. ~49–68 quanta.
   "0.014" does not correspond to any s6→s7 manager delta; "~60 quanta"
   overstates the smaller jump by ~22%. The conclusion (far above the
   0.00025 quantum) survives, but the printed derived numbers do not
   reproduce.

3. **[speculative — unsupported by retained artifacts] The
   `OSError: [Errno 28]` attribution for the failed run.** Grep across all
   retained journals, summaries, and logs (including `archive_v4_failed/`)
   finds **no occurrence of "Errno 28" or "OSError"**. The archived tick-9
   error says only: eval "produced no metrics_eval.json … docker exec
   failed (1): cat … No such file or directory", and points to a
   `/workspace/...` log that is not among the artifacts. Disk exhaustion is
   a plausible root cause but the specific errno cited in the md cannot be
   verified from anything shipped alongside it. Similarly the md's
   "detected … not by the exit code alone" narrative can't be checked —
   see P2-1.

4. **[verified] All timing/size claims are unverifiable: every driver log
   is 0 bytes.** `*_driver_v4_seed*.log` and `archive_v4_failed/*.log` are
   all empty files. Therefore "exit 0, 1h58m", "~8 h multi-seed campaign",
   and "each segment leaves ~3.6 GB … 4 runs ≈ 120 GB" rest on nothing in
   the artifact set. Note also (a) the run script's own comment estimates
   "~4.5 h" for the 40-segment default, which is inconsistent with the
   md's "~8 h" unless re-runs are being counted, and (b) the disk
   arithmetic doesn't close: 3.6 GB × 40 segments = 144 GB, not "≈ 120 GB".
   The md labels its "Run integrity" section [verified]; the duration and
   disk figures inside/near it do not meet that bar.

5. **[verified, minor framing] "prefix identity intact" is based on a
   single compared segment.** `prefix_identity_per_seed.*.n_compared = 1`
   — only segment 1 precedes the first decision (tick 2), so the identity
   check compares exactly one segment per seed. The md's wording
   ("identical until the first manager decision (tick 2)") is technically
   accurate, but citing prefix identity as a headline integrity property
   should note it is a 1-segment comparison per seed.

## P2 — nits

1. **[verified] Bug item 3 / "cleanly journaled" is only partially
   evidenced.** The archive journal does show `segment_failed` at tick 10
   and eval errors at tick 9 (good), but with empty driver logs the claims
   about rc=0 masking and how verification caught it are narrative only.

2. **[verified] mpjpe_g per-seed ordering is inconsistent.** Control is
   printed "(55.07/58.18)" = s1337-first, while manager "(56.52/60.43)"
   and every other table in the doc are s42-first. Values correct;
   ordering invites misreading.

3. **[verified] "0.1004" vs report's 0.10035 and "+5.4%" vs 5.41% —**
   rounding is fine, just noting the report itself stores 0.10035 while
   two md locations print 0.1004; no discrepancy in substance.

4. **[verified] "the only confirmed-effect outcome in the study"** is true
   for the four kept journals; the *discarded* failed run also logged a
   `survived_effect_confirmed` at its t6 (archive journal). Fine as
   worded ("in the study"), but worth knowing the effect also appeared in
   the aborted attempt (which mildly *strengthens* reproducibility).

5. **[verified] `manager_decisions` in the report omits the "from" values**
   (only `value` = new value). The md's from→to values were verified
   instead via each tick's `knobs_in` in the journals — all correct — but
   the report alone can't support the "from" side of the md's decision
   table.

## Recomputation notes (what was actually run)

- Cross-checked all 40 journal entries × {progress_rate, mpjpe_l_all_mean,
  mpjpe_all_mean} against the report arrays (round-to-4dp equality): 0
  mismatches.
- Cross-checked the md's 40-cell PRIMARY table against the report arrays:
  0 mismatches.
- Recomputed means: ctrl (0.0988+0.0916)/2 = 0.0952; mgr (0.0951+0.1056)/2
  = 0.10035; relative = +5.4097%.
- Recomputed s6→s7 deltas and quanta (quantum 1/(2·2002) = 0.0002497…):
  s42 +0.0123 = 49.2 q; s1337 +0.0170 = 68.1 q.
- Set-intersection of the two 64-key subsets: empty.
- Byte-equality of segment-1 eval/heldout/knob dicts across arms per seed:
  identical for both seeds.
- grep for "Errno 28"/"OSError" across all artifacts: no hits.

## Bottom line

No fabricated data. The measured tables and the decision/journal audit
trail are fully consistent — unusually clean. Required fixes before citing:
correct the "segments 2–6 in both seeds" claim (P1-1), fix the jump-size
range to ~0.012–0.017 / ~49–68 quanta (P1-2), and either produce evidence
for the Errno-28 / 1h58m / ~8h / ≈120 GB figures or mark them as
unverified operator recollection (P1-3, P1-4).
