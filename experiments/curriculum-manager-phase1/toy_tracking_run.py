# SPDX-License-Identifier: Apache-2.0
"""A tiny LIVE motion-tracking-shaped training run for Phase 1.

=============================  WHAT THIS IS  =============================
A knob-responsive closed-loop toy (numpy, CPU, seconds): N synthetic
"motions" with difficulties; a per-motion skill vector that improves in
proportion to sampling probability; a failure-weighted sampler with the
SAME floor/cap semantics as SONIC's (uniform_sampling_rate,
adp_samp_failure_rate_max_over_mean); a termination threshold that trades
success rate against training pressure. It emits the exact train/eval/
sampler JSONL record shapes `sonic-run-digest` consumes, including a real
held-out subset (motions never trained on; their skill moves only via
generalization spillover).

The point (doc 08 Phase 1): the manager's actions must CAUSE observable
changes in later digests — a live optimizer, not a replay. This validates
the manager loop end-to-end; it says nothing about SONIC performance.

=========================  WHAT THIS IS *NOT*  =========================
NOT SONIC, not PPO, not IsaacLab. Dynamics are hand-made and favor no
particular knob; numbers are [measured]-on-synthetic.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np


@dataclasses.dataclass
class ToyConfig:
    n_motions: int = 60
    heldout_fraction: float = 0.15
    seed: int = 0
    # dynamics
    learn_rate: float = 0.06          # skill gain per unit sampling mass
    generalization: float = 0.25      # held-out spillover from mean train gain
    forget_rate: float = 0.012        # skill decay when a motion is under-sampled
    noise: float = 0.01
    # threshold semantics: higher threshold => bigger success margin,
    # but less training pressure (less skill gain per visit)
    threshold_margin_scale: float = 0.55
    pressure_scale: float = 0.5


class ToyTrackingRun:
    """Live loop. Call `advance(iters, knobs)` then read `*_records`."""

    def __init__(self, cfg: Optional[ToyConfig] = None):
        self.cfg = cfg or ToyConfig()
        rng = np.random.default_rng(self.cfg.seed)
        n = self.cfg.n_motions
        self.difficulty = np.clip(rng.beta(2, 2, n), 0.05, 0.95)
        self.skill = np.clip(rng.normal(0.25, 0.05, n), 0.05, 0.5)
        n_held = max(2, int(n * self.cfg.heldout_fraction))
        held = rng.choice(n, size=n_held, replace=False)
        self.heldout_mask = np.zeros(n, dtype=bool)
        self.heldout_mask[held] = True
        self.keys = [f"toy_{i:03d}" for i in range(n)]
        self.failure_ema = np.full(n, 0.5)
        self.it = 0
        self.rng = rng

    # ── mechanics ────────────────────────────────────────────────────
    def _margin(self, threshold: float) -> float:
        # threshold in [0.15, 0.5] (registry hard range) → margin in [0, scale*0.35]
        return (threshold - 0.15) * self.cfg.threshold_margin_scale

    def _success_prob(self, threshold: float) -> np.ndarray:
        return np.clip(self.skill - self.difficulty + 0.45 + self._margin(threshold), 0.02, 0.995)

    def _sampling_prob(self, uniform_rate: float, cap_over_mean: float) -> np.ndarray:
        """SONIC-shaped: failure-weighted, capped at mean*cap, uniform floor mix."""
        train = ~self.heldout_mask
        f = self.failure_ema[train].copy()
        cap = max(f.mean(), 1e-8) * cap_over_mean
        f = np.clip(f, 0.0, cap)
        f = f / f.sum() if f.sum() > 0 else np.full(f.shape, 1 / f.size)
        u = np.full(f.shape, 1 / f.size)
        p = (1 - uniform_rate) * f + uniform_rate * u
        full = np.zeros(self.cfg.n_motions)
        full[train] = p / p.sum()
        return full

    def advance(self, iters: int, knobs: Dict[str, float]) -> Dict[str, List[dict]]:
        """Run `iters` training iterations under the given knob values.

        Knobs read (registry names; defaults = registry defaults):
          uniform_sampling_rate, adp_samp_failure_rate_max_over_mean,
          termination_threshold.anchor_pos
        Returns {"train": [...], "eval": [...], "sampler": [...]} records.
        """
        cfg = self.cfg
        uniform = float(knobs.get("uniform_sampling_rate", 0.1))
        cap = float(knobs.get("adp_samp_failure_rate_max_over_mean", 50.0))
        threshold = float(knobs.get("termination_threshold.anchor_pos", 0.30))
        pressure = 1.0 + cfg.pressure_scale * (0.30 - threshold) / 0.35

        train_records, sampler_records = [], []
        train = ~self.heldout_mask
        for _ in range(iters):
            self.it += 1
            p = self._sampling_prob(uniform, cap)
            succ = self._success_prob(threshold)
            # skill dynamics: visited motions learn (more under pressure),
            # under-sampled ones slowly decay toward their start
            gain = cfg.learn_rate * pressure * p * (1 - self.skill)
            decay = cfg.forget_rate * (p < 1 / (4 * cfg.n_motions)) * (self.skill - 0.2)
            self.skill = np.clip(
                self.skill + np.where(train, gain - np.maximum(decay, 0), 0.0)
                + self.rng.normal(0, cfg.noise, cfg.n_motions) * train,
                0.02, 0.98,
            )
            # held-out spillover from mean train gain
            self.skill[self.heldout_mask] = np.clip(
                self.skill[self.heldout_mask]
                + cfg.generalization * gain[train].mean(), 0.02, 0.98,
            )
            # failure EMA (SONIC-style slow EMA)
            fail = 1 - succ
            self.failure_ema = 0.98 * self.failure_ema + 0.02 * fail

            if self.it % 50 == 0:
                train_records.append({
                    "it": self.it,
                    "policy/approxkl_avg": float(0.01 * pressure * self.rng.uniform(0.8, 1.2)),
                    "loss/entropy_avg": float(np.clip(1.2 - self.skill[train].mean(), 0.1, 1.2)),
                    "loss/value_avg": float(0.5 * (1 - self.skill[train].mean())),
                    "lr": 2e-5,
                    "Episode/tracking_anchor_pos": float(succ[train].mean()),
                })
            if self.it % 100 == 0:
                sampler_records.append({
                    "it": self.it,
                    "failure_rate": self.failure_ema[train].tolist(),
                })

        # one eval pass at the end of the window (both metrics, like the
        # stock eval + held-out watcher pair)
        succ = self._success_prob(threshold)
        # eval uses a FIXED relaxed threshold (0.25-ish margin), not the knob;
        # rates are expectations (many-rollout limit) + small sampling noise,
        # so the protected metric is a usable signal rather than Bernoulli
        # noise over a handful of motions
        eval_succ = np.clip(self.skill - self.difficulty + 0.45 + self._margin(0.35), 0.02, 0.995)
        failed = [self.keys[i] for i in np.where(train)[0] if eval_succ[i] < 0.5]
        heldout_rate = float(np.clip(
            eval_succ[self.heldout_mask].mean() + self.rng.normal(0, 0.01), 0.0, 1.0
        ))
        eval_records = [{
            "it": self.it,
            "success_rate": round(float(1 - len(failed) / train.sum()), 4),
            "heldout_success_rate": round(heldout_rate, 4),
            "failed_keys": sorted(failed),
            "mpjpe_all_mean": round(float(120 * (1 - succ[train].mean())), 2),
        }]
        return {"train": train_records, "eval": eval_records, "sampler": sampler_records}
