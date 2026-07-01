"""TINY, honest policy-improvement demonstration for RLVR verifier rewards.

=============================  WHAT THIS IS  =============================
A *mechanism demo*: two toy parametric policies optimized purely from the
verifier rewards in ``verifiers.py``, on SYNTHETIC structured-QA tasks that
mimic the shape of TAO data-skill outputs. It shows that a standard REINFORCE /
reward-weighted update driven by a rule/verifier reward makes reward go UP over
iterations -- i.e. the "reward engine" closes an RL loop.

=========================  WHAT THIS IS *NOT*  =========================
NOT a real VLA/LLM RL run. No GR00T, no cosmos-reason, no TAO training, no GPU,
no real images/videos. The policies are numpy toys (a softmax bandit over
answer letters, and a 1-parameter numeric guesser). Numbers are
[measured]-on-synthetic. This proves the verifier-reward *interface* and its
optimizability, not model quality.

All randomness is seeded; runs in well under a second on CPU.
"""

from __future__ import annotations

import numpy as np

from verifiers import multiple_choice_exact, numeric_tolerance

SEED = 0


# --------------------------------------------------------------------------- #
# Demo A: softmax bandit over multiple-choice answers  (REINFORCE)
# --------------------------------------------------------------------------- #
def demo_multiple_choice(n_iters: int = 300, n_choices: int = 4, lr: float = 0.3):
    """Policy = softmax over `n_choices` answer letters, no context (bandit).

    Synthetic task: a fixed gold answer letter. Reward = multiple_choice_exact.
    Update = vanilla REINFORCE:  theta += lr * (R - baseline) * grad_logprob.
    Expected: policy concentrates mass on the gold letter, mean reward -> ~1.0.

    Returns a list of per-iteration mean rewards (moving over a small batch).
    """
    rng = np.random.default_rng(SEED)
    gold_idx = 2  # gold answer is letter "C"
    gold_letter = chr(ord("A") + gold_idx)
    theta = np.zeros(n_choices)  # logits
    batch = 16

    curve = []
    for _ in range(n_iters):
        logits = theta - theta.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        actions = rng.choice(n_choices, size=batch, p=probs)
        rewards = np.array(
            [multiple_choice_exact(chr(ord("A") + a), gold_letter) for a in actions]
        )
        baseline = rewards.mean()

        # grad of log softmax: (onehot(action) - probs)
        grad = np.zeros_like(theta)
        for a, r in zip(actions, rewards):
            onehot = np.zeros(n_choices)
            onehot[a] = 1.0
            grad += (r - baseline) * (onehot - probs)
        grad /= batch
        theta += lr * grad

        curve.append(float(rewards.mean()))
    return curve, probs, gold_idx


# --------------------------------------------------------------------------- #
# Demo B: 1-parameter numeric guesser  (reward-weighted / REINFORCE on mean)
# --------------------------------------------------------------------------- #
def demo_numeric(n_iters: int = 800, lr: float = 0.8, tol: float = 3.0):
    """Policy = Gaussian(mu, sigma) numeric guesser; only `mu` is learned.

    Synthetic task: gold numeric answer (e.g. an object count / duration).
    Reward = numeric_tolerance(pred, gold, tol, shaped=True) -- dense inside the
    band. Update = REINFORCE on the Gaussian mean:
        mu += lr * mean[ (R - baseline) * (a - mu) / sigma^2 ].
    Expected: mu drifts toward gold, mean shaped reward -> near 1.0.

    `sigma` starts wide enough that the initial guess distribution overlaps the
    reward band (otherwise the reward is sparse and REINFORCE gets no signal --
    a real, honest RLVR exploration constraint, not a bug), then anneals toward
    the tolerance so the policy can sharpen once it has found the band.

    Returns a list of per-iteration mean rewards.
    """
    rng = np.random.default_rng(SEED + 1)
    gold = 7.0
    mu = 0.0
    sigma0, sigma_min = 6.0, 1.0
    batch = 64

    curve = []
    for t in range(n_iters):
        sigma = max(sigma_min, sigma0 * (1.0 - t / n_iters))
        actions = rng.normal(mu, sigma, size=batch)
        rewards = np.array(
            [numeric_tolerance(a, gold, tol=tol, shaped=True) for a in actions]
        )
        baseline = rewards.mean()
        grad = np.mean((rewards - baseline) * (actions - mu) / (sigma ** 2))
        mu += lr * grad
        curve.append(float(rewards.mean()))
    return curve, mu, gold


# --------------------------------------------------------------------------- #
def _summarize(name, curve, k=20):
    first = float(np.mean(curve[:k]))
    last = float(np.mean(curve[-k:]))
    peak = float(max(curve))
    print(f"  {name}")
    print(f"    first-{k}-mean : {first:.3f}")
    print(f"    last-{k}-mean  : {last:.3f}")
    print(f"    peak          : {peak:.3f}")
    print(f"    delta         : {last - first:+.3f}  "
          f"({'IMPROVED' if last > first + 0.05 else 'no clear improvement'})")
    return first, last


def _sparkline(curve, width=50):
    xs = np.linspace(0, len(curve) - 1, width).astype(int)
    vals = [curve[i] for i in xs]
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((v - lo) / rng * 7))] for v in vals)


def main():
    np.random.seed(SEED)
    print("=" * 70)
    print("RLVR verifier-reward policy-improvement demo  [toy / [measured]-on-synthetic]")
    print("NOT a real VLA/LLM RL run -- numpy toy policies, verifier rewards only.")
    print("=" * 70)

    mc_curve, mc_probs, gold_idx = demo_multiple_choice()
    num_curve, mu, gold = demo_numeric()

    print("\nReward vs iteration (mean batch reward; left=start, right=end):")
    print(f"  MC bandit (REINFORCE):  {_sparkline(mc_curve)}")
    print(f"  numeric guesser      :  {_sparkline(num_curve)}")

    print("\nDemo A -- multiple_choice_exact (softmax bandit, REINFORCE)")
    a_first, a_last = _summarize("multiple-choice", mc_curve)
    print(f"    final P(gold letter '{chr(ord('A')+gold_idx)}') = {mc_probs[gold_idx]:.3f}")

    print("\nDemo B -- numeric_tolerance (Gaussian mean, REINFORCE)")
    b_first, b_last = _summarize("numeric", num_curve)
    print(f"    final mu = {mu:.3f}  (gold = {gold})")

    both_up = (a_last > a_first + 0.05) and (b_last > b_first + 0.05)
    print("\n" + "=" * 70)
    print(f"SUMMARY: MC {a_first:.3f} -> {a_last:.3f} | "
          f"numeric {b_first:.3f} -> {b_last:.3f} | "
          f"{'BOTH CURVES IMPROVED (reward engine closes the loop)' if both_up else 'CHECK: no clear improvement'}")
    print("=" * 70)
    return 0 if both_up else 1


if __name__ == "__main__":
    raise SystemExit(main())
