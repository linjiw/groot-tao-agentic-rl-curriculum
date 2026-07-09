# SONIC tier-0 σ-EMA shim — launch recipe (G0/G2)

Insert `SigmaEMAController` into live GR00T-WBC training via a Hydra `func`
override + PYTHONPATH, **no pinned-submodule edits**. Verified end-to-end on
CPU (import, signature contract, numeric identity) 2026-07-09; the only
untested-until-G0 piece is behaviour under the running sim (needs GPU).

## 1. Deploy the shim into a container-visible path

The container mounts `/workspace` (host `/workspace` == container `/workspace`)
but NOT the host repo path. So copy the needed packages under `/workspace`:

```bash
RMC=/home/ec2-user/work/groot-tao-agentic-rl-curriculum/experiments/run-manager-core
DEPLOY=/workspace/rmc_tier0
mkdir -p $DEPLOY/core $DEPLOY/adapters/sonic_tier0
cp $RMC/core/*.py                       $DEPLOY/core/
cp $RMC/adapters/__init__.py            $DEPLOY/adapters/
cp $RMC/adapters/sonic_tier0/*.py       $DEPLOY/adapters/sonic_tier0/
```

(Re-run after any shim edit. The container writes root-owned `__pycache__`;
clean via `docker exec isaac-lab-base rm -rf /workspace/rmc_tier0/**/__pycache__`,
never from the host.)

## 2. Launch overrides

Add to the training command (via `job_adapter.build_train_command`
`extra_overrides=`), prefixing the interpreter with the PYTHONPATH:

```
PYTHONPATH=/workspace/rmc_tier0:$PYTHONPATH   # env, before /isaac-sim/python.sh
++manager_env.rewards.tracking_anchor_pos.func=adapters.sonic_tier0.sonic_sigma_ema_term:SigmaEMAAnchorPos
```

Behaviour is switched by env vars (NOT new Hydra keys — keeps the pinned
config schema untouched):

| Env var | Default | Meaning |
|---|---|---|
| `SONIC_TIER0_ACTIVE` | `0` | `0`=NO-OP (bit-identical to stock, the G0/stock arm); `1`=ACTIVE (PBHC σ-EMA) |
| `SONIC_TIER0_EMA_RATE` | `0.001` | σ-EMA smoothing rate α (meta-knob) |
| `SONIC_TIER0_SIGMA_FLOOR` | `0.1` | σ floor as a FRACTION of the term's std (meta-knob) |
| `SONIC_TIER0_SIDECAR_DIR` | (unset) | dir for σ-state persistence across segment resumes (F8) |
| `SONIC_TIER0_TRACE` | (unset) | append-JSONL σ-trace path (journal) |
| `SONIC_TIER0_LOG_EVERY` | `0` | trace every N steps (0=off) |

## 3. G0 gate-0 acceptance (the whole point)

Two 10-iter smoke segments from the same checkpoint/seed:
- **stock**: no override at all.
- **noop**: the override above with `SONIC_TIER0_ACTIVE=0`.

Then `core.equivalence.compare_journals(stock, noop)` MUST return
`bit_identical`. If not, the *insertion itself* perturbs training (import
side-effects, op reordering) and must be fixed before any σ-EMA claim.

Rationale it should pass: in NO-OP mode `__call__` returns the stock
function's output object unchanged (`return r_stock`), and the stock
function is imported by name and called with identical args — same ops,
same order.

## 4. G2 (after G0 + a library-native warm start)

`stock` vs `σ-EMA` (`SONIC_TIER0_ACTIVE=1`), 3 seeds × 10 segments, with
`SONIC_TIER0_SIDECAR_DIR` set so σ persists across the segment relaunches.
Endpoint + decision rule per doc 10 §4-G2. σ trajectory journaled via the
trace file feeds the digest / the doc-10 per-motion decomposition.

## Files
- `sigma_ema_kernel.py` — torch-free numeric core (`r_active = r_stock ** (std²/σ²)`; no-op = exponent 1.0).
- `sigma_ema_binding.py` — `SigmaEMAController` + SONIC unit mapping (std↔σ) + atomic sidecar persistence.
- `sonic_sigma_ema_term.py` — torch/isaaclab `ManagerTermBase` reward-term (the CLI `func` target).
- Tests: `../../tests/test_sonic_tier0.py` (32 tests, host-CPU) — kernel identity, PBHC monotonicity, sidecar resume, and the isaaclab signature contract.
