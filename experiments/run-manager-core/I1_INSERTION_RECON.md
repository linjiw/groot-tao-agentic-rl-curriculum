# I1 — Tier-0 insertion feasibility recon (2026-07-09)

Goal (doc 10 §3-I1): can the two tier-0 controllers (`SigmaEMAController`,
`BinLPSampler`) be inserted into live SONIC/GR00T-WBC training via **Hydra config
overrides + PYTHONPATH-visible modules**, with **zero edits to the pinned submodule**
`/workspace/GR00T-WholeBodyControl`? All findings [verified] by reading code or running
a harmless CPU-only container probe (no GPU allocation).

## Verdict

| Seam | Controller | Hydra-swappable? | Verdict |
|---|---|---|---|
| Reward-term `func` | SigmaEMAController (σ-EMA) | **YES** | **GREEN — critical path (G0→G2) unblocked** |
| Motion-lib class | BinLPSampler (bin-LP) | NO (hardcoded class) | AMBER — needs a monkeypatch entrypoint; **deferred to G3** |

**Decision:** σ-EMA alone carries G0/G1/G2 (doc 10 explicitly makes G2 σ-EMA-only, the
PBHC controller with the strongest external evidence; bin-LP is the G3 second arm).
The AMBER seam does **not** block the gate. No STOP-rule trigger: neither insertion
requires editing the submodule.

## Verified facts

### F1 — PYTHONPATH injection works through the training interpreter [verified, probed]
`/isaac-sim/python.sh` (the interpreter `job_adapter.build_train_command` uses,
`job_adapter.py:42,118`) imports a module placed under a `/workspace` dir when
`PYTHONPATH` is prepended:
```
docker exec isaac-lab-base bash -c "cd /workspace/GR00T-WholeBodyControl && \
  PYTHONPATH=/workspace/_i1_probe:$PYTHONPATH /isaac-sim/python.sh -c \
  'import i1_probe_mod; print(i1_probe_mod.VALUE)'"   →  IMPORT_OK 42
```
Host `/workspace` == container `/workspace` (bind mount, verified). So a controller
module living under our repo can be exposed by mounting/symlinking it to a `/workspace`
dir and setting `PYTHONPATH`. **Caveat:** the container writes `__pycache__/*.pyc` as
root — put the shim in a dir where root-owned `.pyc` is acceptable, or ship it as a
directory we don't need to `rm` from the host (use docker-exec to clean, per the E0
root-owned-file pattern already in the driver).

### F2 — Reward-term func is a CLI-overridable `module:attr` string [verified]
- Reward terms in the resolved config are
  `{_target_: isaaclab.managers.RewardTermCfg, func: "gear_sonic.envs.manager_env.mdp:tracking_anchor_pos_error", weight, params:{command_name, std}}`
  (real run config `cmp_control_seed42/control_s1-.../config.yaml:666-679`).
- isaaclab resolves the string via `manager_base._resolve_common_term_cfg` →
  `string_to_callable(name)` which does `importlib.import_module(mod_name); getattr(mod, attr)`
  on ANY `module:attr` (`isaaclab/utils/string.py:138-170`). No allowlist.
- Therefore `++manager_env.rewards.tracking_anchor_pos.func=<our_mod>:<our_attr>`
  swaps the reward function. (Same `++` override family as the live-verified
  `++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.uniform_sampling_rate`.)

### F3 — RewardManager supports STATEFUL callable classes [verified]
`reward_manager.py`: `_prepare_terms` appends terms whose `func isinstance ManagerTermBase`
to `_class_term_cfgs` (`:246`); `reset(env_ids)` is called on them (`:125`); `compute(dt)`
calls `term_cfg.func(self._env, **term_cfg.params)` **every env step** (`:150`).
`ManagerTermBase` (`manager_base.py:31`) is the callable-class base: `__init__(cfg, env)`,
`reset()`, `__call__(*args)`. So a σ-EMA reward term can be a `ManagerTermBase` subclass
holding σ state across steps — the idiomatic isaaclab pattern. (A plain module-level
function keyed by a global also works and is simpler for a first no-op smoke.)

### F4 — Reward cadence = per env step, vectorized over envs [verified]
`RewardManager.compute(dt)` runs once per env step; `term_cfg.func` returns a
`(num_envs,)` tensor. PBHC's per-step σ update maps directly: the term computes the
per-env squared error (e.g. `tracking_anchor_pos_error`'s `sq_dist`, `rewards.py:77`),
feeds a scalar reduction (mean over envs) to the σ-EMA, then returns `exp(-sq_dist/σ²)`.

### F5 — Bit-identical no-op design [verified by construction]
Safest no-op = the shim **calls the original SONIC function unchanged** and only
*observes* the error to advance its EMA; while σ is configured == the original `std`
and updates are disabled, the returned tensor is bit-identical (same op, same order,
`torch.exp(-sq_dist/(std*std))`). This is the G0 gate-0 check
(`compare_journals(stock, shim_noop) == bit_identical`). Reimplementing the kernel is
rejected (float-order risk). See `sonic_sigma_ema_term.py` design below.

### F6 — Motion-lib class is HARDCODED, not `_target_`-instantiated [verified] → AMBER
`gear_sonic/config/manager_env/commands/terms/motion.yaml` `motion_lib_cfg` has **no
`_target_`**; the class is chosen in code:
`commands.py:35` `from gear_sonic.utils.motion_lib import motion_lib_robot` and
`commands.py:231` `self.motion_lib = motion_lib_robot.MotionLibRobot(motion_lib_cfg, ...)`
(also `:638`). So bin-LP cannot be a Hydra `_target_` swap. Non-submodule-editing route
(for G3): a wrapper training entrypoint under PYTHONPATH that imports SONIC's
`train_agent_trl` main, monkeypatches `motion_lib_robot.MotionLibRobot` with our
subclass (which overrides `update_adaptive_sampling_probabilities`,
`motion_lib_base.py:2558-2586`, keeping the `uniform_sampling_rate` grounding mix
`:2578-2585`), then calls main. Feasible, but more surface area → deferred to G3 per the
doc-10 sequencing (G2 is σ-EMA-only).

### F7 — σ logging seam for the journal [verified]
`job_adapter.parse_console_log` reads the rich console box, mapping labels →
train-stream keys via `_LABEL_MAP` (`job_adapter.py:256-262`, e.g. "Mean rewards" →
Episode/rew_mean) with regex `_KV = ([A-Za-z_][\w/ ]*?):\s*(-?\d+...)` (`:251`). A shim
that prints `sigma_ema_anchor_pos: <value>` once per iteration would be picked up by the
same regex if we add one `_LABEL_MAP` entry (a tiny, test-covered extension on OUR side).
Alternative: write a sidecar `sigma_trace.jsonl` under the experiment dir. Recommend the
sidecar (decouples from console formatting, survives log rotation).

### F8 — σ state across segment resumes [verified need, design]
Segments relaunch from `last.pt` with knobs constant (`job_adapter` segment lifecycle).
The reward term is reconstructed each segment → σ would reset unless persisted. For G2
(σ-EMA), persist σ to a sidecar file keyed by experiment/arm and reload on
construction; the meta-knobs (ema_rate, sigma_floor) come from config each segment.
The no-op smoke (single 10-iter segment) doesn't exercise this; note it for the G2 driver.

### F9 — isaaclab statically validates the term `__call__` signature [verified, container probe] — **live-only bug caught**
`manager_base._resolve_common_term_cfg` (`:340-378`) does, for a class term:
`func_static = cls.__call__`; if `len(sig_args) > min_argc(=2)`, it asserts
`set(args[2:]) == set(term_params + args_with_defaults)`. A bare `def __call__(self,
env, **kwargs)` presents a param literally named `kwargs`, so the set-equality FAILS at
training startup (before a single GPU step). Fix: declare the params EXPLICITLY —
`__call__(self, env, command_name, std, body_names=None)` — matching the three wrapped
tracking funcs' shared signature. Container probe confirms lhs==rhs=={command_name, std,
body_names} after the fix. And the class is instantiated as
`term_cfg.func(cfg=term_cfg, env=self._env)` (`manager_base.py:418`) → our
`__init__(self, cfg, env)`. This is precisely the class of failure the CPU spike exists
to catch. Regression-locked by `test_sonic_tier0.py::TestIsaacLabContract`.

### F10 — the shim must be COPIED under /workspace, not symlinked [verified, container probe]
The container's `/workspace` bind mount does NOT include the host repo path
(`/home/ec2-user/...`), so a host symlink `/workspace/rmc_tier0 -> <repo>` is a dangling
path inside the container. Deploy = `cp` the `core/` + `adapters/sonic_tier0/` packages
into `/workspace/rmc_tier0` (recipe in `adapters/sonic_tier0/README.md`). Container
writes root-owned `__pycache__` there — clean via docker-exec, never host `rm`.

### F11 — `_HAVE_TORCH=False` in a bare interpreter is a FALSE negative [verified]
`from isaaclab.managers import ManagerTermBase` fails with `ModuleNotFoundError: pxr` in
a bare `/isaac-sim/python.sh` (USD bindings load only after `AppLauncher` starts the
sim). During real training the reward `func` string is resolved AFTER the sim launches,
so the import succeeds and `ManagerTermBase` is the real base. The `ManagerTermBase =
object` host fallback only lets us introspect signatures in tests; it is never the base
in-container. (torch itself imports fine: 2.7.0+cu128.)

## Build plan (this spike delivers items 1-2; 3 is G0)

1. `adapters/sonic_tier0/sonic_sigma_ema_term.py` — a `ManagerTermBase` reward-term
   shim wrapping `SigmaEMAController` around SONIC's tracking-error kernel; a NO-OP
   mode (σ frozen == std) proven bit-identical, and an ACTIVE mode (per-step
   σ←min(σ,EMA(err)) floor-clamped). CPU-unit-tested against a fake env.
2. `adapters/sonic_tier0/README.md` + the exact Hydra override strings and the
   PYTHONPATH launch recipe; extend `job_adapter` mapping notes (no code change to the
   pinned submodule).
3. **G0** (GPU, after pilot frees box): 10-iter smoke with the no-op shim swapped in;
   assert resolved config shows our `func`, and `compare_journals(stock, noop)` ==
   `bit_identical`. This is the gate-0 acceptance in doc 10 §4-G0.

## Adversarial review (2026-07-09) — one defensive fix applied

An independent review of the numeric core confirmed the load-bearing claims
(no-op bit-identity, unit mapping std↔σ, the `r_active = r_stock**(std²/σ²)` identity,
and no reachable raise-path from normal training) and found ONE real gap: `_load`
restored σ-state without reconciling against the freshly-constructed `sigma_init=std`.
A σ sidecar written under a different `std` (config changed std across segments, or a
foreign/copied sidecar) could restore `σ > std` → retune exponent < 1 → **reward
inflated above stock**, silently breaking the monotone/no-op invariants. Fixed: `_load`
now (a) REFUSES a sidecar whose stored `std` ≠ current std (fresh start), and (b) clamps
restored σ to `[floor, std]` so a corrupt payload can never produce exponent < 1.
Regression-locked by `test_load_refuses_sidecar_from_different_std` +
`test_load_clamps_corrupt_sigma_above_std`. Not reachable on a single-config resume, but
G2 persists σ across segments so the guard matters. (34 tier-0 tests, 197 core total.)

## Anti-findings / risks

- **R1 (F1 caveat):** root-owned `.pyc` in the shim dir — mitigate with docker-exec
  cleanup or a read-only shim dir; never `rm` root files from host.
- **R2 (F8):** σ resume-persistence is real work for the G2 driver, not the smoke.
- **R3:** bin-LP (G3) monkeypatch entrypoint is more invasive; re-review before G3.
- **R4:** the σ-EMA "tracking error" must be defined precisely (which term(s), what
  reduction). First cut: wrap the single `tracking_anchor_pos` term (its `sq_dist`),
  mean-over-envs feeding one scalar σ. Multi-term σ is a later refinement.
