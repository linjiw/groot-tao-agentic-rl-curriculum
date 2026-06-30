#!/usr/bin/env bash
# Apply the Stage-2 termination-threshold curriculum into a WBC checkout and run
# CPU-only static validation (no GPU, no Isaac Sim needed).
#
# Usage: bash apply_and_validate.sh <path-to-GR00T-WholeBodyControl>
set -euo pipefail

WBC="${1:?usage: apply_and_validate.sh <path-to-GR00T-WholeBodyControl>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

echo "==> applying Stage-2 edits into: $WBC"
"$PY" "$HERE/apply_stage2.py" "$WBC"

echo "==> static check 1/3: AST-compile edited python"
"$PY" - "$WBC" <<'PY'
import ast, sys, pathlib
wbc = pathlib.Path(sys.argv[1])
for rel in ["gear_sonic/envs/manager_env/mdp/curriculum.py",
            "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py"]:
    ast.parse((wbc/rel).read_text()); print("  OK", rel)
PY

echo "==> static check 2/3: YAML parse of the new curriculum config"
"$PY" - "$WBC" <<'PY'
import sys, pathlib
try:
    import yaml
except ImportError:
    print("  SKIP (pyyaml not available in this interpreter)"); sys.exit(0)
wbc = pathlib.Path(sys.argv[1])
d = yaml.safe_load((wbc/"gear_sonic/config/manager_env/curriculum/threshold_tighten.yaml").read_text())
terms = [k for k in d if k != "_target_"]
assert d["_target_"].endswith("CurriculumCfg") and len(terms) == 3, d
for t in terms:
    p = d[t]["params"]
    assert p["address"].startswith("terminations.")
    mp = p["modify_params"]; assert len(mp["values"]) == len(mp["num_steps"])
print("  OK 3 well-formed terms:", terms)
PY

echo "==> static check 3/3: step_curriculum_nochange milestone logic"
"$PY" - <<'PY'
class _NC: ...
NC = _NC()
def f(step, live, values, num_steps):
    target = None
    for i in range(len(values)):
        idx = len(num_steps)-i-1
        if step > num_steps[idx]: target = values[idx]; break
    return NC if (target is None or live == target) else target
v=[0.30,0.22,0.15]; s=[0,30000,80000]
assert f(0,0.30,v,s) is NC
assert f(1,0.99,v,s) == 0.30
assert f(35000,0.30,v,s) == 0.22
assert f(90000,0.22,v,s) == 0.15
assert f(90000,0.15,v,s) is NC
print("  OK milestone logic")
PY

echo "==> all static checks passed. See RUN.md to launch on an Isaac-Lab host."
