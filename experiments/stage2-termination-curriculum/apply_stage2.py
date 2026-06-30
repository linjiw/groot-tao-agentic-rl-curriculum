#!/usr/bin/env python3
"""Idempotently apply the Stage-2 termination-threshold curriculum into a
GR00T-WholeBodyControl checkout.

Uses exact-anchor string edits (verified against WBC @0e35637), not line numbers,
so it survives minor upstream drift and is safe to re-run.

Edits:
  1. gear_sonic/envs/manager_env/mdp/curriculum.py
       - add 3 fields to CurriculumCfg
       - append step_curriculum_nochange() helper
  2. gear_sonic/envs/manager_env/modular_tracking_env_cfg.py
       - extend the hardcoded modify_fn injection block for the 3 new terms
  3. copy config/threshold_tighten.yaml into config/manager_env/curriculum/

Usage:  python apply_stage2.py /path/to/GR00T-WholeBodyControl
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def patch_file(path: Path, anchor: str, addition: str, marker: str) -> str:
    text = path.read_text()
    if marker in text:
        return f"  SKIP {path.name}: already patched ({marker!r} present)"
    if anchor not in text:
        raise SystemExit(f"  FAIL {path.name}: anchor not found — upstream drifted?\n    anchor={anchor!r}")
    path.write_text(text.replace(anchor, anchor + addition, 1))
    return f"  OK   {path.name}: inserted after anchor"


def main(wbc: Path):
    cur = wbc / "gear_sonic/envs/manager_env/mdp/curriculum.py"
    inj = wbc / "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py"
    cfg_dst = wbc / "gear_sonic/config/manager_env/curriculum/threshold_tighten.yaml"
    cfg_src = HERE / "config/threshold_tighten.yaml"
    for p in (cur, inj, cfg_src):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    # 1a. add fields to the CurriculumCfg dataclass
    print(patch_file(
        cur,
        anchor="    force_push_curriculum = None\n    force_push_linear_curriculum = None\n",
        addition=(
            "    anchor_pos_threshold_curriculum = None\n"
            "    ee_body_pos_threshold_curriculum = None\n"
            "    foot_pos_xyz_threshold_curriculum = None\n"
        ),
        marker="anchor_pos_threshold_curriculum = None",
    ))

    # 1b. append the NO_CHANGE-aware step helper at end of curriculum.py
    helper = '''

def step_curriculum_nochange(env, env_ids, original_value, values, num_steps):
    """step_curriculum variant that returns ``modify_env_param.NO_CHANGE`` unless
    the target threshold for the current step differs from the live value.

    Avoids a redundant per-step setter write while preserving step_curriculum's
    milestone semantics (highest passed milestone wins; before the first
    milestone the live value is left untouched).
    """
    from isaaclab.envs.mdp.curriculums import modify_env_param

    assert len(values) == len(num_steps)
    step = env.common_step_counter
    target = None
    for i in range(len(values)):
        idx = len(num_steps) - i - 1
        if step > num_steps[idx]:
            target = values[idx]
            break
    if target is None or original_value == target:
        return modify_env_param.NO_CHANGE
    return target
'''
    print(patch_file(
        cur,
        anchor="    # After last milestone → final value\n    return values[-1]\n",
        addition=helper,
        marker="def step_curriculum_nochange",
    ))

    # 2. extend the hardcoded injection block
    inj_anchor = (
        '            self.curriculum.force_push_linear_curriculum.params["modify_fn"] = getattr(\n'
        '                module, "linear_curriculum"\n'
        '            )\n'
    )
    inj_add = '''
        for _curr_name in (
            "anchor_pos_threshold_curriculum",
            "ee_body_pos_threshold_curriculum",
            "foot_pos_xyz_threshold_curriculum",
        ):
            if hasattr(self.curriculum, _curr_name) and getattr(self.curriculum, _curr_name):
                module = importlib.import_module("gear_sonic.envs.manager_env.mdp")
                getattr(self.curriculum, _curr_name).params["modify_fn"] = getattr(
                    module, "step_curriculum_nochange"
                )
'''
    print(patch_file(inj, inj_anchor, inj_add, "anchor_pos_threshold_curriculum"))

    # 3. drop in the config
    if cfg_dst.exists() and cfg_dst.read_text() == cfg_src.read_text():
        print(f"  SKIP {cfg_dst.name}: already present and identical")
    else:
        shutil.copyfile(cfg_src, cfg_dst)
        print(f"  OK   copied {cfg_dst.name} into config/manager_env/curriculum/")

    print("done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve())
