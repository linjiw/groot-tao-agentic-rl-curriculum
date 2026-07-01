"""Domain-agnostic curriculum schedule interpolator (SONIC ``schedule_dict`` port).

[verified] SOURCE:
  external/GR00T-WholeBodyControl/gear_sonic/trl/utils/scheduler.py
  function ``update_scheduled_params(obj, scheduler_dict, step, split_char="@")``
  lines 296-353, plus the ``@``-path navigation helpers lines 17-293.

  ``schedule_dict`` is referenced in the design doc
  docs/design/07-review-and-revised-roadmap.md line 89 as the
  "@-path, linear/segment" curriculum-schedule serialization format. The real
  implementation lives in scheduler.py and is used from eval_agent_trl.py
  (``scheduler.update_scheduled_params(schedule_wrapper, config.trainer.schedule_dict, step)``).

WHAT IS PORTED HERE (faithfully, [verified] against source):
  1. The ``linear`` interpolation over ``(seg_steps, seg_vals)`` breakpoints.
  2. The ``segment`` (step/hold) schedule over the same breakpoints.
  3. The ``@``-delimited object-path navigation (attribute / bracket / method
     call), so a schedule target string like
     ``env@event_manager@get_term_cfg('push_robot')['params']['x'][0]`` can be
     resolved and written back on an arbitrary host object.

WHAT IS DELIBERATELY OMITTED / GENERALIZED:
  * ``val_type`` coercion in SONIC uses ``eval(val_type)(val)`` (e.g. "float").
    Reproduced here via a small safe type map instead of ``eval`` on arbitrary
    names, so the module stays dependency-light and safe. Behavior is identical
    for the documented types (float/int/bool/str).
  * ``DictConfig`` (omegaconf) handling collapses to plain ``dict`` handling;
    omegaconf is not a dependency here. dict merge / ``overwrite_dict`` semantics
    are preserved for plain dicts.
  * ``trigger_func`` invocation is preserved (calls a zero-arg method resolved
    via the same @-path navigation) but only when a step lands exactly on a
    breakpoint, matching SONIC.

The interpolation core (``interpolate_schedule``) is pure-python and testable
without any host object, torch, or omegaconf.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Pure interpolation core  ([verified] scheduler.py:306-320)
# --------------------------------------------------------------------------- #

def interpolate_schedule(
    sch_type: str,
    seg_steps: Sequence[float],
    seg_vals: Sequence[Any],
    step: float,
) -> Any:
    """Compute the scheduled value at ``step``.

    Faithful port of the branch logic in SONIC ``update_scheduled_params``
    (scheduler.py:306-320).

    ``linear``:
      Find the last breakpoint index ``i`` with ``seg_steps[i] <= step``. If it is
      the final breakpoint, return ``seg_vals[i]`` (clamp/hold at the end).
      Otherwise linearly interpolate between ``seg_vals[i]`` and
      ``seg_vals[i+1]`` using ``t = (step - seg_steps[i]) / (seg_steps[i+1] - seg_steps[i])``
      clamped to ``[0, 1]``.

    ``segment``:
      Find the last breakpoint index ``i`` with ``seg_steps[i] <= step`` and hold
      ``seg_vals[i]`` (piecewise-constant / step function).

    Args:
        sch_type: "linear" or "segment".
        seg_steps: Ascending breakpoint steps.
        seg_vals: Values at each breakpoint (same length as seg_steps).
        step: Current training step.

    Returns:
        The interpolated / held value at ``step``.

    Note:
        Like SONIC, the ``while step < seg_steps[i]`` search assumes
        ``step >= seg_steps[0]``; for ``step`` below the first breakpoint SONIC
        would underflow the index. We preserve SONIC semantics for
        ``step >= seg_steps[0]`` and additionally clamp to index 0 below the
        first breakpoint to avoid negative indexing.
    """
    if len(seg_steps) != len(seg_vals):
        raise ValueError("seg_steps and seg_vals must have equal length")
    if len(seg_vals) == 0:
        raise ValueError("seg_steps/seg_vals must be non-empty")

    # [verified] SONIC: i = len(seg_vals) - 1; while step < seg_steps[i]: i -= 1
    i = len(seg_vals) - 1
    while i > 0 and step < seg_steps[i]:
        i -= 1

    if sch_type == "linear":
        if i == len(seg_vals) - 1:
            # [verified] final segment: hold last value
            return seg_vals[i]
        # [verified] linear interpolation between i and i+1
        denom = seg_steps[i + 1] - seg_steps[i]
        t = (step - seg_steps[i]) / denom if denom != 0 else 1.0
        t = max(0.0, min(1.0, t))
        return (1.0 - t) * seg_vals[i] + t * seg_vals[i + 1]
    elif sch_type == "segment":
        # [verified] piecewise-constant hold
        return seg_vals[i]
    else:
        raise ValueError(f"Unknown schedule type: {sch_type!r} (expected 'linear' or 'segment')")


# --------------------------------------------------------------------------- #
# Safe value-type coercion (generalizes SONIC's eval(val_type)(val))
# --------------------------------------------------------------------------- #

_VAL_TYPES = {"float": float, "int": int, "bool": bool, "str": str}


def _coerce(val: Any, val_type: str) -> Any:
    """Coerce ``val`` to ``val_type`` via a safe type map.

    SONIC uses ``eval(val_type)(val)``; we restrict to documented types.
    """
    if val_type not in _VAL_TYPES:
        raise ValueError(f"Unsupported val_type: {val_type!r}")
    return _VAL_TYPES[val_type](val)


# --------------------------------------------------------------------------- #
# @-path navigation  ([verified] scheduler.py:17-293, port)
# --------------------------------------------------------------------------- #

def _find_matching(s: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index of the matching close bracket/paren for the one at ``start``."""
    count = 1
    i = start + 1
    while i < len(s) and count > 0:
        if s[i] == open_ch:
            count += 1
        elif s[i] == close_ch:
            count -= 1
        i += 1
    return i - 1


def _evaluate_arg(arg_str: str) -> Any:
    """Evaluate a single function-arg string to a python value.

    Port of scheduler.py:191-221 but without a bare ``eval`` fallback on
    arbitrary text (kept dependency-light and safe): recognizes string, numeric,
    and boolean/None literals; otherwise returns the raw string.
    """
    arg_str = arg_str.strip()
    if (arg_str.startswith("'") and arg_str.endswith("'")) or (
        arg_str.startswith('"') and arg_str.endswith('"')
    ):
        return arg_str[1:-1]
    if arg_str.lstrip("-").replace(".", "", 1).isdigit():
        return float(arg_str) if "." in arg_str else int(arg_str)
    low = arg_str.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none":
        return None
    return arg_str


def _parse_function_args(args_str: str) -> List[Any]:
    """Split and evaluate comma-separated function args (port of scheduler.py:143-188)."""
    if not args_str.strip():
        return []
    args: List[Any] = []
    current = ""
    paren = bracket = 0
    in_quotes = False
    quote_char: Optional[str] = None
    for ch in args_str:
        if ch in ('"', "'") and not in_quotes:
            in_quotes, quote_char = True, ch
            current += ch
        elif ch == quote_char and in_quotes:
            in_quotes, quote_char = False, None
            current += ch
        elif not in_quotes:
            if ch == "(":
                paren += 1; current += ch
            elif ch == ")":
                paren -= 1; current += ch
            elif ch == "[":
                bracket += 1; current += ch
            elif ch == "]":
                bracket -= 1; current += ch
            elif ch == "," and paren == 0 and bracket == 0:
                args.append(_evaluate_arg(current.strip())); current = ""
            else:
                current += ch
        else:
            current += ch
    if current.strip():
        args.append(_evaluate_arg(current.strip()))
    return args


def _bracket_key(content: str) -> Any:
    """Resolve a bracket-access key string to a python key (port of scheduler.py:55-74)."""
    if (content.startswith("'") and content.endswith("'")) or (
        content.startswith('"') and content.endswith('"')
    ):
        return content[1:-1]
    if content.lstrip("-").isdigit():
        return int(content)
    return content


def _process_path_segment(obj: Any, segment: str) -> Any:
    """Resolve one @-delimited segment against ``obj`` (port of scheduler.py:36-114)."""
    current = obj
    i = 0
    while i < len(segment):
        if segment[i] == "[":
            end = _find_matching(segment, i, "[", "]")
            current = current[_bracket_key(segment[i + 1 : end])]
            i = end + 1
        else:
            attr_start = i
            while i < len(segment) and (segment[i].isalnum() or segment[i] == "_"):
                i += 1
            if attr_start < i:
                attr_name = segment[attr_start:i]
                if i < len(segment) and segment[i] == "(":
                    paren_end = _find_matching(segment, i, "(", ")")
                    args = _parse_function_args(segment[i + 1 : paren_end])
                    current = getattr(current, attr_name)(*args)
                    i = paren_end + 1
                elif attr_name.lstrip("-").isdigit():
                    current = current[int(attr_name)]
                else:
                    current = getattr(current, attr_name)
            else:
                i += 1
    return current


def navigate_object_path(obj: Any, path: str, split_char: str = "@") -> Any:
    """Navigate a full @-delimited object path (port of scheduler.py:17-33)."""
    current = obj
    for segment in path.split(split_char):
        current = _process_path_segment(current, segment)
    return current


def _is_complex_path(path: str) -> bool:
    return "[" in path or "("in path


def _get_final_target(obj: Any, target_attr: str) -> Any:
    if _is_complex_path(target_attr):
        return _process_path_segment(obj, target_attr)
    if target_attr.lstrip("-").isdigit():
        return obj[int(target_attr)]
    return getattr(obj, target_attr)


def _set_final_target(obj: Any, target_attr: str, value: Any) -> None:
    """Set the final target value, handling complex paths (port of scheduler.py:236-293)."""
    if not _is_complex_path(target_attr):
        if target_attr.lstrip("-").isdigit():
            obj[int(target_attr)] = value
        else:
            setattr(obj, target_attr, value)
        return
    last_bracket = target_attr.rfind("[")
    last_paren = target_attr.rfind("(")
    if last_bracket > last_paren:
        bracket_end = _find_matching(target_attr, last_bracket, "[", "]")
        parent_path = target_attr[:last_bracket]
        content = target_attr[last_bracket + 1 : bracket_end]
        parent = _process_path_segment(obj, parent_path) if parent_path else obj
        parent[_bracket_key(content)] = value
    else:
        if target_attr.lstrip("-").isdigit():
            obj[int(target_attr)] = value
        else:
            setattr(obj, target_attr, value)


# --------------------------------------------------------------------------- #
# Top-level driver  ([verified] scheduler.py:296-353, port)
# --------------------------------------------------------------------------- #

def update_scheduled_params(
    obj: Any,
    scheduler_dict: Dict[str, Dict[str, Any]],
    step: float,
    split_char: str = "@",
) -> Dict[str, Any]:
    """Apply a ``schedule_dict`` to ``obj`` at ``step`` and return resolved values.

    Faithful port of SONIC ``update_scheduled_params`` (scheduler.py:296-353),
    with omegaconf ``DictConfig`` handling collapsed to plain ``dict`` and the
    ``eval(val_type)`` coercion replaced by a safe type map.

    Each entry maps a target @-path to a config with keys:
      * ``type``: "linear" or "segment"
      * ``seg_steps``: ascending breakpoint steps
      * ``seg_vals``: values at each breakpoint
      * ``val_type`` (optional, default "float"): float/int/bool/str
      * ``overwrite_dict`` (optional): replace vs. merge when value is a dict
      * ``trigger_func`` (optional): @-path to a zero-arg method fired when
        ``step`` lands exactly on the resolved breakpoint.

    Args:
        obj: Host object to mutate.
        scheduler_dict: The schedule specification.
        step: Current training step.
        split_char: Path delimiter (default "@").

    Returns:
        Mapping of each target path to the value written at ``step``.
    """
    scheduled_params: Dict[str, Any] = {}
    for target, cfg in scheduler_dict.items():
        sch_type = cfg["type"]
        val_type = cfg.get("val_type", "float")
        target_attr = target
        target_obj = obj
        if split_char in target:
            target_obj_str, target_attr = target.rsplit(split_char, 1)
            target_obj = navigate_object_path(obj, target_obj_str, split_char)

        val = interpolate_schedule(sch_type, cfg["seg_steps"], cfg["seg_vals"], step)
        val = _coerce(val, val_type)

        if isinstance(val, dict):
            tmp = _get_final_target(target_obj, target_attr)
            if cfg.get("overwrite_dict", False):
                _set_final_target(target_obj, target_attr, val)
            else:
                for k, v in val.items():
                    if isinstance(tmp, dict):
                        tmp[k] = v
                    else:
                        setattr(tmp, k, v)
        else:
            _set_final_target(target_obj, target_attr, val)

        scheduled_params[target] = val

        # [verified] trigger_func on exact breakpoint (scheduler.py:342-351)
        if "trigger_func" in cfg:
            seg_steps = cfg["seg_steps"]
            i = len(seg_steps) - 1
            while i > 0 and step < seg_steps[i]:
                i -= 1
            if step == seg_steps[i]:
                target_func = cfg["trigger_func"]
                if split_char in target_func:
                    tfo_str, tf_name = target_func.rsplit(split_char, 1)
                    tfo = navigate_object_path(obj, tfo_str, split_char)
                else:
                    tfo, tf_name = obj, target_func
                getattr(tfo, tf_name)()

    return scheduled_params
