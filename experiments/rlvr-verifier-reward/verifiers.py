"""RLVR verifier-reward reference implementation (the "reward engine").

A family of *deterministic*, *pure* verifiable-reward functions over structured
QA answers. Each function returns a scalar reward in ``[0, 1]`` given a
``prediction`` and a ``gold`` reference. These are the "verifier" half of an
RL-from-Verifiable-Rewards (RLVR) loop: the RL chassis (rollouts, PPO/REINFORCE)
proposes structured answers, and these functions score them without any learned
reward model.

Design context (docs/design/07-review-and-revised-roadmap.md, section 5, line 104):
    "a rule/verifier reward over grounded/structured QA (multiple-choice,
     numeric, IoU, referring-expression match) using standard RLVR."

Each verifier is tied to a TAO data skill that *already ships* and produces the
kind of structured annotation the verifier consumes. Those skills act as the
"teacher" that emits gold references (via a VLM annotator); the verifier then
scores a policy's prediction against that gold. Skill provenance is documented
per function and cross-checked against each skill's SKILL.md (see RESULTS.md).

All functions are:
  * pure       -- no I/O, no global state, no mutation of inputs.
  * total      -- defined on their whole documented input domain (clamped).
  * bounded    -- output is always a float in [0.0, 1.0].
  * numpy/stdlib only.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence


# --------------------------------------------------------------------------- #
# 1. Multiple-choice exact match
# --------------------------------------------------------------------------- #
def multiple_choice_exact(prediction, gold) -> float:
    """Reward 1.0 iff the predicted choice matches the gold choice, else 0.0.

    Accepts either a letter (``"A"``, ``"b"``) or a 0-based/label index. Letters
    are compared case-insensitively after stripping whitespace and any trailing
    punctuation (e.g. ``"B."`` -> ``"B"``). Integers are compared directly.
    A letter and an int are considered equal when the letter's 0-based ordinal
    equals the int (``"A" == 0``, ``"C" == 2``).

    TAO data skill: **tao-generate-video-reasoning-annotations**. Its Step-4
    output emits ``mcq.json`` / ``bcq.json`` files in the ``tao-vl-reason-v1.0``
    envelope, where each item has a ``question`` and a gold ``answer`` for
    multiple-choice / binary-choice questions. This verifier scores a policy's
    chosen option against that gold answer.

    Args:
        prediction: predicted choice -- ``str`` letter/label or ``int`` index.
        gold: gold choice -- ``str`` letter/label or ``int`` index.

    Returns:
        1.0 on exact match, else 0.0.
    """

    def _norm(x):
        if isinstance(x, bool):
            # avoid bool being treated as int silently
            return int(x)
        if isinstance(x, int):
            return x
        s = str(x).strip().strip(".)").strip().upper()
        if len(s) == 1 and s.isalpha():
            return ord(s) - ord("A")  # letter -> 0-based ordinal
        # numeric string like "2"
        try:
            return int(s)
        except ValueError:
            return s  # fall back to raw normalized string

    return 1.0 if _norm(prediction) == _norm(gold) else 0.0


# --------------------------------------------------------------------------- #
# 2. Numeric tolerance (shaped)
# --------------------------------------------------------------------------- #
def numeric_tolerance(
    prediction: float,
    gold: float,
    tol: float,
    shaped: bool = True,
) -> float:
    """Reward for a numeric answer within an absolute tolerance ``tol``.

    Let ``err = |prediction - gold|``.
      * If ``err > tol``  -> reward 0.0 (outside the acceptance band).
      * If ``err <= tol`` and ``shaped=False`` -> reward 1.0 (hard pass/fail).
      * If ``err <= tol`` and ``shaped=True``  -> linearly shaped reward
        ``1 - err/tol`` in ``[0, 1]`` (exact hit = 1.0, at the tolerance
        boundary = 0.0). Shaping gives a denser gradient for the optimizer
        while remaining a pure rule.

    ``tol`` must be > 0.

    TAO data skill: **tao-generate-video-reasoning-annotations**. Its Step-4
    QA output includes open-ended / counting-style questions whose ``answer``
    is a number (counts, durations, temporal-localization timestamps). This
    verifier scores a numeric prediction against that gold number with a
    task-appropriate tolerance.

    Args:
        prediction: predicted numeric value.
        gold: gold numeric value.
        tol: absolute tolerance (must be > 0).
        shaped: if True return the linearly-shaped reward inside the band;
            if False return a hard 1.0/0.0.

    Returns:
        Reward in [0.0, 1.0].
    """
    if tol <= 0:
        raise ValueError("tol must be > 0")
    err = abs(float(prediction) - float(gold))
    if err > tol:
        return 0.0
    if not shaped:
        return 1.0
    return max(0.0, min(1.0, 1.0 - err / tol))


# --------------------------------------------------------------------------- #
# 3. IoU reward (bbox)
# --------------------------------------------------------------------------- #
def iou_reward(pred_bbox: Sequence[float], gold_bbox: Sequence[float]) -> float:
    """Real intersection-over-union of two axis-aligned boxes, as reward.

    Boxes are ``[x1, y1, x2, y2]`` in a consistent coordinate space (e.g. pixel
    space). Coordinates are normalized so that ``x1 <= x2`` and ``y1 <= y2``
    (swapped if given inverted). IoU is ``intersection_area / union_area`` and
    is already in ``[0, 1]``: 1.0 for identical boxes, 0.0 for disjoint boxes.
    Degenerate boxes with zero area yield 0.0 union -> reward 0.0.

    TAO data skill: **tao-generate-image-grounding** (and, for grouped grounding
    phrases, **tao-generate-referring-expressions**). Image-grounding Step-1
    output stores ``expressions[].instances[].bbox = [x1,y1,x2,y2]`` in pixel
    space; referring-expressions Step-2 output stores the same
    ``expressions[].instances[].bbox`` shape. This verifier scores a policy's
    predicted bbox against the gold bbox for a given referring expression.

    Args:
        pred_bbox: predicted box ``[x1, y1, x2, y2]``.
        gold_bbox: gold box ``[x1, y1, x2, y2]``.

    Returns:
        IoU in [0.0, 1.0].
    """
    px1, py1, px2, py2 = _order_box(pred_bbox)
    gx1, gy1, gx2, gy2 = _order_box(gold_bbox)

    # intersection
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    area_g = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = area_p + area_g - inter
    if union <= 0.0:
        return 0.0
    return max(0.0, min(1.0, inter / union))


def _order_box(box: Sequence[float]):
    x1, y1, x2, y2 = (float(v) for v in box)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


# --------------------------------------------------------------------------- #
# 4. Referring-expression match (normalized string / token-F1)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_text(s: str) -> list:
    """Lowercase, strip punctuation, drop a small set of articles, tokenize."""
    toks = _TOKEN_RE.findall(str(s).lower())
    stop = {"a", "an", "the"}
    return [t for t in toks if t not in stop]


def referring_expression_match(
    prediction: str,
    gold: str,
    mode: str = "token_f1",
) -> float:
    """String-match reward for a predicted referring expression vs gold text.

    Two modes:
      * ``"exact"``    -> 1.0 iff the normalized token sequences are identical,
        else 0.0. Normalization lowercases, strips punctuation, and removes the
        articles {a, an, the}.
      * ``"token_f1"`` -> bag-of-tokens F1 between prediction and gold over the
        normalized tokens (harmonic mean of precision and recall). This gives a
        graded reward for partial matches -- e.g. "man in red shirt" vs "the man
        wearing a red shirt" scores partial credit rather than 0. Empty-vs-empty
        scores 1.0; empty-vs-nonempty scores 0.0.

    TAO data skill: **tao-generate-referring-expressions**. Its Step-0 region
    expressions and Step-2 grounding expressions emit natural-language
    ``text`` / ``description`` phrases tied to bboxes; the double-check Step-3
    refines them. This verifier scores a policy's generated referring phrase
    against that gold phrase. (A noisier dense fallback would be BERTScore-F1;
    token-F1 is the cheap deterministic rule used here.)

    Args:
        prediction: predicted referring-expression string.
        gold: gold referring-expression string.
        mode: ``"token_f1"`` (default) or ``"exact"``.

    Returns:
        Reward in [0.0, 1.0].
    """
    p_toks = _normalize_text(prediction)
    g_toks = _normalize_text(gold)

    if mode == "exact":
        return 1.0 if p_toks == g_toks else 0.0
    if mode != "token_f1":
        raise ValueError(f"unknown mode: {mode!r}")

    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0

    # multiset (bag) overlap for token-F1
    from collections import Counter

    pc, gc = Counter(p_toks), Counter(g_toks)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Registry -- lets the demo / callers look verifiers up by name.
# --------------------------------------------------------------------------- #
VERIFIERS = {
    "multiple_choice_exact": multiple_choice_exact,
    "numeric_tolerance": numeric_tolerance,
    "iou_reward": iou_reward,
    "referring_expression_match": referring_expression_match,
}
