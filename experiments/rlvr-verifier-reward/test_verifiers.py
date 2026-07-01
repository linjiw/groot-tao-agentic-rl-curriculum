"""Known-answer tests for the RLVR verifier-reward functions.

Run:  python -m pytest test_verifiers.py -v
Fallback (no pytest): python test_verifiers.py  -> runs a self-test.

Every verifier is checked against hand-computed expected values, including
boundary/edge cases. IoU expectations are derived by hand from box geometry.
"""

from __future__ import annotations

import math

from verifiers import (
    iou_reward,
    multiple_choice_exact,
    numeric_tolerance,
    referring_expression_match,
)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# multiple_choice_exact
# --------------------------------------------------------------------------- #
def test_mc_letter_hit():
    assert multiple_choice_exact("B", "B") == 1.0


def test_mc_letter_miss():
    assert multiple_choice_exact("A", "C") == 0.0


def test_mc_case_and_punctuation_insensitive():
    assert multiple_choice_exact("b.", "B") == 1.0
    assert multiple_choice_exact(" c) ", "C") == 1.0


def test_mc_letter_vs_index_equivalence():
    # "A" is 0-based ordinal 0, "C" -> 2
    assert multiple_choice_exact("A", 0) == 1.0
    assert multiple_choice_exact("C", 2) == 1.0
    assert multiple_choice_exact("A", 1) == 0.0


def test_mc_int_vs_int():
    assert multiple_choice_exact(3, 3) == 1.0
    assert multiple_choice_exact(3, 4) == 0.0


# --------------------------------------------------------------------------- #
# numeric_tolerance
# --------------------------------------------------------------------------- #
def test_numeric_exact_hit_shaped():
    assert numeric_tolerance(5.0, 5.0, tol=1.0) == 1.0


def test_numeric_within_band_shaped():
    # err=0.5, tol=1.0 -> 1 - 0.5/1.0 = 0.5
    assert approx(numeric_tolerance(5.5, 5.0, tol=1.0), 0.5)


def test_numeric_at_boundary_shaped_is_zero():
    # err == tol -> shaped reward 0.0 (still counts as "within" but no credit)
    assert approx(numeric_tolerance(6.0, 5.0, tol=1.0), 0.0)


def test_numeric_outside_band():
    assert numeric_tolerance(7.0, 5.0, tol=1.0) == 0.0


def test_numeric_hard_passfail():
    assert numeric_tolerance(5.9, 5.0, tol=1.0, shaped=False) == 1.0
    assert numeric_tolerance(6.1, 5.0, tol=1.0, shaped=False) == 0.0


def test_numeric_bad_tol_raises():
    try:
        numeric_tolerance(1.0, 1.0, tol=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for tol <= 0")


# --------------------------------------------------------------------------- #
# iou_reward
# --------------------------------------------------------------------------- #
def test_iou_identical_boxes():
    assert approx(iou_reward([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)


def test_iou_disjoint_boxes():
    assert iou_reward([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap_known_value():
    # A = [0,0,2,2] area 4; B = [1,0,3,2] area 4.
    # intersection x in [1,2], y in [0,2] -> 1*2 = 2. union = 4+4-2 = 6.
    # IoU = 2/6 = 1/3.
    assert approx(iou_reward([0, 0, 2, 2], [1, 0, 3, 2]), 1.0 / 3.0)


def test_iou_contained_box_known_value():
    # small [0,0,1,1] area1 fully inside big [0,0,2,2] area4.
    # inter=1, union=4 -> 0.25
    assert approx(iou_reward([0, 0, 1, 1], [0, 0, 2, 2]), 0.25)


def test_iou_handles_inverted_coords():
    # inverted corners describe the same box -> IoU 1.0
    assert approx(iou_reward([10, 10, 0, 0], [0, 0, 10, 10]), 1.0)


def test_iou_degenerate_zero_area():
    assert iou_reward([5, 5, 5, 5], [5, 5, 5, 5]) == 0.0


# --------------------------------------------------------------------------- #
# referring_expression_match
# --------------------------------------------------------------------------- #
def test_refexp_exact_after_normalization():
    assert referring_expression_match("The Red Car.", "red car", mode="exact") == 1.0


def test_refexp_exact_miss():
    assert referring_expression_match("blue car", "red car", mode="exact") == 0.0


def test_refexp_token_f1_perfect():
    assert approx(referring_expression_match("red car", "red car"), 1.0)


def test_refexp_token_f1_partial_known_value():
    # pred = "man in red shirt" -> tokens (in-stop removed? "in" kept) 
    # normalized drops only {a,an,the}: pred=[man,in,red,shirt] (4)
    # gold "the man wearing a red shirt" -> [man,wearing,red,shirt] (4)
    # overlap tokens: man,red,shirt = 3
    # precision=3/4, recall=3/4, F1=0.75
    got = referring_expression_match("man in red shirt", "the man wearing a red shirt")
    assert approx(got, 0.75)


def test_refexp_token_f1_no_overlap():
    assert referring_expression_match("blue truck", "red car") == 0.0


def test_refexp_empty_both():
    assert referring_expression_match("", "") == 1.0


def test_refexp_empty_one_side():
    assert referring_expression_match("", "red car") == 0.0


# --------------------------------------------------------------------------- #
# stdlib self-test fallback (only used when pytest is unavailable)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    fns = [g for n, g in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
