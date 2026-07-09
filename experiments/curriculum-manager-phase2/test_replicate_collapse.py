# SPDX-License-Identifier: Apache-2.0
"""Tests for replicate_collapse (doc 10 §0.2 / V2).

Includes a REAL-DATA test against the v4 seed-42 control replicate journals
where rep2 == rep3 bit-exactly (the finding that motivated this module):
launched 4, distinct 3.
"""

import json
import math
import os

import pytest

import replicate_collapse as rc

HERE = os.path.dirname(__file__)


class TestSeriesBitIdentical:
    def test_identical_series_match(self):
        a = [0.1, 0.2, 0.3]
        assert rc.series_bit_identical(a, list(a))

    def test_one_quantum_apart_are_distinct(self):
        assert not rc.series_bit_identical([0.1, 0.2, 0.3], [0.1, 0.2, 0.3000001])

    def test_none_vs_number_is_distinct(self):
        # one arm ran a segment, the other didn't -> different runs
        assert not rc.series_bit_identical([0.1, None, 0.3], [0.1, 0.2, 0.3])

    def test_both_none_at_index_ok(self):
        assert rc.series_bit_identical([0.1, None, 0.3], [0.1, None, 0.3])

    def test_min_overlap_guards_stub_merge(self):
        # two runs sharing only the (identical) warm-start seg must NOT merge
        assert not rc.series_bit_identical([0.086], [0.086], min_overlap=2)
        assert rc.series_bit_identical([0.086], [0.086], min_overlap=1)

    def test_nan_never_identical(self):
        assert not rc.series_bit_identical([float("nan")], [float("nan")],
                                           min_overlap=1)

    def test_ragged_lengths_compare_on_overlap(self):
        # rep2 (9 segs) is rep3 (10 segs) shifted — but as raw series they
        # only match where indices align; this checks the overlap semantics
        assert rc.series_bit_identical([0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4])
        assert not rc.series_bit_identical([0.1, 0.2, 0.9], [0.1, 0.2, 0.3, 0.4])


class TestCollapse:
    def test_no_duplicates(self):
        info = rc.collapse_replicates({
            "a": [0.1, 0.2, 0.3],
            "b": [0.1, 0.2, 0.31],
            "c": [0.1, 0.2, 0.29],
        })
        assert info["n_launched"] == 3
        assert info["n_distinct"] == 3
        assert not info["collapsed"]
        assert info["duplicate_note"] is None

    def test_one_duplicate_pair_collapses(self):
        info = rc.collapse_replicates({
            "rep1": [0.08, 0.09, 0.10],
            "rep2": [0.08, 0.11, 0.12],
            "rep3": [0.08, 0.11, 0.12],       # == rep2
            "rep4": [0.08, 0.09, 0.11],
        })
        assert info["n_launched"] == 4
        assert info["n_distinct"] == 3
        assert info["collapsed"]
        assert ["rep2", "rep3"] in info["groups"]
        assert info["representatives"] == ["rep1", "rep2", "rep4"]
        assert "rep2==rep3" in info["duplicate_note"]

    def test_representative_is_first_seen(self):
        info = rc.collapse_replicates({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "z": [1.0, 2.0, 3.0],
        })
        assert info["n_distinct"] == 1
        assert info["representatives"] == ["x"]
        assert info["groups"] == [["x", "y", "z"]]

    def test_distinct_finals_one_per_trajectory(self):
        info = rc.distinct_finals({
            "rep1": [0.08, 0.09, 0.0988],
            "rep2": [0.08, 0.11, 0.1063],
            "rep3": [0.08, 0.11, 0.1063],     # dup of rep2
            "rep4": [0.08, 0.09, 0.0969],
        })
        assert info["n_distinct"] == 3
        # finals from representatives only: rep1, rep2, rep4
        assert info["distinct_finals"] == [0.0988, 0.1063, 0.0969]


def _load_progress_series(path):
    with open(path) as fh:
        j = json.load(fh)
    return [((e or {}).get("eval") or {}).get("progress_rate")
            for e in j if isinstance(e, dict) and e.get("eval")]


class TestRealV4Replicates:
    """The motivating real data: seed-42 control rep2 == rep3 bit-exactly."""

    def _paths(self):
        names = {
            "rep1": "control_journal_v4_seed42.json",
            "rep2": "control_journal_v4_seed42_rep2.json",
            "rep3": "control_journal_v4_seed42_rep3.json",
            "rep4": "control_journal_v4_seed42_rep4.json",
        }
        return {k: os.path.join(HERE, v) for k, v in names.items()}

    def _by_name(self, path):
        with open(path) as fh:
            j = json.load(fh)
        return {e["segment"]: ((e or {}).get("eval") or {}).get("progress_rate")
                for e in j if isinstance(e, dict) and e.get("eval")}

    def test_v4_seed42_raw_index_is_conservative(self):
        # rep2 has 9 eval segments (missing control_s2); rep3 has 10. As RAW
        # index-aligned series they are NOT duplicates (s2 onward is shifted),
        # so raw-index collapse conservatively keeps all 4. This documents
        # that collapse_replicates must NOT be used on ragged real journals.
        paths = self._paths()
        if not all(os.path.exists(p) for p in paths.values()):
            pytest.skip("v4 replicate journals not present")
        named = {k: _load_progress_series(p) for k, p in paths.items()}
        info = rc.collapse_replicates(named)
        assert info["n_launched"] == 4
        assert info["n_distinct"] == 4          # ragged -> conservatively distinct

    def test_v4_seed42_by_segment_collapses_4_to_3(self):
        # The real finding (doc 10 §0.2): keyed by SEGMENT NAME, rep2 == rep3
        # bit-exactly on all 9 shared segments -> distinct = 3, not 4.
        paths = self._paths()
        if not all(os.path.exists(p) for p in paths.values()):
            pytest.skip("v4 replicate journals not present")
        maps = {k: self._by_name(p) for k, p in paths.items()}
        info = rc.collapse_by_segment(maps)
        assert info["n_launched"] == 4
        assert info["n_distinct"] == 3
        assert info["collapsed"]
        # rep2 and rep3 must be in the same group
        grp = [g for g in info["groups"] if "rep2" in g][0]
        assert "rep3" in grp

    def test_v4_seed42_noise_band_over_3_distinct(self):
        # doc 10 §0.2: the band over DISTINCT trajectories (3, not 4) at the
        # final segment; the doc reports ~9.4% relative range / ~5% band.
        paths = self._paths()
        if not all(os.path.exists(p) for p in paths.values()):
            pytest.skip("v4 replicate journals not present")
        maps = {k: self._by_name(p) for k, p in paths.items()}
        band = rc.noise_band_from_replicates(maps, endpoint_segments=["control_s10"])
        assert band["n_launched"] == 4
        assert band["n_distinct"] == 3
        assert band["sigma"] is not None            # >=2 distinct -> real band
        # doc-10 finals {0.0988, 0.1063, 0.0969}; range/mean ~= 0.093
        assert band["rel_range"] == pytest.approx(0.0935, abs=0.01)


class TestNoiseBand:
    def test_refuses_band_with_one_distinct(self):
        band = rc.noise_band_from_replicates(
            {"a": {"s1": 0.1, "s2": 0.2}, "b": {"s1": 0.1, "s2": 0.2}},
            endpoint_segments=["s2"])
        assert band["n_distinct"] == 1
        assert band["sigma"] is None
        assert "need >= 2" in band["band_note"]

    def test_band_over_distinct_only(self):
        # 3 launches, 2 distinct (b==c) -> band over the 2 distinct endpoints
        band = rc.noise_band_from_replicates({
            "a": {"s1": 0.08, "s2": 0.10},
            "b": {"s1": 0.08, "s2": 0.12},
            "c": {"s1": 0.08, "s2": 0.12},   # dup of b
        }, endpoint_segments=["s2"])
        assert band["n_distinct"] == 2
        assert sorted(band["distinct_endpoints"]) == [0.10, 0.12]
        assert band["mean"] == pytest.approx(0.11)
        assert band["range"] == pytest.approx(0.02)

    def test_endpoint_is_mean_over_segments(self):
        band = rc.noise_band_from_replicates({
            "a": {"s9": 0.10, "s10": 0.12},
            "b": {"s9": 0.20, "s10": 0.22},
        }, endpoint_segments=["s9", "s10"])
        # per-run endpoint = mean of the 2 segments: a->0.11, b->0.21
        assert sorted(band["distinct_endpoints"]) == [pytest.approx(0.11),
                                                      pytest.approx(0.21)]
