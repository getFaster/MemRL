from __future__ import annotations

import numpy as np
import pytest

from memrl.diagnostics import AGE_BIN_EDGES, AgeHistogram, fixed_age_histogram, frame_coverage, table_frame


def test_fixed_age_histogram_uses_exact_contract_edges():
    result = fixed_age_histogram([0, 9, 10, 24, 25, 99_999, 100_000])
    assert result["edges"] == list(AGE_BIN_EDGES)
    assert result["counts"][0] == 2
    assert result["counts"][1] == 2
    assert result["counts"][2] == 1
    assert result["counts"][-1] == 2
    assert result["total"] == 7


def test_histogram_is_mergeable_and_rejects_out_of_range():
    left = AgeHistogram()
    right = AgeHistogram()
    left.update([1, 12])
    right.update([30])
    left.merge(right)
    assert left.to_dict()["total"] == 3
    with pytest.raises(ValueError, match="outside"):
        left.update([100_001])


def test_frame_coverage_and_missing_table_images():
    valid = np.asarray([True, False, True, False])
    occupied = np.asarray([True, True, False, False])
    frames = np.arange(16, dtype=np.uint8).reshape(4, 2, 2)
    assert frame_coverage(valid) == 0.5
    assert frame_coverage(valid, occupied) == 0.5
    np.testing.assert_array_equal(table_frame(frames, valid, 0), frames[0])
    assert table_frame(frames, valid, 1) is None
    assert table_frame(None, valid, 0) is None
