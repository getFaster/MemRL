import pytest

from memrl.train import _finite_float


def test_nonfinite_update_metrics_fail_fast():
    assert _finite_float(1.25) == 1.25
    with pytest.raises(FloatingPointError):
        _finite_float(float("nan"))
    with pytest.raises(FloatingPointError):
        _finite_float(float("inf"))
