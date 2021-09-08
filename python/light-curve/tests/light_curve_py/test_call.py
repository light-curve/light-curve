import numpy as np
from numpy.testing import assert_allclose
from light_curve.light_curve_py import MaximumSlope


def test_normalize():
    t = [1, 4, 3, 8]
    m = [2, 15, 4, 3]
    feature = MaximumSlope()
    actual = feature(t, m, None, sorted=False)
    desired = 11.0
    assert_allclose(actual, desired)


def test_fill_zero_division():
    t = [1, 1, 3, 4]
    feature = MaximumSlope()
    actual = feature(t, t, None, sorted=False, fill_value=[1.0])
    desired = 1.0
    assert_allclose(actual, desired)


def test_fill_nan_values():
    t = [np.nan, 1, 3, 4]
    feature = MaximumSlope()
    actual = feature(t, t, None, sorted=False, fill_value=[1.0])
    desired = 1.0
    assert_allclose(actual, desired)
