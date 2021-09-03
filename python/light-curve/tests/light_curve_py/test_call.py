from numpy.testing import assert_allclose

from light_curve.light_curve_py import MaximumSlope, Kurtosis


def test_normalize():
    t = [1, 4, 3, 8]
    m = [2, 15, 4, 3]
    feature = MaximumSlope()
    actual = feature(t, m, None, sorted=False)
    desired = 11.0
    assert_allclose(actual, desired)
