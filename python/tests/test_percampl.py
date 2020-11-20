from numpy.testing import assert_allclose

from light_curve_pure import PercentAmplitude


def test_medabsdev():
    m = [1.0, 1.0, 3.0, 4.0, 7.0]
    feature = PercentAmplitude()
    actual = feature(m, m, None)
    desired = 4.0
    assert_allclose(actual, desired)
