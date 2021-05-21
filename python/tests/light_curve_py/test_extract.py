from numpy.testing import assert_allclose
import numpy as np

from light_curve.light_curve_py import Extractor, Mean, Cusum, LinearTrend


def test_extractor():
    f = Extractor((Mean(), Cusum(), LinearTrend()))
    feature = Mean()
    feature_2 = Cusum()
    feature_3 = LinearTrend()
    a = np.array([0, 1, 2, 3, 4])
    desired = np.hstack((feature(a, a, None), feature_2(a, a, None), feature_3(a, a, None)))
    actual = f(a, a, None)
    assert_allclose(actual, desired)
