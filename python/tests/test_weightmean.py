import numpy as np
from numpy.testing import assert_allclose

from light_curve_pure import WeightedMean


def test_weightmean():
    a = [2.0, 3.0, 1.0, 9.0, 5.0]
    b = [0.3, 0.4, 0.5, 1, 1]
    feature = WeightedMean()
    actual = feature(a, a, b)
    desired = 5.964
    assert_allclose(actual, desired)
