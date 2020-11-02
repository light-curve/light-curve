import numpy as np
from numpy.testing import assert_allclose

from light_curve_pure import Cusum


def test_cusum():
    m = [1, 2, 3, 4, 5, 5]
    feature = Cusum()
    actual = feature(m, m, None)
    desired = 0.408248290463863
    assert_allclose(actual, desired)
