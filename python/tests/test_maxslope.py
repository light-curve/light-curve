import numpy as np
from numpy.testing import assert_allclose

from light_curve_pure import MaximumSlope


def test_maxslope():
    m = [2, 4, 15, 3]
    t = [1, 3, 8, 4]
    feature = MaximumSlope()
    actual = feature(t, m, None)
    desired = 3.0
    print(actual)
    assert_allclose(actual, desired)
