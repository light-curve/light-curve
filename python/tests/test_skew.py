import numpy as np
from numpy.testing import assert_allclose

from light_curve_pure import Skew


def test_skew():
    m = [1.0, 2.0, 3.0, 50.0, 25.0]
    feature = Skew()
    actual = feature(m, m, None)
    desired = 1.307253786
    assert_allclose(actual, desired)
