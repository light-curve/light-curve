import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import Otsu


def test_otsu_1():
    feature = Otsu()
    m = np.array([0.5, 1.5, 1.5, 1.5])
    t = np.linspace(0, 1.7, 4)
    actual = feature(t, m, None)
    desired = -1.0
    assert_allclose(actual, desired)


def test_otsu_2():
    feature = Otsu()
    m = np.array([0.5, 0.5, 0.5])
    t = np.linspace(0, 1.7, 3)
    actual = feature(t, m, None)
    desired = 0.0
    assert_allclose(actual, desired)
