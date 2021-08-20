import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import StetsonJ


def test_stetsonj_1():
    feature = StetsonJ()
    x = np.linspace(0.0, 2 * np.pi, 2000)
    sin = np.sin(x)
    error = np.ones(2000)
    actual = feature(sin, sin, error)
    desired = 63.121
    assert_allclose(actual, desired, rtol=1e-03)
