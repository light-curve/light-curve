import numpy as np
from numpy.testing import assert_allclose

from light_curve.light_curve_py import MagnitudeNNotDetBeforeFd, FluxNNotDetBeforeFd


def test_flux():
    feature = FluxNNotDetBeforeFd(10)
    m = np.array([1, 2, 10, 40, 50])
    sigma = np.array([0.4, 0.2, 0.01, 0.03, 0.02])
    actual = feature(m, m, sigma)
    desired = 2
    assert actual == desired


def test_magnitude():
    feature = MagnitudeNNotDetBeforeFd(-1)
    m = np.array([-1, -1, -1, 40, 50])
    actual = feature(m, m, m)
    desired = 3
    assert actual == desired
