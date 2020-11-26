import numpy as np
from numpy.testing import assert_allclose

from light_curve_pure import EtaE


def test_etae():
    a = [1, 2, 3, 8]
    t = [1, 2, 3, 4]
    feature = EtaE()
    actual = feature(a, t, None)
    desired = 2.2213333333333334
    assert_allclose(actual, desired)
