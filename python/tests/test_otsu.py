import numpy as np
from numpy.testing import assert_allclose

from light_curve_pure import Otsu


def test_otsu_1():
    feature = Otsu(bins_number=5)
    m = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 5.5, 5.5, 5.5, 5.5, 0.5])
    t = np.linspace(0, 1.7, 18)
    desired_t_1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.7])
    desired_m_1 = np.array([])
    actual = feature(t, m, None)
    desired = 3.0
    print(actual)
    assert_allclose(actual, desired)


def test_otsu_2():
    pass
