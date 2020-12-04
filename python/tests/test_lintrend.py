import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import curve_fit

from light_curve_pure import LinearTrend


def test_lintrend_1():
    f = lambda x, slope, c: x * slope + c
    m = np.array([2.0, 4.0, 7.0, 10.0, 15.0, 28.0])
    t = np.array([1.0, 3.0, 6.0, 10.0, 14.0, 17.0])
    n = len(t)

    feature = LinearTrend()
    actual = feature(t, m, None)

    slope, c = curve_fit(f, t, m)[0]
    chisq = np.sum((m - (t * slope + c)) ** 2) / (n - 2)
    sxx = np.var(t) * n

    desired = (slope, np.sqrt(chisq / sxx), chisq)
    assert_allclose(actual, desired)


def test_lintrend_2():
    m = np.array([2.0, 4.0])
    t = np.array([1.0, 2.0])

    feature = LinearTrend()
    actual = feature(t, m, None)
    desired = (2.0, 0, 0)
    assert_allclose(actual, desired)
