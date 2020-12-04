import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import curve_fit

from light_curve_pure import LinearFit


def test_linfit():
    f = lambda x, slope, c: x * slope + c
    m = np.array([2.0, 4.0, 7.0, 10.0, 15.0, 28.0])
    t = np.array([1.0, 3.0, 6.0, 10.0, 14.0, 17.0])
    sigma = np.array([0.1, 0.1, 0.2, 1.3, 0.4, 1.5])
    n = len(t)

    feature = LinearFit()
    actual = feature(t, m, sigma)

    slope, intercept = curve_fit(f, t, m, sigma=sigma)[0]
    chisq = np.sum(((m - t * slope - intercept) / sigma) ** 2) / (n - 2)
    sxx = np.var(t) * n

    desired = (slope, np.sqrt(chisq / sxx), chisq)
    assert_allclose(actual, desired)
