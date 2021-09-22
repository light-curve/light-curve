import numpy as np
from numpy.testing import assert_allclose
from light_curve.light_curve_py import Extractor, Amplitude, OtsuSplit, Eta
from light_curve.light_curve_py.features._base import BaseFeature


class PythonFeature(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return m

    @property
    def size(self):
        return 1


def test_extractor_1():  # all python features
    t = np.linspace(0.0, 1.0, 50)
    perfect_m = 1e3 * t + 1e2
    err = np.sqrt(perfect_m)
    m = perfect_m + np.random.normal(0, err)

    otsusplit = OtsuSplit()
    testfeature = PythonFeature()
    extractor = Extractor((otsusplit, testfeature))
    actual = extractor(t, m, err)
    desired = np.concatenate([otsusplit(t, m, err), testfeature(t, m, err)])
    assert np.all(actual == desired)


def test_extractor_2():  # all rust features
    t = np.linspace(0.0, 1.0, 50)
    perfect_m = 1e3 * t + 1e2
    err = np.sqrt(perfect_m)
    m = perfect_m + np.random.normal(0, err)

    amplitude = Amplitude()
    eta = Eta()
    extractor = Extractor((amplitude, eta))
    actual = extractor(t, m, err)
    desired = np.concatenate([np.atleast_1d(amplitude(t, m, err)), np.atleast_1d(eta(t, m, err))])
    assert np.all(actual == desired)


def test_extractor_3():  # python / rust mixed
    t = np.linspace(0.0, 1.0, 50)
    perfect_m = 1e3 * t + 1e2
    err = np.sqrt(perfect_m)
    m = perfect_m + np.random.normal(0, err)

    amplitude = Amplitude()
    otsusplit = OtsuSplit()
    extractor = Extractor((amplitude, otsusplit))
    actual = extractor(t, m, err)
    desired = np.concatenate([np.atleast_1d(amplitude(t, m, err)), np.atleast_1d(otsusplit(t, m, err))])
    assert np.all(actual == desired)
