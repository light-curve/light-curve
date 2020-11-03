import numpy as np

from ._base import BaseFeature


class WeightedMean(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        return np.average(m, weights=np.power(sigma, 2))


__all__ = ("WeightedMean",)
