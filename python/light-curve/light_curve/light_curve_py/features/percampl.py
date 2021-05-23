import numpy as np

from ._base import BaseFeature


class PercentAmplitude(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        median = np.median(m)
        return np.max((np.max(m) - median, median - np.min(m)))


__all__ = ("PercentAmplitude",)
