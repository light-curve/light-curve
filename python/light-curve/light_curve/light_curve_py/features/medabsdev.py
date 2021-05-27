import numpy as np

from ._base import BaseFeature


class MedianAbsoluteDeviation(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        return np.median((np.abs(m - np.median(m))))


__all__ = ("MedianAbsoluteDeviation",)
