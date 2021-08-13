import numpy as np

from ._base import BaseFeature


class MedianAbsoluteDeviation(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return np.median((np.abs(m - np.median(m))))


__all__ = ("MedianAbsoluteDeviation",)
