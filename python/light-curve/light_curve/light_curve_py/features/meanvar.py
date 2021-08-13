import numpy as np

from ._base import BaseFeature


class MeanVariance(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return np.std(m, ddof=1) / np.mean(m)


__all__ = ("MeanVariance",)
