import numpy as np

from ._base import BaseFeature


class MeanVariance(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        return np.std(m, ddof=1) / np.mean(m)


__all__ = ("MeanVariance",)
