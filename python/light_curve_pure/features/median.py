import numpy as np

from ._base import BaseFeature


class Median(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        return np.median(m)


__all__ = ("Median",)
