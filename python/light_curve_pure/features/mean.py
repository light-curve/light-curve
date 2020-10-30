import numpy as np

from ._base import BaseFeature


class Mean(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        return np.mean(m)


__all__ = ("Mean",)
