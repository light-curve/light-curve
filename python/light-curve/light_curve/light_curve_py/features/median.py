import numpy as np

from ._base import BaseFeature


class Median(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return np.median(m)


__all__ = ("Median",)
