import numpy as np

from ._base import BaseFeature


class Mean(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return np.mean(m)


__all__ = ("Mean",)
