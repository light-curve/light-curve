import numpy as np

from ._base import BaseFeature


class Amplitude(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return 0.5 * np.ptp(m)


__all__ = ("Amplitude",)
