import numpy as np

from ._base import BaseFeature


class Amplitude(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        return 0.5 * np.ptp(m)


__all__ = ("Amplitude",)
