import numpy as np

from ._base import BaseFeature
from scipy.stats import skew


class Skew(BaseFeature):
    def _eval(self, t, m, sigma=None):
        return skew(m, bias=False)


__all__ = ("Skew",)
