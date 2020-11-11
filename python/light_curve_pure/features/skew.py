import numpy as np

from ._base import BaseFeature
from scipy.stats import skew


class Skew(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        n = len(m)
        return skew(m, bias=False)


__all__ = ("Skew",)
