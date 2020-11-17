import numpy as np

from ._base import BaseFeature
from scipy.stats import anderson


class AndersonDarlingNormal(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        n = len(m)
        return anderson(m).statistic * (1 + 4 / n - 25 / n ** 2)
    

__all__ = ("AndersonDarlingNormal",)

