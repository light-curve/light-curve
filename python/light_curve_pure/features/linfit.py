import numpy as np

from ._base import BaseFeature
from ..lstsq import least_squares


class LinearFit(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        n = len(t)

        slope, chi2 = least_squares(t, m, sigma)

        red_chi2 = chi2 / (n - 2)

        weighted_t2 = np.average(t ** 2, weights=np.power(sigma, -2))
        weighted_t = np.average(t, weights=np.power(sigma, -2)) ** 2

        sigma_sum = np.sum(1 / sigma ** 2)
        delta = sigma_sum ** 2 * (weighted_t2 - weighted_t)

        return slope, np.sqrt(sigma_sum / delta), red_chi2


__all__ = ("LinearFit",)
