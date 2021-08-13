import numpy as np

from ._base import BaseFeature


class StetsonJ(BaseFeature):
    def _eval(self, t, m, sigma=None):
        n = len(m)
        mean = np.average(m, weights=np.power(sigma, -2))
        delta = ((m - mean) / np.power(sigma, -2)) * np.sqrt(n / (n - 1))
        product = delta[:-1] * delta[1:]
        signs = np.sign(product)
        return np.sum(signs * np.sqrt(np.abs(product)))


__all__ = ("StetsonJ",)
