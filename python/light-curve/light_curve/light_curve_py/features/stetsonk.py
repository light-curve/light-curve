import numpy as np

from ._base import BaseFeature


class StetsonK(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        m_mean = np.average(m, weights=np.power(sigma, -2))
        numerator = np.sum(np.abs((m - m_mean) / sigma))
        chisq = np.sum(((m - m_mean) / sigma) ** 2)
        return numerator / np.sqrt(len(m) * chisq)


__all__ = ("StetsonK",)
