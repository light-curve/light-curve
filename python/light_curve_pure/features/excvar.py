import numpy as np

from ._base import BaseFeature


class ExcessVariance:
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        m_mean = np.mean(m)
        d_mean = np.mean(np.power(sigma, 2))
        m_std = np.std(m, ddof=1)
        return (m_std ** 2 - d_mean) / m_mean ** 2


__all__ = ("ExcessVariance",)
