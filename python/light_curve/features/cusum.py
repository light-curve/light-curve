import numpy as np

from ._base import BaseFeature


class Cusum(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        m_mean = np.mean(m)
        m_std = np.std(m, ddof=1)
        m_new = np.cumsum(m - m_mean)
        result = m_new / (len(m) * m_std)
        return np.ptp(result)


__all__ = ("Cusum",)
