import numpy as np

from ._base import BaseFeature



class Skew(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        n = len(m)
        m_mean = np.mean(m)
        m_sum = sum(np.power(m - m_mean, 3)) / n
        m_std = np.power(np.std(m), 3)
        return (np.sqrt(n * (n - 1)) * m_sum) / ((n - 2) * m_std)


__all__ = ("Skew",)
