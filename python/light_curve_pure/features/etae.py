import numpy as np

from ._base import BaseFeature


class EtaE(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        n = len(m)
        m_std = np.var(m, ddof=1)
        m_sum = np.sum(((np.subtract(m[1:], m[:-1]) / np.subtract(t[1:], t[:-1])) ** 2))
        return m_sum * (t[n - 1] - t[0]) ** 2 / ((n - 1) ** 3 * m_std)


__all__ = ("EtaE",)
