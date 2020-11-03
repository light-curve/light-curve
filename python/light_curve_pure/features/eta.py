import numpy as np

from ._base import BaseFeature


class Eta(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        n = len(m)
        m_std = np.std(m, ddof=1) ** 2
        m_sum = sum([(m[i + 1] - m[i]) ** 2 for i in range(n - 1)])
        return m_sum / ((n - 1) * m_std)


__all__ = ("Eta",)
