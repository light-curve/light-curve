import numpy as np

from ._base import BaseFeature


class MaximumSlope(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        m_span = [m[i + 1] - m[i] for i in range(len(m) - 1)]
        t_span = [t[i + 1] - t[i] for i in range(len(t) - 1)]
        div = [abs(i / j) for i, j in zip(m_span, t_span)]
        return max(div)


__all__ = ("MaximumSlope",)
