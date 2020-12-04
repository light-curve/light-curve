import numpy as np

from ._base import BaseFeature


class LinearTrend(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        n = len(t)

        if n == 2:
            return (m[1] - m[0]) / (t[1] - t[0]), 0, 0

        A = np.vstack([t, np.ones(len(t))]).T
        solution, residuals, rank, s = np.linalg.lstsq(A, m, rcond=None)
        slope, intercept = solution
        residuals = np.float(residuals)
        chisq = residuals / (n - 2)
        sxx = np.var(t, ddof=n - 1)
        return slope, np.sqrt(chisq / sxx), chisq


__all__ = ("LinearTrend",)
