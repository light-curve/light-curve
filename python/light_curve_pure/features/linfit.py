import numpy as np

from ._base import BaseFeature


class LinearFit(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        n = len(t)
        w = np.diag(1 / sigma)
        A = np.vstack([t, np.ones(len(t))])

        A_weighted = np.dot(A, w).T
        m_weighted = np.dot(m, w)

        solution, residuals, rank, s = np.linalg.lstsq(A_weighted, m_weighted, rcond=None)
        slope, intercept = solution
        residuals = np.float(residuals)
        chisq = residuals / (n - 2)
        sxx = np.var(t, ddof=n - 1)

        return slope, np.sqrt(chisq / sxx), chisq


__all__ = ("LinearFit",)
