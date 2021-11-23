import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

from light_curve.light_curve_py.warnings import warn_experimental


@dataclass
class BaseFeature(ABC):
    @staticmethod
    def _normalize_input(t, m, sigma, sorted):
        t = np.asarray(t)
        m = np.asarray(m)
        if sigma is not None:
            sigma = np.asarray(sigma)
        if sorted is None:
            diff = np.diff(t)
            if np.any(diff == 0):
                raise ValueError("t must be unique")
            if np.any(diff < 0):
                raise ValueError("t must be sorted")
        elif not sorted:
            idx = np.argsort(t)
            t = t[idx]
            m = m[idx]
            if sigma is not None:
                sigma = sigma[idx]

        return t, m, sigma

    def _eval_and_fill(self, t, m, sigma, fill_value):
        try:
            a = self._eval(t, m, sigma)
            if np.any(np.isnan(a)):
                raise ValueError
            return a
        except (ValueError, ZeroDivisionError) as e:
            if fill_value is not None:
                return np.full(self.size, fill_value)
            raise e

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        t, m, sigma = self._normalize_input(t, m, sigma, sorted)
        return self._eval_and_fill(t, m, sigma, fill_value)

    def __post_init__(self):
        warn_experimental(
            "Feature {} is experimental and not supported by meta-features implemented in Rust".format(
                type(self).__name__
            )
        )

    def many(self, lcs, sorted=None, fill_value=None, n_jobs=-1):
        """Extract features in bulk

        This exists for computability only and doesn't support parallel
        execution, that's why `n_jobs=1` must be used
        """
        if n_jobs != 1:
            raise NotImplementedError("Parallel execution is not supported by this feature, use n_jobs=1")
        return np.stack([self(*lc, sorted=sorted, fill_value=fill_value) for lc in lcs])

    @property
    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def _eval(self, t, m, sigma=None):
        pass
