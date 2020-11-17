import numpy as np

from dataclasses import dataclass
from ._base import BaseFeature


@dataclass()
class MedianBufferRangePercentage(BaseFeature):
    q: float = 0.1

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        median = np.median(m)
        pm = np.array([+1, -1])
        upb, db = median + pm * self.q * (np.max(m) - np.min(m)) / 2
        return np.sum((db < m) & (m < upb)) / len(m)


__all__ = ("MedianBufferRangePercentage",)
