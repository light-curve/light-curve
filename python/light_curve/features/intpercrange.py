from dataclasses import dataclass
from ._base import BaseFeature

from scipy.stats.mstats import mquantiles


@dataclass()
class InterPercentileRange(BaseFeature):
    p: float = 0.25

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        q1, q2 = mquantiles(m, [self.p, 1 - self.p], alphap=0.5, betap=0.5)
        return q2 - q1


__all__ = ("InterPercentileRange",)
