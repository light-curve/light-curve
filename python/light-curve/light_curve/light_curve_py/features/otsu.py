from dataclasses import dataclass
import numpy as np

from ._base import BaseFeature


@dataclass()
class Otsu(BaseFeature):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        n = len(m)
        amounts = np.arange(1, n)

        w0 = amounts / n
        w1 = 1 - w0

        cumsum0 = np.cumsum(m)[:-1]
        cumsum1 = np.cumsum(m[::-1])[:-1][::-1]
        mean0 = cumsum0 / amounts
        mean1 = cumsum1 / amounts[::-1]

        inter_class_variance = w0 * w1 * (mean0 - mean1) ** 2
        arg = np.argmax(inter_class_variance)
        threshold = m[arg]

        return mean0[arg] - mean1[arg]


__all__ = ("Otsu",)
