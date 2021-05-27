from dataclasses import dataclass
import numpy as np

from ._base import BaseFeature


@dataclass()
class Otsu(BaseFeature):
    bins_number: int = 256  # optimal bins num?

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        hist, edges = np.histogram(m, bins=self.bins_number)
        n = np.size(m)

        edges_mid = (edges[:-1] + edges[1:]) / 2
        cumsum = np.cumsum(hist)
        cumsum2 = np.cumsum(hist[::-1])
        heights = edges_mid * hist  # rename

        weight1 = cumsum / n
        weight2 = 1 - weight1

        mean1 = np.cumsum(heights) / cumsum
        mean2 = (np.cumsum(heights[::-1]) / cumsum2)[::-1]

        inter_class_variance = weight1 * weight2 * (mean1 - mean2) ** 2
        arg = np.argmax(inter_class_variance)
        threshold = edges_mid[arg]

        idx = np.argwhere(m < edges_mid[arg])
        idx2 = np.argwhere(m > edges_mid[arg])

        return threshold, (t[idx], m[idx], sigma[idx]), (t[idx2], m[idx2], sigma[idx2])


__all__ = ("Otsu",)
