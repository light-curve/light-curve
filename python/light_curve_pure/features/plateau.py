from dataclasses import dataclass

import numpy as np
from scipy.stats import sigmaclip

from ._base import BaseFeature


@dataclass()
class Plateau(BaseFeature):
    n_std: float = 3

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        plateau_size = 0
        while True:
            m, lower, upper = sigmaclip(m, low=self.n_std, high=self.n_std)
            print(m.size, plateau_size)
            if m.size == plateau_size:
                break
            plateau_size = m.size
        return plateau_size / t.size


__all__ = ("Plateau",)
