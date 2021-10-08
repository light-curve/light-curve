import numpy as np

from dataclasses import dataclass
from ._base import BaseFeature


@dataclass()
class MagnitudeNNotDetBeforeFd(BaseFeature):
    """
    sigma_non_detection may not be NaN
    """

    sigma_non_detection: float = np.Inf

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        detections = np.argwhere(sigma != self.sigma_non_detection).flatten()
        first_det_idx = detections[0]
        return first_det_idx


__all__ = ("MagnitudeNNotDetBeforeFd",)
