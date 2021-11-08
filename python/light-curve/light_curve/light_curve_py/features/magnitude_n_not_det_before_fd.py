import numpy as np

from dataclasses import dataclass
from ._base import BaseFeature


@dataclass()
class MagnitudeNNotDetBeforeFd(BaseFeature):
    """Number of non detections before the first detection for measurements of the magnitude.

    Feature use a user-defined value to mark non-detections: measurements with sigma equal to this value
    considered as non detections. Strictly_fainter flag allows counting non-detections with a strictly smaller
    upper limit than the first detection magnitude (there is no such feature in the original article).

    - Depends on: **magnitude**
    - Minimum number of observations: **2**
    - Number of features: **1**

    P. Sánchez-Sáez et al 2021, [DOI:10.3847/1538-3881/abd5c1](https://doi.org/10.3847/1538-3881/abd5c1)
    """

    sigma_non_detection: float = np.Inf
    """Sigma value to mark the non detections values, may not be NaN.
    """

    strictly_fainter: bool = False
    """Flag to determine if to find non-detections with strictly smaller upper limit than the first detection magnitude.
    """

    def _eval(self, t, m, sigma=None, sorted=None, fill_value=None):
        detections = np.argwhere(sigma != self.sigma_non_detection).flatten()
        first_detection_idx = detections[0]

        if self.strictly_fainter:
            detection_m = m[first_detection_idx]
            non_detection_less = np.count_nonzero(m[:first_detection_idx] < detection_m)
            return non_detection_less

        return first_detection_idx

    @property
    def names(self):
        return "magn_n_non_detections_before_fd"

    @property
    def descriptions(self):
        return "number of non detections before the first detection for magnitudes"

    @property
    def size(self):
        return 1


__all__ = ("MagnitudeNNotDetBeforeFd",)
