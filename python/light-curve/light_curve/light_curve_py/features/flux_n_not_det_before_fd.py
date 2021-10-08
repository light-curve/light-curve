import numpy as np
from dataclasses import dataclass
from ._base import BaseFeature


@dataclass()
class FluxNNotDetBeforeFd(BaseFeature):
    signal_to_noise: float

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        detections = np.argwhere(m > self.signal_to_noise * sigma).flatten()
        first_detection_idx = detections[0]
        return first_detection_idx


__all__ = ("FluxNNotDetBeforeFd",)
