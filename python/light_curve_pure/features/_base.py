from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection

import numpy as np


class BaseFeature(ABC):
    @abstractmethod
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        pass


@dataclass
class BaseMetaFeature(BaseFeature):
    features: Collection[BaseFeature]

    @abstractmethod
    def transform(self, t, m, sigma):
        pass

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        t, m, sigma = self.transform(t, m, sigma)
        result = np.concatenate([np.atleast_1d(feature(t, m, sigma)) for feature in self.features])
        return result
