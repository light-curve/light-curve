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

    @abstarctmethod
    def transform(self, t, m, sigma):
        pass

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        t, m, sigma = self.transform(t, m, sigma)
        # Make it more numpy, remember about Features returning several values
        result = []
        for feature in features:
            result.append(feature(t, m, sigma))
        return result


# Move to separate sub-module
class Extractor(BaseMetaFeature):
    features = (Mean(), Amplitude())

    def transform(self, t, m, sigma):
        return t, m, sigma
