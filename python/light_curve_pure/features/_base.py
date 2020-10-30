from abc import ABC, abstractmethod

import numpy as np


class BaseFeature(ABC):
    @abstractmethod
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        pass
