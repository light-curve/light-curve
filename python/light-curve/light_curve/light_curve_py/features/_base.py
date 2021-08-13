from abc import ABC, abstractmethod


class BaseFeature(ABC):
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        return self._eval(t, m, sigma)

    @abstractmethod
    def _eval(self, t, m, sigma=None):
        pass
