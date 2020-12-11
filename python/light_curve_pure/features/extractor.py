from dataclasses import dataclass

from ._base import BaseMetaFeature


@dataclass()
class Extractor(BaseMetaFeature):
    def transform(self, t, m, sigma):
        return t, m, sigma


__all__ = ("Extractor",)
