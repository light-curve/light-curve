from dataclasses import dataclass
from typing import Collection, Union

import numpy as np

from ._base import BaseFeature
from light_curve.light_curve_ext import Extractor as _RustExtractor, _FeatureEvaluator as _RustBaseFeature


@dataclass()
class _PyExtractor(BaseFeature):
    features: Collection[Union[BaseFeature, _RustBaseFeature]] = ()

    def _eval(self, t, m, sigma=None):
        return np.concatenate([np.atleast_1d(feature._eval(t, m, sigma)) for feature in self.features])


class Extractor:
    def __new__(cls, features: Collection[Union[BaseFeature, _RustBaseFeature]]):
        if len(features) > 0 and all(isinstance(feature, _RustBaseFeature) for feature in features):
            return _RustExtractor(features)
        else:
            return _PyExtractor(features)


__all__ = ("Extractor",)
