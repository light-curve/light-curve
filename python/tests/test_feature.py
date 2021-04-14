import inspect

import numpy as np
import pytest

import light_curve as lc


def _feature_classes(module, exclude_parametric=True):
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.ismodule(obj):
            yield from _feature_classes(obj)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, lc._FeatureEvaluator):
            continue
        # Skip classes with non-trivial constructors
        if exclude_parametric:
            try:
                obj()
            except TypeError:
                continue
        yield obj


feature_classes = tuple(_feature_classes(lc))


@pytest.mark.parametrize("cls", feature_classes)
def test_negative_strides(cls):
    t = np.linspace(1, 0, 20)[::-2]
    m = np.exp(t)[:]
    err = np.random.uniform(0.1, 0.2, t.shape)
    obj = cls()
    obj(t, m, err)
