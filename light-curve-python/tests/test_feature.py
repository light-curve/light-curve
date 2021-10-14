import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose

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


non_param_feature_classes = frozenset(_feature_classes(lc, True))
all_feature_classes = frozenset(_feature_classes(lc, False))


@pytest.mark.parametrize("cls", non_param_feature_classes)
def test_negative_strides(cls):
    t = np.linspace(1, 0, 20)[::-2]
    m = np.exp(t)[:]
    err = np.random.uniform(0.1, 0.2, t.shape)
    obj = cls()
    obj(t, m, err)


@pytest.mark.parametrize("cls", non_param_feature_classes)
def test_float32_vs_float64(cls):
    rng = np.random.default_rng(0)
    n = 128

    obj = cls()

    t = np.sort(rng.normal(0, 1, n))
    m = t.copy()
    sigma = np.full_like(t, 0.1)

    results = [obj(t.astype(dtype), m.astype(dtype), sigma.astype(dtype), sorted=True)
               for dtype in [np.float32, np.float64]]
    assert_allclose(*results, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cls", all_feature_classes)
def test_nonempty_docstring(cls):
    assert len(cls.__doc__) > 10
