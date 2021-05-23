import numpy as np
import pytest
from numpy.testing import assert_array_equal

from light_curve import DmDt


def random_lc(n, sigma=True, rng=None):
    rng = np.random.default_rng(rng)
    t = np.sort(rng.uniform(0, 10, n))
    m = rng.normal(0, 1, n)
    lc = (t, m)
    if sigma:
        sigma = rng.uniform(0.01, 0.1, n)
        lc += (sigma,)
    return lc


def sine_lc(n, sigma=True):
    t = np.linspace(0, 10, n)
    m = np.sin(t)
    lc = (t, m)
    if sigma:
        sigma = np.full_like(t, 0.1)
        lc += (sigma,)
    return lc


def test_dmdt_count_lgdt_three_obs():
    dmdt = DmDt(
        min_lgdt=0, max_lgdt=np.log10(3), max_abs_dm=1, lgdt_size=2, dm_size=32, norm=[]
    )

    t = np.array([0, 1, 2], dtype=np.float32)

    desired = np.array([2, 1])
    actual = dmdt.count_lgdt(t)

    assert_array_equal(actual, desired)


@pytest.mark.parametrize("lc", [sine_lc(101), random_lc(101)])
def test_dmdt_count_lgdt_many_one(lc):
    dmdt = DmDt(
        min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=[]
    )

    desired = dmdt.count_lgdt(lc[0])
    assert np.any(desired != 0)
    actual = dmdt.count_lgdt_many([lc[0]])

    assert actual.shape[0] == 1
    assert_array_equal(actual[0], desired)


@pytest.mark.parametrize(
    "lcs",
    [[sine_lc(101), sine_lc(11)], [random_lc(101), random_lc(101), random_lc(11)]],
)
def test_dmdt_count_lgdt_many(lcs):
    dmdt = DmDt(
        min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=[]
    )

    desired = [dmdt.count_lgdt(t) for t, *_ in lcs]
    actual = dmdt.count_lgdt_many([t for t, *_ in lcs])

    assert_array_equal(actual, desired)


def test_dmdt_points_three_obs():
    dmdt = DmDt(
        min_lgdt=0, max_lgdt=np.log10(3), max_abs_dm=3, lgdt_size=2, dm_size=4, norm=[]
    )

    t = np.array([0, 1, 2], dtype=np.float32)
    m = np.array([0, 1, 2], dtype=np.float32)

    desired = np.array(
        [
            [0, 0, 2, 0],
            [0, 0, 0, 1],
        ]
    )
    actual = dmdt.points(t, m)

    assert_array_equal(actual, desired)


@pytest.mark.parametrize("lc", [sine_lc(101, False), random_lc(101, False)])
@pytest.mark.parametrize("norm", [[], ["lgdt"], ["max"], ["lgdt", "max"]])
def test_dmdt_points_many_one(lc, norm):
    dmdt = DmDt(
        min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=norm
    )

    desired = dmdt.points(*lc)
    assert np.any(desired != 0)
    actual = dmdt.points_many([lc])

    assert actual.shape[0] == 1
    assert_array_equal(actual[0], desired)


@pytest.mark.parametrize(
    "lcs",
    [
        [sine_lc(101, False), sine_lc(11, False)],
        [random_lc(101, False), random_lc(101, False), random_lc(11, False)],
    ],
)
@pytest.mark.parametrize("norm", [[], ["lgdt"], ["max"], ["lgdt", "max"]])
def test_dmdt_points_many(lcs, norm):
    dmdt = DmDt(
        min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=norm
    )

    desired = [dmdt.points(*lc) for lc in lcs]
    actual = dmdt.points_many(lcs)

    assert_array_equal(actual, desired)


@pytest.mark.parametrize("lc", [sine_lc(101), random_lc(101)])
@pytest.mark.parametrize("norm", [[], ["lgdt"], ["max"], ["lgdt", "max"]])
@pytest.mark.parametrize("approx_erf", [True, False])
def test_dmdt_gausses_many_one(lc, norm, approx_erf):
    dmdt = DmDt(
        min_lgdt=-1,
        max_lgdt=1,
        max_abs_dm=2,
        lgdt_size=32,
        dm_size=32,
        norm=norm,
        approx_erf=approx_erf,
    )

    desired = dmdt.gausses(*lc)
    assert np.any(desired != 0)
    actual = dmdt.gausses_many([lc])

    assert actual.shape[0] == 1
    assert_array_equal(actual[0], desired)


@pytest.mark.parametrize(
    "lcs",
    [[sine_lc(101), sine_lc(11)], [random_lc(101), random_lc(101), random_lc(11)]],
)
@pytest.mark.parametrize("norm", [[], ["lgdt"], ["max"], ["lgdt", "max"]])
@pytest.mark.parametrize("approx_erf", [True, False])
def test_dmdt_gausses_many(lcs, norm, approx_erf):
    dmdt = DmDt(
        min_lgdt=-1,
        max_lgdt=1,
        max_abs_dm=2,
        lgdt_size=32,
        dm_size=32,
        norm=norm,
        approx_erf=approx_erf,
    )

    desired = [dmdt.gausses(*lc) for lc in lcs]
    actual = dmdt.gausses_many(lcs)

    assert_array_equal(actual, desired)
