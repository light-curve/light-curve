import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

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


def test_dmdt_count_dt_three_obs():
    dmdt = DmDt.from_borders(min_lgdt=0, max_lgdt=np.log10(3), max_abs_dm=1, lgdt_size=2, dm_size=32, norm=[])

    t = np.array([0, 1, 2], dtype=np.float32)

    desired = np.array([2, 1])
    actual = dmdt.count_dt(t)

    assert_array_equal(actual, desired)

def test_log_linear_grids():
    lc = random_lc(101)

    min_lgdt = -1
    max_lgdt = 1
    lgdt_size = 32
    max_abs_dm = 2
    dm_size = 32

    min_dt = 10 ** min_lgdt
    max_dt = 10 ** max_lgdt

    dt_grid = np.logspace(min_lgdt, max_lgdt, lgdt_size + 1)
    dm_grid = np.linspace(-max_abs_dm, max_abs_dm, dm_size + 1)

    dmdt_from_borders = DmDt.from_borders(min_lgdt=min_lgdt, max_lgdt=max_lgdt, max_abs_dm=max_abs_dm,
                                          lgdt_size=lgdt_size, dm_size=dm_size)
    dmdt_auto = DmDt(dt=dt_grid, dm=dm_grid, dt_type='auto', dm_type='auto')
    dmdt_log_linear = DmDt(dt=dt_grid, dm=dm_grid, dt_type='log', dm_type='linear')
    dmdt_asis = DmDt(dt=dt_grid, dm=dm_grid, dt_type='asis', dm_type='asis')

    for dmdt in (dmdt_from_borders, dmdt_auto, dmdt_log_linear, dmdt_asis,):
        assert_allclose(dmdt.min_dt, min_dt)
        assert_allclose(dmdt.max_dt, max_dt)
        assert_allclose(dmdt.min_dm, -max_abs_dm)
        assert_allclose(dmdt.max_dm, max_abs_dm)
        assert_allclose(dmdt.dt_grid, dt_grid)
        assert_allclose(dmdt.dm_grid, dm_grid)

    points = dmdt_from_borders.points(lc[0], lc[1])
    gausses = dmdt_from_borders.gausses(*lc)
    for dmdt in (dmdt_auto, dmdt_log_linear, dmdt_asis,):
        assert_allclose(dmdt.points(lc[0], lc[1]), points)
        assert_allclose(dmdt.gausses(*lc), gausses)



@pytest.mark.parametrize("lc", [sine_lc(101), random_lc(101)])
def test_dmdt_count_dt_many_one(lc):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=[])

    desired = dmdt.count_dt(lc[0])
    assert np.any(desired != 0)
    actual = dmdt.count_dt_many([lc[0]])

    assert actual.shape[0] == 1
    assert_array_equal(actual[0], desired)


@pytest.mark.parametrize(
    "lcs",
    [[sine_lc(101), sine_lc(11)], [random_lc(101), random_lc(101), random_lc(11)]],
)
def test_dmdt_count_dt_many(lcs):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=[])

    desired = [dmdt.count_dt(t) for t, *_ in lcs]
    actual = dmdt.count_dt_many([t for t, *_ in lcs])

    assert_array_equal(actual, desired)


def test_dmdt_points_three_obs():
    dmdt = DmDt.from_borders(min_lgdt=0, max_lgdt=np.log10(3), max_abs_dm=3, lgdt_size=2, dm_size=4, norm=[])

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
@pytest.mark.parametrize("norm", [[], ["dt"], ["max"], ["dt", "max"]])
def test_dmdt_points_many_one(lc, norm):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=norm)

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
@pytest.mark.parametrize("norm", [[], ["dt"], ["max"], ["dt", "max"]])
def test_dmdt_points_many(lcs, norm):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=norm)

    desired = [dmdt.points(*lc) for lc in lcs]
    actual = dmdt.points_many(lcs)

    assert_array_equal(actual, desired)


@pytest.mark.parametrize("lc", [sine_lc(101), random_lc(101)])
@pytest.mark.parametrize("norm", [[], ["dt"], ["max"], ["dt", "max"]])
@pytest.mark.parametrize("approx_erf", [True, False])
def test_dmdt_gausses_many_one(lc, norm, approx_erf):
    dmdt = DmDt.from_borders(
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
@pytest.mark.parametrize("norm", [[], ["dt"], ["max"], ["dt", "max"]])
@pytest.mark.parametrize("approx_erf", [True, False])
def test_dmdt_gausses_many(lcs, norm, approx_erf):
    dmdt = DmDt.from_borders(
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
