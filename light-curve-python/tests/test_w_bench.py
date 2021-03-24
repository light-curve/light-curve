import feets
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import curve_fit

import light_curve as lc


class _FeatureTest:
    # Default values of `assert_allclose`
    rtol = 1e-7
    atol = 0

    n_obs = 1000
    t_min = 0.0
    t_max = 1000.0
    m_min = 15.0
    m_max = 21.0
    sigma_min = 0.01
    sigma_max = 0.2

    def generate_data(self):
        t = np.sort(np.random.uniform(self.t_min, self.t_max, self.n_obs))
        m = np.random.uniform(self.m_min, self.m_max, self.n_obs)
        sigma = np.random.uniform(self.sigma_min, self.sigma_max, self.n_obs)
        return t, m, sigma

    def test_feature_length(self):
        t, m, sigma = self.generate_data()
        result = self.feature(t, m, sigma, sorted=None)
        assert len(result) == len(self.feature.names) == len(self.feature.descriptions)

    def test_benchmark_feature(self, benchmark):
        t, m, sigma = self.generate_data()

        benchmark.group = str(type(self).__name__)
        benchmark(self.feature, t, m, sigma, sorted=True)


class _NaiveTest:
    naive = None

    def test_close_to_naive(self):
        t, m, sigma = self.generate_data()
        assert_allclose(self.feature(t, m, sigma), self.naive(t, m, sigma), rtol=self.rtol, atol=self.atol)

    def test_benchmark_naive(self, benchmark):
        t, m, sigma = self.generate_data()

        benchmark.group = type(self).__name__
        benchmark(self.naive, t, m, sigma)


class _FeetsTest:
    feets_feature = None
    feets_skip_test = False

    def setup_method(self):
        self.feets_extractor = feets.FeatureSpace(only=[self.feets_feature], data=["time", "magnitude", "error"])

    def feets(self, t, m, sigma):
        _, result = self.feets_extractor.extract(t, m, sigma)
        return result

    def test_close_to_feets(self):
        if self.feets_skip_test:
            pytest.skip("feets is expected to be different from light_curve, reason: " + self.feets_skip_test)
        t, m, sigma = self.generate_data()
        assert_allclose(self.feature(t, m, sigma)[:1], self.feets(t, m, sigma)[:1], rtol=self.rtol, atol=self.atol)

    def test_benchmark_feets(self, benchmark):
        t, m, sigma = self.generate_data()

        benchmark.group = type(self).__name__
        benchmark(self.feets, t, m, sigma)


class TestAmplitude(_FeatureTest, _NaiveTest):
    feature = lc.Amplitude()

    def naive(self, t, m, sigma):
        return 0.5 * (np.max(m) - np.min(m))


class TestAndersonDarlingNormal(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.AndersonDarlingNormal()

    feets_feature = "AndersonDarling"
    feets_skip_test = "feets uses biased statistics"

    def naive(self, t, m, sigma):
        return stats.anderson(m).statistic * (1.0 + 4.0 / m.size - 25.0 / m.size ** 2)


if hasattr(lc, 'BazinFit'):
    class TestBazinFit(_FeatureTest, _NaiveTest):
        feature = lc.BazinFit()
        rtol = 1e-4  # Precision used in the feature implementation

        @staticmethod
        def _model(t, a, b, t0, rise, fall):
            dt = t - t0
            return b + a * np.exp(-dt / fall) / (1.0 + np.exp(-dt / rise))

        def _params(self):
            a = 1000
            b = 0
            t0 = 0.5 * (self.t_min + self.t_max)
            rise = 0.3 * (self.t_max - self.t_min)
            fall = 0.4 * (self.t_max - self.t_min)
            return a, b, t0, rise, fall

        # Random data yields to random results because target function has a lot of local minima
        # BTW, this test shouldn't use fixed random seed because the curve has good enough S/N to be fitted for any give
        # noise sample
        def generate_data(self):
            rng = np.random.default_rng(0)
            t = np.linspace(self.t_min, self.t_max, self.n_obs)
            sigma = np.ones_like(t)
            m = self._model(t, *self._params()) + sigma * rng.normal(size=self.n_obs)
            return t, m, sigma

        def naive(self, t, m, sigma):
            params, _cov = curve_fit(
                self._model,
                xdata=t,
                ydata=m,
                sigma=sigma,
                xtol=self.rtol,
                # We give really good parameters estimation!
                p0=self._params(),
            )
            reduced_chi2 = np.sum(np.square((self._model(t, *params) - m) / sigma)) / (t.size - params.size)
            return_value = tuple(params) + (reduced_chi2,)
            return return_value


class TestBeyond1Std(_FeatureTest, _NaiveTest, _FeetsTest):
    nstd = 1.0

    feature = lc.BeyondNStd(nstd)

    feets_feature = "Beyond1Std"
    feets_skip_test = "feets uses biased statistics"

    def naive(self, t, m, sigma):
        mean = np.mean(m)
        interval = self.nstd * np.std(m, ddof=1)
        return np.count_nonzero(np.abs(m - mean) > interval) / m.size


class TestCusum(_FeatureTest, _FeetsTest):
    feature = lc.Cusum()

    feets_feature = "Rcs"
    feets_skip_test = "feets uses biased statistics"


class TestEta(_FeatureTest, _NaiveTest):
    feature = lc.Eta()

    def naive(self, t, m, sigma):
        return np.sum(np.square(m[1:] - m[:-1])) / (np.var(m, ddof=0) * m.size)


class TestEtaE(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.EtaE()

    feets_feature = "Eta_e"
    feets_skip_test = "feets fixed EtaE from the original paper in different way"

    def naive(self, t, m, sigma):
        return (
            np.sum(np.square((m[1:] - m[:-1]) / (t[1:] - t[:-1])))
            * (t[-1] - t[0]) ** 2
            / (np.var(m, ddof=0) * m.size * (m.size - 1) ** 2)
        )


class TestExcessVariance(_FeatureTest, _NaiveTest):
    feature = lc.ExcessVariance()

    def naive(self, t, m, sigma):
        return (np.var(m, ddof=1) - np.mean(sigma ** 2)) / np.mean(m) ** 2


class TestInterPercentileRange(_FeatureTest, _FeetsTest):
    quantile = 0.25

    feature = lc.InterPercentileRange(quantile)

    feets_feature = "Q31"
    feets_skip_test = "feets uses different quantile type"


class TestKurtosis(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.Kurtosis()

    feets_feature = "SmallKurtosis"
    feets_skip_test = "feets uses equation for unbiased kurtosis, but put biased standard deviation there"

    def naive(self, t, m, sigma):
        return stats.kurtosis(m, fisher=True, bias=False)


class TestLinearTrend(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.LinearTrend()

    feets_feature = "LinearTrend"

    def naive(self, t, m, sigma):
        (slope, _), ((slope_sigma2, _), _) = np.polyfit(t, m, deg=1, cov=True)
        return np.array([slope, np.sqrt(slope_sigma2)])


class _TestMagnitudePercentageRatio(_FeatureTest, _FeetsTest):
    quantile_numerator = -1
    quantile_denumerator = -1

    feets_skip_test = "feets uses different quantile type"

    def setup_method(self):
        super().setup_method()
        self.feature = lc.MagnitudePercentageRatio(self.quantile_numerator, self.quantile_denumerator)


class TestMagnitudePercentageRatio40(_TestMagnitudePercentageRatio):
    quantile_numerator = 0.4
    quantile_denumerator = 0.05

    feets_feature = "FluxPercentileRatioMid20"


class TestMagnitudePercentageRatio25(_TestMagnitudePercentageRatio):
    quantile_numerator = 0.25
    quantile_denumerator = 0.05

    feets_feature = "FluxPercentileRatioMid50"


class TestMagnitudePercentageRatio10(_TestMagnitudePercentageRatio):
    quantile_numerator = 0.10
    quantile_denumerator = 0.05

    feets_feature = "FluxPercentileRatioMid80"


class TestMaximumSlope(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.MaximumSlope()

    feets_feature = "MaxSlope"

    def naive(self, t, m, sigma):
        return np.max(np.abs((m[1:] - m[:-1]) / (t[1:] - t[:-1])))


class TestMean(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.Mean()

    feets_feature = "Mean"

    def naive(self, t, m, sigma):
        return np.mean(m)


class TestMeanVariance(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.MeanVariance()

    feets_feature = "Meanvariance"
    feets_skip_test = "feets uses biased statistics"

    def naive(self, t, m, sigma):
        return np.std(m, ddof=1) / np.mean(m)


class TestMedian(_FeatureTest, _NaiveTest):
    feature = lc.Median()

    def naive(self, t, m, sigma):
        return np.median(m)


class TestMedianAbsoluteDeviation(_FeatureTest, _FeetsTest):
    feature = lc.MedianAbsoluteDeviation()

    feets_feature = "MedianAbsDev"


class TestMedianBufferRangePercentage(_FeatureTest, _FeetsTest):
    # feets says it uses 0.1 of amplitude (a half range between max and min),
    # but factually it uses 0.1 of full range between max and min
    quantile = 0.2

    feature = lc.MedianBufferRangePercentage(quantile)

    feets_feature = "MedianBRP"


class TestPercentAmplitude(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.PercentAmplitude()

    feets_feature = "PercentAmplitude"
    feets_skip_test = "feets divides value by median"

    def naive(self, t, m, sigma):
        median = np.median(m)
        return max(np.max(m) - median, median - np.min(m))


class TestPercentDifferenceMagnitudePercentile(_FeatureTest, _FeetsTest):
    quantile = 0.05

    feature = lc.PercentDifferenceMagnitudePercentile(quantile)

    feets_feature = "PercentDifferenceFluxPercentile"
    feets_skip_test = "feets uses different quantile type"


class TestReducedChi2(_FeatureTest, _NaiveTest):
    feature = lc.ReducedChi2()

    def naive(self, t, m, sigma):
        w = 1.0 / np.square(sigma)
        return np.sum(np.square(m - np.average(m, weights=w)) * w) / (m.size - 1)


class TestSkew(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.Skew()

    feets_feature = "Skew"
    feets_skip_test = "feets uses biased statistics"

    def naive(self, t, m, sigma):
        return stats.skew(m, bias=False)


class TestStandardDeviation(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.StandardDeviation()

    feets_feature = "Std"
    feets_skip_test = "feets uses biased statistics"

    def naive(self, t, m, sigma):
        return np.std(m, ddof=1)


class TestStetsonK(_FeatureTest, _NaiveTest, _FeetsTest):
    feature = lc.StetsonK()

    feets_feature = "StetsonK"

    def naive(self, t, m, sigma):
        x = (m - np.average(m, weights=1.0 / sigma ** 2)) / sigma
        return np.sum(np.abs(x)) / np.sqrt(np.sum(np.square(x)) * m.size)


class TestWeightedMean(_FeatureTest, _NaiveTest):
    feature = lc.WeightedMean()

    def naive(self, t, m, sigma):
        return np.average(m, weights=1.0 / sigma ** 2)


class TestAllNaive(_FeatureTest, _NaiveTest):
    def setup_method(self):
        features = []
        self.naive_features = []
        for cls in _NaiveTest.__subclasses__():
            if cls.naive is None or not hasattr(cls, "feature"):
                continue
            features.append(cls.feature)
            self.naive_features.append(cls().naive)
        self.feature = lc.Extractor(*features)

    def naive(self, t, m, sigma):
        return np.concatenate([np.atleast_1d(f(t, m, sigma)) for f in self.naive_features])


class TestAllFeets(_FeatureTest, _FeetsTest):
    feets_skip_test = "skip for TestAllFeets"

    def setup_method(self):
        features = []
        feets_features = []
        for cls in _FeetsTest.__subclasses__():
            if cls.feets_feature is None or not hasattr(cls, "feature"):
                continue
            features.append(cls.feature)
            feets_features.append(cls.feets_feature)
        self.feature = lc.Extractor(*features)
        self.feets_extractor = feets.FeatureSpace(only=feets_features, data=["time", "magnitude", "error"])
