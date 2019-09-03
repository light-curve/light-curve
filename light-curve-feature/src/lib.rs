//! # Light curve feature
//!
//! `light-curve-feature` is a part of [`light-curve`](https://docs.rs/light-curve) family that
//! implements extraction of numerous light curve features used in astrophysics.
//!
//! ```
//! use light_curve_feature::*;
//!
//! // Let's find amplitude and reduced Chi-squared of the light curve
//! let fe = feat_extr!(Amplitude::default(), ReducedChi2::default());
//! // Define light curve
//! let time = [0.0, 1.0, 2.0, 3.0, 4.0];
//! let magn = [-1.0, 2.0, 1.0, 3.0, 4.5];
//! let magn_err_squared = [0.2, 0.1, 0.5, 0.1, 0.2];
//! let ts = TimeSeries::new(&time[..], &magn[..], Some(&magn_err_squared[..]));
//! // Get results and print
//! let result = fe.eval(ts);
//! let names = fe.get_names();
//! println!("{:?}", names.iter().zip(result.iter()).collect::<Vec<_>>());
//! ```
use conv::prelude::*;

mod fit;
use fit::fit_straight_line;

mod float_trait;
use float_trait::Float;

pub mod statistics;
use statistics::Statistics;

mod periodogram;
use periodogram::{PeriodogramFreq, PeriodogramFreqFactors};

pub mod time_series;
pub use time_series::TimeSeries;

/// Constructs a `FeatureExtractor` object from a list of objects that implement `FeatureEvaluator`
/// ```
/// use light_curve_feature::*;
///
/// let fe = feat_extr!(BeyondNStd::new(1.0), Cusum::default());
/// ```
#[macro_export]
macro_rules! feat_extr{
    ( $( $x: expr ),* $(,)? ) => {
        FeatureExtractor::new(
            vec![$(
                Box::new($x),
            )*]
        )
    }
}

/// The engine that extract features one by one
///
/// Generic parameter `T` should be a float type
pub struct FeatureExtractor<T> {
    features: VecFE<T>,
}

impl<T> FeatureExtractor<T>
where
    T: Float,
{
    pub fn new(features: VecFE<T>) -> Self {
        Self { features }
    }

    /// Get a vector of computed features.
    /// The length of the returned vector is guaranteed to be the same as returned by `get_names()`
    pub fn eval(&self, mut ts: TimeSeries<T>) -> Vec<T> {
        self.features.iter().flat_map(|x| x.eval(&mut ts)).collect()
    }

    /// Get a vector of feature names.
    /// The length of the returned vector is guaranteed to be the same as returned by `eval()`
    pub fn get_names(&self) -> Vec<&str> {
        self.features.iter().flat_map(|x| x.get_names()).collect()
    }
}

/// The trait each feature should implement
pub trait FeatureEvaluator<T: Float>: Send + Sync {
    /// Should return the non-empty vector of feature values. The length and feature order should
    /// correspond to `get_names()` output
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T>;
    /// Should return the non-empty vector of feature names. The length and feature order should
    /// correspond to `eval()` output
    fn get_names(&self) -> Vec<&str>;
}

pub type VecFE<T> = Vec<Box<dyn FeatureEvaluator<T>>>;

/// Half amplitude of magnitude
///
/// $$
/// \mathrm{amplitude} \equiv \frac{\left( \max{(m)} - \min{(m)} \right)}{2}
/// $$
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
/// ```
/// use light_curve_feature::*;
///
/// let fe = feat_extr!(Amplitude::default());
/// let time = [0.0, 1.0];  // Doesn't depend on time
/// let magn = [0.0, 2.0];
/// let ts = TimeSeries::new(&time[..], &magn[..], None);
/// assert_eq!(vec![1.0], fe.eval(ts));
/// ```
#[derive(Default)]
pub struct Amplitude {}

impl Amplitude {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for Amplitude
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        vec![T::half() * (ts.m.get_max() - ts.m.get_min())]
    }
    fn get_names(&self) -> Vec<&str> {
        vec!["amplitude"]
    }
}

/// Fraction of observations beyond $n\\,\sigma\_m$ from the mean magnitude $\langle m \rangle$
///
/// $$
/// \mathrm{beyond}~n\\,\sigma\_m \equiv \frac{\sum\_i I\_{|m - \langle m \rangle| > n\\,\sigma\_m}(m_i)}{N},
/// $$
/// where $I$ is the [indicator function](https://en.wikipedia.org/wiki/Indicator_function),
/// $N$ is the number of observations,
/// $\langle m \rangle$ is the mean magnitude
/// and $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
/// ```
/// use light_curve_feature::*;
/// use light_curve_common::all_close;
/// use std::f64::consts::SQRT_2;
///
/// let fe = feat_extr!(BeyondNStd::default(), BeyondNStd::new(2.0));
/// let time = [0.0; 21];  // Doesn't depend on time
/// let mut magn = vec![0.0; 17];
/// magn.extend_from_slice(&[SQRT_2, -SQRT_2, 2.0 * SQRT_2, -2.0 * SQRT_2]);
/// let mut ts = TimeSeries::new(&time[..], &magn[..], None);
/// assert_eq!(0.0, ts.m.get_mean());
/// assert!((1.0 - ts.m.get_std()).abs() < 1e-15);
/// assert_eq!(vec![4.0 / 21.0, 2.0 / 21.0], fe.eval(ts));
/// ```
pub struct BeyondNStd<T> {
    nstd: T,
    name: String,
}

impl<T> BeyondNStd<T>
where
    T: Float,
{
    pub fn new(nstd: T) -> Self {
        assert!(nstd > T::zero(), "nstd should be positive");
        Self {
            nstd,
            name: format!("beyond_{:.0}_std", nstd),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

impl<T> Default for BeyondNStd<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(T::one())
    }
}

impl<T> FeatureEvaluator<T> for BeyondNStd<T>
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let m_mean = ts.m.get_mean();
        let threshold = ts.m.get_std() * self.nstd;
        vec![
            ts.m.sample
                .iter()
                .cloned()
                .filter(|&y| T::abs(y - m_mean) > threshold)
                .count()
                .value_as::<T>()
                .unwrap()
                / ts.lenf(),
        ]
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }
}

/// Cusum — a range of cumulative sums
///
/// $$
/// \mathrm{cusum} \equiv \max(S) - \min(S),
/// $$
/// where
/// $$
/// S_j \equiv \frac1{N\\,\sigma_m} \sum_{i=0}^j{\left(m\_i - \langle m \rangle\right)},
/// $$
/// $N$ is the number of observations,
/// $\langle m \rangle$ is the mean magnitude
/// and $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// Kim et al. 2014, [DOI:10.1051/0004-6361/201323252](https://doi.org/10.1051/0004-6361/201323252)
#[derive(Default)]
pub struct Cusum {}

impl Cusum {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for Cusum
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let m_mean = ts.m.get_mean();
        let cumsum: Vec<_> =
            ts.m.sample
                .iter()
                .scan(T::zero(), |sum, &y| {
                    *sum += y - m_mean;
                    Some(*sum)
                })
                .collect();
        vec![(cumsum[..].maximum() - cumsum[..].minimum()) / (ts.m.get_std() * ts.lenf())]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["cusum"]
    }
}

/// Von Neummann $\eta$
///
/// $$
/// \eta \equiv \frac1{(N - 1)\\,\sigma_m^2} \sum_{i=0}^{N-2}(m_{i+1} - m_i)^2,
/// $$
/// where $N$ is the number of observations,
/// $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// Kim et al. 2014, [DOI:10.1051/0004-6361/201323252](https://doi.org/10.1051/0004-6361/201323252)
#[derive(Default)]
pub struct Eta {}

impl Eta {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for Eta
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        vec![
            (1..ts.lenu())
                .map(|i| (ts.m.sample[i] - ts.m.sample[i - 1]).powi(2))
                .sum::<T>()
                / (ts.lenf() - T::one())
                / ts.m.get_std().powi(2),
        ]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["eta"]
    }
}

/// $\eta^e$ — modernisation of [Eta](./struct.Eta.html) for unevenly time series
///
/// $$
/// \eta^e \equiv (t_{N-1} - t_0)^2 \frac{\sum_{i=0}^{N-2} \left(\frac{m_{i+1} - m_i}{t_{i+1} - t_i}\right)^2}{(N - 1)\\,\sigma_m^2}
/// $$
/// where $N$ is the number of observations,
/// $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
/// Note that this definition is a bit different from both \[Kim et al. 2014] and
/// [feets](https://feets.readthedocs.io/en/latest/)
///
/// - Depends on: **time**, **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// Kim et al. 2014, [DOI:10.1051/0004-6361/201323252](https://doi.org/10.1051/0004-6361/201323252)
#[derive(Default)]
pub struct EtaE {}

impl EtaE {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for EtaE
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let sq_slope_sum = (1..ts.lenu())
            .map(|i| {
                ((ts.m.sample[i] - ts.m.sample[i - 1]) / (ts.t.sample[i] - ts.t.sample[i - 1]))
                    .powi(2)
            })
            .filter(|&x| x.is_finite())
            .sum::<T>();
        vec![
            (ts.t.sample[ts.lenu() - 1] - ts.t.sample[0]).powi(2) * sq_slope_sum
                / ts.m.get_std().powi(2)
                / (ts.lenf() - T::one()),
        ]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["eta_e"]
    }
}

/// Kurtosis of magnitude $G_2$
///
/// $$
/// G_2 \equiv \frac{N\\,(N + 1)}{(N - 1)(N - 2)(N - 3)} \frac{\sum_i(m_i - \langle m \rangle)^4}{\sigma_m^2}
/// \- 3\frac{(N + 1)^2}{(N - 2)(N - 3)},
/// $$
/// where $N$ is the number of observations,
/// $\langle m \rangle$ is the mean magnitude,
/// $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **4**
/// - Number of features: **1**
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Kurtosis#Estimators_of_population_kurtosis)
#[derive(Default)]
pub struct Kurtosis {}

impl Kurtosis {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for Kurtosis
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        assert!(ts.lenu() > 3, "Kurtosis requires at least 4 points");
        let m_mean = ts.m.get_mean();
        let n = ts.lenf();
        let n1 = n + T::one();
        let n_1 = n - T::one();
        let n_2 = n - T::two();
        let n_3 = n - T::three();
        vec![
            ts.m.sample.iter().map(|&x| (x - m_mean).powi(4)).sum::<T>() / ts.m.get_std().powi(4)
                * n
                * n1
                / (n_1 * n_2 * n_3)
                - T::three() * n_1.powi(2) / (n_2 * n_3),
        ]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["kurtosis"]
    }
}

/// The slope and noise of the light curve without observation errors in the linear fit
///
/// Least squares fit of the linear stochastic model with constant gaussian noise $\Sigma$ assuming
/// observation errors to be zero:
/// $$
/// m_i = c + \mathrm{slope}\\,t_i + \Sigma \varepsilon_i
/// $$
/// where $c$ is a constant,
/// $\\{\varepsilon_i\\}$ are standard distributed random values.
/// $\mathrm{slope}$ and $\Sigma$ are returned, if $N = 2$ than no least squares fit is done, a
/// slope between a pair of observations $(m_1 - m_0) / (t_1 - t_0)$ and $\Sigma = 0$ is returned.
///
/// - Depends on: **time**, **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **2**
#[derive(Default)]
pub struct LinearTrend {}

impl LinearTrend {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for LinearTrend
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        if ts.lenu() == 2 {
            return vec![
                (ts.m.sample[1] - ts.m.sample[0]) / (ts.t.sample[1] - ts.t.sample[0]),
                T::zero(),
            ];
        }
        let result = fit_straight_line(ts.t.sample, ts.m.sample, None);
        vec![result.slope, T::sqrt(result.slope_sigma2)]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["linear_trend", "linear_trend_sigma"]
    }
}

/// The slope, its error and reduced $\chi^2$ of the light curve in the linear fit
///
/// Least squares fit of the linear stochastic model with gaussian noise described by observation
/// errors $\\{\delta_i\\}$:
/// $$
/// m_i = c + \mathrm{slope}\\,t_i + \delta_i \varepsilon_i
/// $$
/// where $c$ is a constant,
/// $\\{\varepsilon_i\\}$ are standard distributed random values.
///
/// - Depends on: **time**, **magnitude**, **magnitude error**
/// - Minimum number of observations: **2**
/// - Number of features: **3**
#[derive(Default)]
pub struct LinearFit {}

impl LinearFit {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for LinearFit
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        match ts.err2.as_ref() {
            Some(err2) => {
                let result = fit_straight_line(ts.t.sample, ts.m.sample, Some(err2.sample));
                vec![
                    result.slope,
                    T::sqrt(result.slope_sigma2),
                    result.reduced_chi2,
                ]
            }
            None => vec![T::nan(); 3],
        }
    }

    fn get_names(&self) -> Vec<&str> {
        vec![
            "linear_fit_slope",
            "linear_fit_slope_sigma",
            "linear_fit_reduced_chi2",
        ]
    }
}

/// Magnitude percentage ratio
///
/// $$
/// \mathrm{magnitude~}q\mathrm{~to~}n\mathrm{~ratio} \equiv \frac{Q(1-n) - Q(n)}{Q(1-d) - Q(d)},
/// $$
/// where $n$ and $d$ denotes user defined percentage, $Q$ is the quantile function of magnitude
/// distribution.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
pub struct MagnitudePercentageRatio {
    quantile_numerator: f32,
    quantile_denominator: f32,
    name: String,
}

impl MagnitudePercentageRatio {
    pub fn new(quantile_numerator: f32, quantile_denominator: f32) -> Self {
        assert!(
            (quantile_numerator > 0.0)
                && (quantile_numerator < 0.5)
                && (quantile_denominator > 0.0)
                && (quantile_denominator < 0.5),
            "quantiles should be between zero and half"
        );
        Self {
            quantile_numerator,
            quantile_denominator,
            name: format!(
                "magnitude_percentage_ratio_{:.0}_{:.0}",
                100.0 * quantile_numerator,
                100.0 * quantile_denominator
            ),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

impl Default for MagnitudePercentageRatio {
    fn default() -> Self {
        Self::new(0.4, 0.05)
    }
}

impl<T> FeatureEvaluator<T> for MagnitudePercentageRatio
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let q = [
            self.quantile_numerator,
            1.0 - self.quantile_numerator,
            self.quantile_denominator,
            1.0 - self.quantile_denominator,
        ];
        let ppf = ts.m.get_sorted().ppf_many_from_sorted(&q[..]);
        vec![(ppf[1] - ppf[0]) / (ppf[3] - ppf[2])]
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }
}

/// Maximum slope between two neighbour observations
///
/// $$
/// \mathrm{maximum~slope} \equiv \max_{i=0..N-2}\left(\frac{m_{i+1} - m_i}{t_{i+1} - t_i}\right)
/// $$
///
/// - Depends on: **time**, **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
#[derive(Default)]
pub struct MaximumSlope {}

impl MaximumSlope {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for MaximumSlope
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        vec![(1..ts.lenu())
            .map(|i| {
                T::abs(
                    (ts.m.sample[i] - ts.m.sample[i - 1]) / (ts.t.sample[i] - ts.t.sample[i - 1]),
                )
            })
            .filter(|&x| x.is_finite())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("All points of the light curve have the same time")]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["maximum_slope"]
    }
}

/// Median of the absolute value of the difference between magnitude and its median
///
/// $$
/// \mathrm{median~absolute~deviation} \equiv \mathrm{Median}\left(|m_i - \mathrm{Median}(m)|\right).
/// $$
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
#[derive(Default)]
pub struct MedianAbsoluteDeviation {}

impl MedianAbsoluteDeviation {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for MedianAbsoluteDeviation
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let m_median = ts.m.get_median();
        let deviation: Vec<_> = ts.m.sample.iter().map(|&y| T::abs(y - m_median)).collect();
        vec![deviation[..].median()]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["median_absolute_deviation"]
    }
}

/// Fraction of observations inside $\mathrm{Median}(m) \pm q \times \mathrm{Median}(m)$ interval
///
/// $$
/// \mathrm{median~buffer~range}~q~\mathrm{percentage} \equiv \frac{\sum\_i I\_{|m - \mathrm{Median}(m)| < q\\,\mathrm{Median}(m)}(m_i)}{N},
/// $$
/// where $I$ is the [indicator function](https://en.wikipedia.org/wiki/Indicator_function),
/// and $N$ is the number of observations.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
pub struct MedianBufferRangePercentage<T>
where
    T: Float,
{
    quantile: T,
    name: String,
}

impl<T> MedianBufferRangePercentage<T>
where
    T: Float,
{
    pub fn new(quantile: T) -> Self {
        assert!(quantile > T::zero(), "Quanitle should be positive");
        Self {
            quantile,
            name: format!(
                "median_buffer_range_percentage_{:.0}",
                T::hundred() * quantile
            ),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

impl<T> Default for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(0.1_f32.value_as::<T>().unwrap())
    }
}

impl<T> FeatureEvaluator<T> for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let m_median = ts.m.get_median();
        let threshold = self.quantile * m_median;
        vec![
            ts.m.sample
                .iter()
                .cloned()
                .filter(|&y| T::abs(y - m_median) < threshold)
                .count()
                .value_as::<T>()
                .unwrap()
                / ts.lenf(),
        ]
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }
}

/// Maximum deviation of magnitude from its median
///
/// $$
/// \mathrm{percent~amplitude} \equiv \max_i\left(m_i - \mathrm{Median}(m)\right)
///     = \max\\{\max(m) - \mathrm{Median}(m), \mathrm{Median}(m) - \min(m)\\}.
/// $$
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
#[derive(Default)]
pub struct PercentAmplitude {}

impl PercentAmplitude {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for PercentAmplitude
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let m_min = ts.m.get_min();
        let m_max = ts.m.get_max();
        let m_median = ts.m.get_median();
        vec![*[m_max - m_median, m_median - m_min]
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["percent_amplitude"]
    }
}

/// Ratio of $p$ percentile interval to the median
///
/// $$
/// p\mathrm{~percent~difference~magnitude~percentile} \equiv \frac{Q(1-p) - Q(p)}{\mathrm{Median}(m)}.
/// $$
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
pub struct PercentDifferenceMagnitudePercentile {
    quantile: f32,
    name: String,
}

impl PercentDifferenceMagnitudePercentile {
    pub fn new(quantile: f32) -> Self {
        assert!(
            (quantile > 0.0) && (quantile < 0.5),
            "quantiles should be between zero and half"
        );
        Self {
            quantile,
            name: format!(
                "percent_difference_magnitude_percentile_{:.0}",
                100.0 * quantile
            ),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

impl Default for PercentDifferenceMagnitudePercentile {
    fn default() -> Self {
        Self::new(0.05)
    }
}

impl<T> FeatureEvaluator<T> for PercentDifferenceMagnitudePercentile
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let q = [self.quantile, 1.0 - self.quantile];
        let ppf = ts.m.get_sorted().ppf_many_from_sorted(&q[..]);
        vec![(ppf[1] - ppf[0]) / ts.m.get_median()]
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }
}

// See http://doi.org/10.1088/0004-637X/733/1/10
/// A number of features based on Lomb–Scargle periodogram
///
/// Periodogram $P(\omega)$ is an estimate of spectral density of unevenly time series.
/// `Periodogram::new`'s `peaks` argument corresponds to a number of the most significant spectral
/// density peaks to return. For each peak its period and "signal to noise" ratio is returned.
///
/// $$
/// \mathrm{signal~to~noise~of~peak} \equiv \frac{P(\omega_\mathrm{peak}) - \langle P(\omega) \rangle}{\sigma\_{P(\omega)}}.
/// $$
///
/// `Periodogram` can accept another `dyn FeatureEvaluator` for feature extraction from periodogram
/// as it was time series without observation errors. You can even pass one `Periodogram` to another
/// if you are crazy enough
///
/// - Depends on: **time**, **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **$2 \times \mathrm{peaks}~+...$**
pub struct Periodogram<T> {
    peaks: usize,
    freq: PeriodogramFreq<T>,
    features_extractor: FeatureExtractor<T>,
    peak_names: Vec<String>,
    features_names: Vec<String>,
}

impl<T> Periodogram<T>
where
    T: Float,
{
    pub fn new(peaks: usize) -> Self {
        assert!(peaks > 0, "Number of peaks should be at least one");
        Self {
            peaks,
            freq: PeriodogramFreq::Factors(PeriodogramFreqFactors::default()),
            features_extractor: FeatureExtractor::new(vec![]),
            peak_names: (0..peaks)
                .flat_map(|i| vec![format!("period_{}", i), format!("period_s_to_n_{}", i)])
                .collect(),
            features_names: vec![],
        }
    }

    pub fn set_freq_vec(&mut self, freq: Vec<T>) -> &mut Self {
        self.freq = PeriodogramFreq::Vector(freq);
        self
    }

    pub fn set_freq_factors(&mut self, resolution: T, nyquist: T) -> &mut Self {
        self.freq = PeriodogramFreq::Factors(PeriodogramFreqFactors::new(resolution, nyquist));
        self
    }

    pub fn add_features(&mut self, features: VecFE<T>) -> &mut Self {
        self.features_extractor.features.extend(features);
        self.features_names.extend(
            self.features_extractor
                .get_names()
                .iter()
                .map(|name| "periodogram_".to_owned() + name),
        );
        self
    }
}

impl<T> Default for Periodogram<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(1)
    }
}

impl<T> Periodogram<T> {
    fn period(omega: T) -> T
    where
        T: Float,
    {
        T::two() * T::PI() / omega
    }
}

impl<T> FeatureEvaluator<T> for Periodogram<T>
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let pn = periodogram::Periodogram::from_time_series(ts, &self.freq);
        let freq = pn.get_freq();
        let power = pn.get_power();
        let mut pn_as_ts = pn.ts();
        let mut features = match self.peaks {
            1 => {
                let (omega_max, power_max) = pn_as_ts.max_by_m();
                vec![
                    Self::period(omega_max),
                    pn_as_ts.m.signal_to_noise(power_max),
                ]
            }
            _ => power
                .peak_indices_reverse_sorted()
                .iter()
                .map(|&i| {
                    vec![Self::period(freq[i]), pn_as_ts.m.signal_to_noise(power[i])].into_iter()
                })
                .flatten()
                .chain(vec![T::zero()].into_iter().cycle())
                .take(2 * self.peaks)
                .collect(),
        };
        features.extend(self.features_extractor.eval(pn_as_ts));
        features
    }

    fn get_names(&self) -> Vec<&str> {
        self.peak_names
            .iter()
            .chain(self.features_names.iter())
            .map(|name| name.as_str())
            .collect()
    }
}

/// Reduced $\chi^2$ of magnitude measurements
///
/// $$
/// \mathrm{reduced~}\chi^2 \equiv \frac1{N-1} \sum_i\left(\frac{m_i - \langle m \rangle}{\delta\_i}\right)^2,
/// $$
/// where $N$ is the number of observations,
/// and $\langle m \rangle$ is the mean magnitude.
///
/// - Depends on: **magnitude**, **magnitude error**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic)
#[derive(Default)]
pub struct ReducedChi2 {}

impl ReducedChi2 {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for ReducedChi2
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        vec![ts.get_m_reduced_chi2().unwrap_or(T::nan())]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["chi2"]
    }
}

/// Skewness of magnitude $G_1$
///
/// $$
/// G_1 \equiv \frac{N}{(N - 1)(N - 2)} \frac{\sum_i(m_i - \langle m \rangle)^3}{\sigma_m^3},
/// $$
/// where $N$ is the number of observations,
/// $\langle m \rangle$ is the mean magnitude,
/// $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **3**
/// - Number of features: **1**
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Skewness#Sample_skewness)
#[derive(Default)]
pub struct Skew {}

impl Skew {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for Skew
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        assert!(ts.lenu() > 2, "Skew requires at least 3 points");
        let m_mean = ts.m.get_mean();
        let n = ts.lenf();
        let n_1 = n - T::one();
        let n_2 = n_1 - T::one();
        vec![
            ts.m.sample.iter().map(|&x| (x - m_mean).powi(3)).sum::<T>() / ts.m.get_std().powi(3)
                * n
                / (n_1 * n_2),
        ]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["skew"]
    }
}

/// Standard deviation of magnitude $\sigma_m$
///
/// $$
/// \sigma_m \equiv \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)},
/// $$
///
/// $N$ is the number of observations
/// and $\langle m \rangle$ is the mean magnitude.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Standard_deviation)
#[derive(Default)]
pub struct StandardDeviation {}

impl StandardDeviation {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for StandardDeviation
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        vec![ts.m.get_std()]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["standard_deviation"]
    }
}

/// Stetson $K$ coefficient described light curve shape
///
/// $$
/// \mathrm{Stetson}~K \equiv \frac{\sum_i\left|\frac{m_i - \langle m \rangle}{\delta_i}\right|}{\sqrt{N\\,\chi^2}},
/// $$
/// where N is the number of observations,
/// $\langle m \rangle$ is the mean magnitude
/// and $\chi^2 = \sum_i\left(\frac{m_i - \langle m \rangle}{\delta\_i}\right)^2$.
///
/// - Depends on: **magnitude**, **magnitude error**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// P. B. Statson, 1996. [DOI:10.1086/133808](https://doi.org/10.1086/133808)
#[derive(Default)]
pub struct StetsonK {}

impl StetsonK {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for StetsonK
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let m_weighted_mean = ts.get_m_weighted_mean();
        let m_reduced_chi2 = ts.get_m_reduced_chi2();
        match ts.err2.as_ref() {
            Some(err2) => {
                let mean = m_weighted_mean.unwrap();
                let chi2 = (ts.lenf() - T::one()) * m_reduced_chi2.unwrap();
                vec![
                    ts.m.sample
                        .iter()
                        .zip(err2.sample.iter())
                        .map(|(&y, &err2)| T::abs(y - mean) / T::sqrt(err2))
                        .sum::<T>()
                        / T::sqrt(ts.lenf() * chi2),
                ]
            }
            None => vec![T::nan()],
        }
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["stetson_K"]
    }
}

// To implement
// doi.org/10.1111/j.1365-2966.2012.22061.x
// struct CAR {}

#[cfg(test)]
mod tests {
    use std::f64;

    use rand::prelude::*;

    use super::*;
    use light_curve_common::{all_close, linspace};

    macro_rules! feature_test{
        ($name: ident, $fe: tt, $desired: expr, $y: expr $(,)?) => {
            feature_test!($name, $fe, $desired, $y, $y);
        };
        ($name: ident, $fe: tt, $desired: expr, $x: expr, $y: expr $(,)?) => {
            feature_test!($name, $fe, $desired, $x, $y, None);
        };
        ($name: ident, $fe: tt, $desired: expr, $x: expr, $y: expr, $err2: expr $(,)?) => {
            feature_test!($name, $fe, $desired, $x, $y, $err2, 1e-6);
        };
        ($name: ident, $fe: tt, $desired: expr, $x: expr, $y: expr, $err2: expr, $tol: expr $(,)?) => {
            #[test]
            fn $name() {
                let fe = FeatureExtractor{
                    features: vec!$fe,
                };
                let desired = $desired;
                let x = $x;
                let y = $y;
                let ts = TimeSeries::new(&x[..], &y[..], $err2);
                let actual = fe.eval(ts);
                all_close(&desired[..], &actual[..], $tol);

                let names = fe.get_names();
                assert_eq!(actual.len(), names.len(),
                    "Length of values and names should be the same");
            }
        };
    }

    feature_test!(
        amplitude,
        [Box::new(Amplitude::new())],
        [1.0],
        [0.0_f32, 1.0, 2.0],
    );

    feature_test!(
        beyond_n_std,
        [
            Box::new(BeyondNStd::default()),
            Box::new(BeyondNStd::new(1.0)), // should be the same as the previous one
            Box::new(BeyondNStd::new(2.0)),
        ],
        [0.2, 0.2, 0.0],
        [1.0_f32, 2.0, 3.0, 4.0, 100.0],
    );

    feature_test!(
        cumsum,
        [Box::new(Cusum::new())],
        [0.3589213],
        [1.0_f32, 1.0, 1.0, 5.0, 8.0, 20.0],
    );

    feature_test!(
        eta,
        [Box::new(Eta::new())],
        [1.11338],
        [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 109.0],
    );

    feature_test!(
        eta_e,
        [Box::new(EtaE::new())],
        [6.26210526],
        [1.0_f32, 2.0, 5.0, 10.0],
        [1.0_f32, 1.0, 6.0, 8.0],
    );

    feature_test!(
        kurtosis,
        [Box::new(Kurtosis::new())],
        [-1.2],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
    );

    feature_test!(
        linear_trend,
        [Box::new(LinearTrend::new())],
        [1.38198758, 0.24532195657979344],
        [1.0_f32, 3.0, 5.0, 7.0, 11.0, 13.0],
        [1.0_f32, 2.0, 3.0, 8.0, 10.0, 19.0],
    );

    feature_test!(
        magnitude_percentage_ratio,
        [
            Box::new(MagnitudePercentageRatio::default()),
            Box::new(MagnitudePercentageRatio::new(0.4, 0.05)), // should be the same
            Box::new(MagnitudePercentageRatio::new(0.2, 0.05)),
            Box::new(MagnitudePercentageRatio::new(0.4, 0.1)),
        ],
        [0.12886598, 0.12886598, 0.7628866, 0.13586957],
        [
            80.0_f32, 13.0, 20.0, 20.0, 75.0, 25.0, 100.0, 1.0, 2.0, 3.0, 7.0, 30.0, 5.0, 9.0,
            10.0, 70.0, 80.0, 92.0, 97.0, 17.0
        ],
    );

    feature_test!(
        maximum_slope_positive,
        [Box::new(MaximumSlope::new())],
        [1.0],
        [0.0_f32, 2.0, 4.0, 5.0, 7.0, 9.0],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
    );

    feature_test!(
        maximum_slope_negative,
        [Box::new(MaximumSlope::new())],
        [1.0],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0_f32, 0.5, 1.0, 0.0, 0.5, 1.0],
    );

    feature_test!(
        median_absolute_deviation,
        [Box::new(MedianAbsoluteDeviation::new())],
        [4.0],
        [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0, 6.0, 100.0],
    );

    feature_test!(
        median_buffer_range_percentage,
        [
            Box::new(MedianBufferRangePercentage::default()),
            Box::new(MedianBufferRangePercentage::new(0.1)), // should be the same
            Box::new(MedianBufferRangePercentage::new(0.2)),
        ],
        [0.555555555, 0.555555555, 0.777777777],
        [1.0_f32, 41.0, 49.0, 49.0, 50.0, 51.0, 52.0, 58.0, 100.0],
    );

    feature_test!(
        percent_amplitude,
        [Box::new(PercentAmplitude::new())],
        [96.0],
        [1.0_f32, 1.0, 1.0, 2.0, 4.0, 5.0, 5.0, 98.0, 100.0],
    );

    feature_test!(
        percent_difference_magnitude_percentile,
        [
            Box::new(PercentDifferenceMagnitudePercentile::default()),
            Box::new(PercentDifferenceMagnitudePercentile::new(0.05)), // should be the same
            Box::new(PercentDifferenceMagnitudePercentile::new(0.1)),
        ],
        [4.85, 4.85, 4.6],
        [
            80.0_f32, 13.0, 20.0, 20.0, 75.0, 25.0, 100.0, 1.0, 2.0, 3.0, 7.0, 30.0, 5.0, 9.0,
            10.0, 70.0, 80.0, 92.0, 97.0, 17.0
        ],
    );

    #[test]
    fn periodogram_evenly_sinus() {
        let fe = FeatureExtractor {
            features: vec![Box::new(Periodogram::default())],
        };
        let period = 0.22;
        let x = linspace(0.0_f32, 1.0, 100);
        let y: Vec<_> = x
            .iter()
            .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / period * x + 0.5) + 4.0)
            .collect();
        let ts = TimeSeries::new(&x[..], &y[..], None);
        let desired = [period];
        let actual = [fe.eval(ts)[0]]; // Test period only
        all_close(&desired[..], &actual[..], 1e-3);
    }

    #[test]
    fn periodogram_unevenly_sinus() {
        let fe = FeatureExtractor {
            features: vec![Box::new(Periodogram::default())],
        };
        let period = 0.22;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..100).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / period * x + 0.5) + 4.0)
            .collect();
        let ts = TimeSeries::new(&x[..], &y[..], None);
        let desired = [period];
        let actual = [fe.eval(ts)[0]]; // Test period only
        all_close(&desired[..], &actual[..], 1e-3);
    }

    #[test]
    fn periodogram_one_peak_vs_two_peaks() {
        let fe = FeatureExtractor {
            features: vec![Box::new(Periodogram::new(1)), Box::new(Periodogram::new(2))],
        };
        let period = 0.22;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..100).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / period * x + 0.5) + 4.0)
            .collect();
        let ts = TimeSeries::new(&x[..], &y[..], None);
        let features = fe.eval(ts);
        all_close(
            &[features[0], features[1]],
            &[features[2], features[3]],
            1e-6,
        );
    }

    #[test]
    fn periodogram_unevenly_sinus_cosine() {
        let fe = FeatureExtractor {
            features: vec![Box::new(Periodogram::new(2))],
        };
        let period1 = 0.22;
        let period2 = 0.6;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..1000).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period1 * x + 0.5)
                    + -5.0 * f32::cos(2.0 * std::f32::consts::PI / period2 * x + 0.5)
                    + 4.0
            })
            .collect();
        let ts = TimeSeries::new(&x[..], &y[..], None);
        let desired = [period2, period1];
        let features = fe.eval(ts);
        let actual = [features[0], features[2]]; // Test period only
        all_close(&desired[..], &actual[..], 1e-2);
        assert!(features[1] > features[3])
    }

    #[test]
    fn periodogram_unevenly_sinus_cosine_noised() {
        let fe = FeatureExtractor {
            features: vec![Box::new(Periodogram::new(2))],
        };
        let period1 = 0.22;
        let period2 = 0.6;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..1000).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period1 * x + 0.5)
                    + -5.0 * f32::cos(2.0 * std::f32::consts::PI / period2 * x + 0.5)
                    + 10.0 * rng.gen::<f32>()
                    + 4.0
            })
            .collect();
        let ts = TimeSeries::new(&x[..], &y[..], None);
        let desired = [period2, period1];
        let features = fe.eval(ts);
        let actual = [features[0], features[2]]; // Test period only
        all_close(&desired[..], &actual[..], 3e-2);
        assert!(features[1] > features[3])
    }

    feature_test!(
        skew,
        [Box::new(Skew::new())],
        [0.4626804756753222],
        [2.0_f32, 3.0, 5.0, 7.0, 11.0, 13.0],
    );

    feature_test!(
        standard_deviation,
        [Box::new(StandardDeviation::new())],
        [1.5811388300841898],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
    );

    feature_test!(
        stetson_k_square_wave,
        [Box::new(StetsonK::new())],
        [1.0],
        [1.0; 1000], // isn't used
        (0..1000)
            .map(|i| {
                if i < 500 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect::<Vec<_>>(),
        Some(&[1.0; 1000]),
    );

    // Slow convergence, use high tol
    feature_test!(
        stetson_k_sinus,
        [Box::new(StetsonK::new())],
        [8_f64.sqrt() / f64::consts::PI],
        [1.0; 1000], // isn't used
        linspace(0.0, 2.0 * f64::consts::PI, 1000)
            .iter()
            .map(|&x| f64::sin(x))
            .collect::<Vec<_>>(),
        Some(&[1.0; 1000]),
        1e-3,
    );

    feature_test!(
        stetson_k_sawtooth,
        [Box::new(StetsonK::new())],
        [12_f64.sqrt() / 4.0],
        [1.0; 1000], // isn't used
        linspace(0.0, 1.0, 1000),
        Some(&[1.0; 1000]),
    );

    // It seems that Stetson (1996) formula for this case is wrong by the factor of 2 * sqrt((N-1) / N)
    feature_test!(
        stetson_k_single_peak,
        [Box::new(StetsonK::new())],
        [2.0 * 99.0_f64.sqrt() / 100.0],
        [1.0; 100], // isn't used
        (0..100)
            .map(|i| {
                if i == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect::<Vec<_>>(),
        Some(&[1.0; 100]),
    );
}
