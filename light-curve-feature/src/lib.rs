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
        let value = if ts.m.get_std().is_zero() {
            T::zero()
        } else {
            (cumsum[..].maximum() - cumsum[..].minimum()) / (ts.m.get_std() * ts.lenf())
        };
        vec![value]
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
        let value = if ts.m.get_std().is_zero() {
            T::zero()
        } else {
            (1..ts.lenu())
                .map(|i| (ts.m.sample[i] - ts.m.sample[i - 1]).powi(2))
                .sum::<T>()
                / (ts.lenf() - T::one())
                / ts.m.get_std().powi(2)
        };
        vec![value]
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
        let value = if ts.m.get_std().is_zero() {
            T::zero()
        } else {
            (ts.t.sample[ts.lenu() - 1] - ts.t.sample[0]).powi(2) * sq_slope_sum
                / ts.m.get_std().powi(2)
                / (ts.lenf() - T::one())
        };
        vec![value]
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
        let value = if ts.m.get_std().is_zero() {
            T::zero()
        } else {
            ts.m.sample.iter().map(|&x| (x - m_mean).powi(4)).sum::<T>() / ts.m.get_std().powi(4)
                * n
                * n1
                / (n_1 * n_2 * n_3)
                - T::three() * n_1.powi(2) / (n_2 * n_3)
        };
        vec![value]
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
        let numerator = ppf[1] - ppf[0];
        let denumerator = ppf[3] - ppf[2];
        let value = if numerator.is_zero() & denumerator.is_zero() {
            T::zero()
        } else {
            numerator / denumerator
        };
        vec![value]
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
        let nominator = ppf[1] - ppf[0];
        let denominator = ts.m.get_median();
        let value = if nominator.is_zero() & denominator.is_zero() {
            T::zero()
        } else {
            (ppf[1] - ppf[0]) / ts.m.get_median()
        };
        vec![value]
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
        let mut features: Vec<_> = power
            .peak_indices_reverse_sorted()
            .iter()
            .map(|&i| vec![Self::period(freq[i]), pn_as_ts.m.signal_to_noise(power[i])].into_iter())
            .flatten()
            .chain(vec![T::zero()].into_iter().cycle())
            .take(2 * self.peaks)
            .collect();
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
        let value = if ts.m.get_std().is_zero() {
            T::zero()
        } else {
            ts.m.sample.iter().map(|&x| (x - m_mean).powi(3)).sum::<T>() / ts.m.get_std().powi(3)
                * n
                / (n_1 * n_2)
        };
        vec![value]
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
                let value = if chi2.is_zero() {
                    T::zero()
                } else {
                    ts.m.sample
                        .iter()
                        .zip(err2.sample.iter())
                        .map(|(&y, &err2)| T::abs(y - mean) / T::sqrt(err2))
                        .sum::<T>()
                        / T::sqrt(ts.lenf() * chi2)
                };
                vec![value]
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

    /// See [Issue #2](https://github.com/hombit/light-curve/issues/2)
    #[test]
    fn eta_e_finite() {
        let fe = FeatureExtractor {
            features: vec![Box::new(EtaE::default())],
        };
        let x = [
            58197.50390625,
            58218.48828125,
            58218.5078125,
            58222.46875,
            58222.4921875,
            58230.48046875,
            58244.48046875,
            58246.43359375,
            58247.4609375,
            58247.48046875,
            58247.48046875,
            58249.44140625,
            58249.4765625,
            58255.4609375,
            58256.41796875,
            58256.45703125,
            58257.44140625,
            58257.4609375,
            58258.44140625,
            58259.4453125,
            58262.3828125,
            58263.421875,
            58266.359375,
            58268.42578125,
            58269.41796875,
            58270.40625,
            58271.4609375,
            58273.421875,
            58274.33984375,
            58275.40234375,
            58276.42578125,
            58277.3984375,
            58279.40625,
            58280.36328125,
            58282.44921875,
            58283.3828125,
            58285.3828125,
            58286.34375,
            58288.44921875,
            58289.4453125,
            58290.3828125,
            58291.3203125,
            58292.3359375,
            58293.30078125,
            58296.32421875,
            58297.33984375,
            58298.33984375,
            58301.36328125,
            58302.359375,
            58303.33984375,
            58304.36328125,
            58305.36328125,
            58307.3828125,
            58308.37890625,
            58310.38671875,
            58311.3828125,
            58313.38671875,
            58314.421875,
            58315.33984375,
            58316.34375,
            58317.40625,
            58318.33984375,
            58320.3515625,
            58321.3515625,
            58322.359375,
            58323.27734375,
            58324.23828125,
            58326.3203125,
            58327.31640625,
            58329.3203125,
            58330.37890625,
            58332.296875,
            58333.32421875,
            58334.34765625,
            58336.23046875,
            58338.2109375,
            58340.3046875,
            58341.328125,
            58342.328125,
            58343.32421875,
            58344.31640625,
            58345.32421875,
            58348.21875,
            58351.234375,
            58354.2578125,
            58355.2734375,
            58356.1953125,
            58358.25390625,
            58360.21875,
            58361.234375,
            58366.18359375,
            58370.15234375,
            58373.171875,
            58374.171875,
            58376.171875,
            58425.0859375,
            58427.0859375,
            58428.1015625,
            58430.0859375,
            58431.12890625,
            58432.0859375,
            58433.08984375,
            58436.0859375,
        ];
        let y = [
            17.357999801635742,
            17.329999923706055,
            17.332000732421875,
            17.312999725341797,
            17.30500030517578,
            17.31599998474121,
            17.27899932861328,
            17.305999755859375,
            17.333999633789062,
            17.332000732421875,
            17.332000732421875,
            17.323999404907227,
            17.256000518798828,
            17.308000564575195,
            17.290000915527344,
            17.298999786376953,
            17.270000457763672,
            17.270000457763672,
            17.297000885009766,
            17.288000106811523,
            17.358999252319336,
            17.273000717163086,
            17.354999542236328,
            17.301000595092773,
            17.2810001373291,
            17.299999237060547,
            17.341999053955078,
            17.30500030517578,
            17.29599952697754,
            17.336000442504883,
            17.31399917602539,
            17.336999893188477,
            17.304000854492188,
            17.309999465942383,
            17.304000854492188,
            17.29199981689453,
            17.31100082397461,
            17.28499984741211,
            17.327999114990234,
            17.347999572753906,
            17.32200050354004,
            17.319000244140625,
            17.2810001373291,
            17.327999114990234,
            17.291000366210938,
            17.3439998626709,
            17.336000442504883,
            17.27899932861328,
            17.38800048828125,
            17.27899932861328,
            17.297000885009766,
            17.29599952697754,
            17.312000274658203,
            17.253999710083008,
            17.312000274658203,
            17.284000396728516,
            17.319000244140625,
            17.32200050354004,
            17.290000915527344,
            17.31599998474121,
            17.28499984741211,
            17.30299949645996,
            17.284000396728516,
            17.336000442504883,
            17.31399917602539,
            17.356000900268555,
            17.308000564575195,
            17.31999969482422,
            17.301000595092773,
            17.325000762939453,
            17.30900001525879,
            17.29800033569336,
            17.29199981689453,
            17.339000701904297,
            17.32699966430664,
            17.31800079345703,
            17.320999145507812,
            17.315000534057617,
            17.304000854492188,
            17.327999114990234,
            17.308000564575195,
            17.34000015258789,
            17.325000762939453,
            17.322999954223633,
            17.30900001525879,
            17.308000564575195,
            17.275999069213867,
            17.33799934387207,
            17.343000411987305,
            17.437999725341797,
            17.280000686645508,
            17.305999755859375,
            17.320999145507812,
            17.325000762939453,
            17.32699966430664,
            17.339000701904297,
            17.298999786376953,
            17.29199981689453,
            17.336000442504883,
            17.32699966430664,
            17.28499984741211,
            17.284000396728516,
            17.257999420166016,
        ];
        let ts = TimeSeries::new(&x[..], &y[..], None);
        let actual: f32 = fe.eval(ts)[0];
        assert!(actual.is_finite());
    }

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

    /// See [Issue #3](https://github.com/hombit/light-curve/issues/3)
    #[test]
    fn linear_trend_finite_sigma() {
        let fe = FeatureExtractor {
            features: vec![Box::new(LinearTrend::default())],
        };
        let x = [
            58216.51171875,
            58217.48828125,
            58217.5078125,
            58218.49609375,
            58218.5078125,
            58228.45703125,
            58229.45703125,
            58230.47265625,
            58242.46484375,
            58244.4765625,
            58244.49609375,
            58246.4453125,
            58247.46484375,
            58249.45703125,
            58249.47265625,
            58254.46484375,
            58255.4453125,
            58256.4375,
            58256.46484375,
            58257.421875,
            58257.45703125,
            58258.4453125,
            58259.44921875,
            58261.48828125,
            58262.38671875,
            58263.421875,
            58266.359375,
            58268.42578125,
            58269.4140625,
            58270.44140625,
            58271.4609375,
            58273.421875,
            58274.33984375,
            58275.40234375,
            58276.4296875,
            58277.39453125,
            58279.40625,
            58280.375,
            58282.4453125,
            58283.38671875,
            58285.3828125,
            58286.34375,
            58288.44921875,
            58289.4453125,
            58290.3828125,
            58291.484375,
            58292.46875,
            58293.3203125,
            58294.46875,
            58296.32421875,
            58297.4140625,
            58298.43359375,
            58299.40234375,
            58300.37890625,
            58301.3828125,
            58303.37890625,
            58304.3203125,
            58305.3828125,
            58307.3828125,
            58310.38671875,
            58311.3828125,
            58312.421875,
            58313.38671875,
            58314.41796875,
            58316.33984375,
            58317.40625,
            58318.33984375,
            58320.36328125,
            58321.3515625,
            58322.39453125,
            58323.2734375,
            58324.23828125,
            58325.29296875,
            58326.33984375,
            58327.33984375,
            58329.34375,
            58330.37890625,
            58332.32421875,
            58333.32421875,
            58334.35546875,
            58336.34375,
            58338.3359375,
            58340.3046875,
            58341.328125,
            58342.33203125,
            58343.32421875,
            58344.30859375,
            58345.32421875,
            58346.31640625,
            58349.30078125,
            58351.21484375,
            58354.2578125,
            58354.359375,
            58355.28125,
            58356.1953125,
            58356.29296875,
            58357.2109375,
            58358.25390625,
            58360.27734375,
            58366.1875,
            58370.2578125,
            58373.171875,
            58374.171875,
            58425.0859375,
            58427.109375,
            58428.1015625,
            58431.1328125,
        ];
        let y = [
            18.614999771118164,
            18.714000701904297,
            18.665000915527344,
            18.732999801635742,
            18.658000946044922,
            18.70199966430664,
            18.641000747680664,
            18.631999969482422,
            18.659000396728516,
            18.68899917602539,
            18.75,
            18.767000198364258,
            18.70400047302246,
            18.85300064086914,
            18.7450008392334,
            18.770000457763672,
            18.67799949645996,
            18.70800018310547,
            18.724000930786133,
            18.70400047302246,
            18.680999755859375,
            18.733999252319336,
            18.64900016784668,
            18.67099952697754,
            18.707000732421875,
            18.781999588012695,
            18.691999435424805,
            18.695999145507812,
            18.684999465942383,
            18.72800064086914,
            18.68600082397461,
            18.743000030517578,
            18.718000411987305,
            18.645000457763672,
            18.708999633789062,
            18.69700050354004,
            18.704999923706055,
            18.71500015258789,
            18.729000091552734,
            18.69499969482422,
            18.660999298095703,
            18.718000411987305,
            18.628000259399414,
            18.76799964904785,
            18.733999252319336,
            18.735000610351562,
            18.70800018310547,
            18.753999710083008,
            18.66699981689453,
            18.735000610351562,
            18.697999954223633,
            19.034000396728516,
            18.628999710083008,
            18.711000442504883,
            18.76799964904785,
            18.701000213623047,
            18.687000274658203,
            18.733999252319336,
            18.715999603271484,
            18.69099998474121,
            18.711999893188477,
            18.715999603271484,
            18.764999389648438,
            18.663999557495117,
            18.722000122070312,
            18.70400047302246,
            18.690000534057617,
            18.67099952697754,
            18.65999984741211,
            18.7549991607666,
            18.666000366210938,
            18.60700035095215,
            18.715999603271484,
            18.732999801635742,
            18.788999557495117,
            18.791000366210938,
            18.714000701904297,
            18.738000869750977,
            18.672000885009766,
            18.74799919128418,
            18.69099998474121,
            18.718000411987305,
            18.64699935913086,
            18.70800018310547,
            18.656999588012695,
            18.672000885009766,
            18.711999893188477,
            18.781999588012695,
            18.628000259399414,
            18.698999404907227,
            18.722000122070312,
            18.70599937438965,
            18.645000457763672,
            18.80500030517578,
            18.820999145507812,
            18.75,
            18.77400016784668,
            18.761999130249023,
            19.656999588012695,
            18.76300048828125,
            18.71299934387207,
            18.750999450683594,
            18.70800018310547,
            18.71500015258789,
            18.638999938964844,
            18.677000045776367,
            18.69700050354004,
        ];
        let ts = TimeSeries::new(&x[..], &y[..], None);
        let sigma: f32 = fe.eval(ts)[1];
        assert!(sigma.is_finite());
    }

    /// See [Issue #3](https://github.com/hombit/light-curve/issues/3)
    #[test]
    fn linear_trend_finite_trend_and_sigma_1() {
        let fe = FeatureExtractor {
            features: vec![Box::new(LinearTrend::default())],
        };
        let x = [
            58231.140625,
            58303.4765625,
            58314.44140625,
            58315.484375,
            58316.46875,
            58319.47265625,
            58321.48828125,
            58323.48828125,
            58324.48828125,
            58325.484375,
            58329.48828125,
            58330.41796875,
            58333.48828125,
            58334.4453125,
            58335.46484375,
            58336.4609375,
            58337.48828125,
            58338.48828125,
            58342.484375,
            58343.484375,
            58344.46484375,
            58345.47265625,
            58346.44140625,
            58347.44921875,
            58348.4453125,
            58349.484375,
            58350.4921875,
            58351.484375,
            58352.48828125,
            58353.453125,
            58353.49609375,
            58354.453125,
            58354.484375,
            58355.40625,
            58355.48046875,
            58356.453125,
            58356.484375,
            58357.44921875,
            58357.5078125,
            58358.44921875,
            58359.48828125,
            58360.49609375,
            58361.5078125,
            58363.47265625,
            58364.4921875,
            58365.48828125,
            58366.484375,
            58367.4921875,
            58368.46484375,
            58369.4296875,
            58370.48828125,
            58371.45703125,
            58372.4921875,
            58373.4921875,
            58374.48828125,
            58375.4921875,
            58376.4375,
            58377.4453125,
            58378.42578125,
            58379.4296875,
            58380.453125,
            58382.5,
            58383.515625,
            58384.51171875,
            58385.5078125,
            58386.4375,
            58387.46484375,
            58388.52734375,
            58389.48828125,
            58397.42578125,
            58424.35546875,
            58425.33203125,
            58426.41796875,
            58427.44921875,
            58430.45703125,
            58431.28515625,
            58432.28515625,
            58434.34765625,
            58436.33984375,
            58437.34765625,
            58441.41015625,
            58443.38671875,
            58447.41015625,
            58449.36328125,
            58450.35546875,
            58455.2890625,
            58455.36328125,
            58456.22265625,
            58456.27734375,
            58457.26953125,
            58464.265625,
            58465.265625,
            58468.27734375,
            58471.2421875,
            58472.265625,
            58474.3203125,
            58476.3046875,
            58480.31640625,
            58481.31640625,
            58482.19921875,
        ];
        let y = [
            19.08300018310547,
            18.988000869750977,
            19.086999893188477,
            18.95400047302246,
            19.076000213623047,
            19.076000213623047,
            19.090999603271484,
            18.966999053955078,
            19.041000366210938,
            19.089000701904297,
            19.05699920654297,
            19.097000122070312,
            19.132999420166016,
            19.104000091552734,
            19.06100082397461,
            19.128000259399414,
            19.099000930786133,
            19.06599998474121,
            19.100000381469727,
            19.08300018310547,
            19.1200008392334,
            19.115999221801758,
            19.128999710083008,
            19.07900047302246,
            19.16699981689453,
            19.179000854492188,
            19.1560001373291,
            19.16200065612793,
            19.110000610351562,
            19.14900016784668,
            19.10700035095215,
            19.104999542236328,
            19.145000457763672,
            19.091999053955078,
            19.091999053955078,
            19.225000381469727,
            19.086000442504883,
            19.054000854492188,
            19.17799949645996,
            19.17099952697754,
            19.1200008392334,
            19.02899932861328,
            19.18000030517578,
            19.10700035095215,
            19.118000030517578,
            19.128000259399414,
            19.166000366210938,
            19.08300018310547,
            19.124000549316406,
            19.106000900268555,
            19.10700035095215,
            19.097999572753906,
            19.106000900268555,
            19.107999801635742,
            19.075000762939453,
            18.965999603271484,
            19.134000778198242,
            19.136999130249023,
            19.150999069213867,
            19.1200008392334,
            19.149999618530273,
            19.152999877929688,
            19.013999938964844,
            19.06800079345703,
            19.101999282836914,
            19.093000411987305,
            19.107999801635742,
            19.054000854492188,
            19.062000274658203,
            19.174999237060547,
            19.05299949645996,
            19.04400062561035,
            19.149999618530273,
            19.136999130249023,
            19.152999877929688,
            19.16900062561035,
            18.986000061035156,
            19.204999923706055,
            19.091999053955078,
            19.038999557495117,
            19.246999740600586,
            19.107999801635742,
            19.082000732421875,
            19.148000717163086,
            19.128999710083008,
            19.1560001373291,
            19.187999725341797,
            19.17300033569336,
            19.163000106811523,
            19.1299991607666,
            19.158000946044922,
            19.163999557495117,
            19.10099983215332,
            19.125,
            19.138999938964844,
            19.09000015258789,
            19.19300079345703,
            19.128000259399414,
            19.143999099731445,
            19.21500015258789,
        ];
        let ts: TimeSeries<f32> = TimeSeries::new(&x[..], &y[..], None);
        let actual = fe.eval(ts);
        assert!(actual.iter().all(|x| x.is_finite()));
    }

    /// See [Issue #3](https://github.com/hombit/light-curve/issues/3)
    #[test]
    fn linear_trend_finite_trend_and_sigma_2() {
        let fe = FeatureExtractor {
            features: vec![Box::new(LinearTrend::default())],
        };
        let x = [
            58231.140625,
            58303.4765625,
            58314.44140625,
            58315.484375,
            58316.46875,
            58319.47265625,
            58321.48828125,
            58323.48828125,
            58324.48828125,
            58325.484375,
            58329.48828125,
            58330.41796875,
            58333.48828125,
            58334.4453125,
            58335.46484375,
            58336.4609375,
            58337.48828125,
            58338.48828125,
            58342.484375,
            58343.484375,
            58344.46484375,
            58345.47265625,
            58346.44140625,
            58347.44921875,
            58348.4453125,
            58349.484375,
            58350.4921875,
            58351.484375,
            58352.48828125,
            58353.453125,
            58353.49609375,
            58354.453125,
            58354.484375,
            58355.40625,
            58355.48046875,
            58356.453125,
            58356.484375,
            58357.44921875,
            58357.5078125,
            58358.44921875,
            58359.48828125,
            58360.49609375,
            58361.5078125,
            58363.47265625,
            58364.4921875,
            58365.48828125,
            58366.484375,
            58367.4921875,
            58368.46484375,
            58369.4296875,
            58370.48828125,
            58371.45703125,
            58372.4921875,
            58373.4921875,
            58374.48828125,
            58375.4921875,
            58376.4375,
            58377.4453125,
            58378.42578125,
            58379.4296875,
            58380.453125,
            58382.5,
            58383.515625,
            58384.51171875,
            58385.5078125,
            58386.4375,
            58387.46484375,
            58388.52734375,
            58389.48828125,
            58397.42578125,
            58424.35546875,
            58425.33203125,
            58426.41796875,
            58427.44921875,
            58430.45703125,
            58431.28515625,
            58432.28515625,
            58434.34765625,
            58436.33984375,
            58437.34765625,
            58441.41015625,
            58443.38671875,
            58447.41015625,
            58449.36328125,
            58450.35546875,
            58455.2890625,
            58455.36328125,
            58456.22265625,
            58456.27734375,
            58457.26953125,
            58464.265625,
            58465.265625,
            58468.27734375,
            58471.2421875,
            58472.265625,
            58474.3203125,
            58476.3046875,
            58480.31640625,
            58481.31640625,
            58482.19921875,
        ];
        let y = [
            17.996000289916992,
            18.047000885009766,
            17.983999252319336,
            18.006999969482422,
            18.062000274658203,
            18.02899932861328,
            18.003999710083008,
            17.97599983215332,
            17.992000579833984,
            18.011999130249023,
            18.055999755859375,
            18.013999938964844,
            17.979999542236328,
            18.023000717163086,
            18.034000396728516,
            18.024999618530273,
            18.027999877929688,
            18.017000198364258,
            18.01300048828125,
            18.040000915527344,
            18.006999969482422,
            18.016000747680664,
            18.006999969482422,
            18.000999450683594,
            17.99799919128418,
            18.000999450683594,
            18.038999557495117,
            18.047000885009766,
            18.011999130249023,
            18.03700065612793,
            18.027999877929688,
            18.0,
            18.006000518798828,
            17.957000732421875,
            18.013999938964844,
            18.017000198364258,
            18.04199981689453,
            18.01799964904785,
            18.101999282836914,
            18.051000595092773,
            18.05699920654297,
            18.01300048828125,
            18.027000427246094,
            18.027000427246094,
            18.031999588012695,
            18.0049991607666,
            18.009000778198242,
            18.059999465942383,
            18.018999099731445,
            18.024999618530273,
            18.035999298095703,
            18.02400016784668,
            18.038000106811523,
            18.06100082397461,
            18.02899932861328,
            18.038000106811523,
            18.047000885009766,
            18.01799964904785,
            18.0310001373291,
            18.034000396728516,
            17.97100067138672,
            18.02400016784668,
            18.033000946044922,
            18.018999099731445,
            18.05500030517578,
            18.030000686645508,
            18.02199935913086,
            18.014999389648438,
            18.006000518798828,
            18.045000076293945,
            17.981000900268555,
            18.040000915527344,
            18.003000259399414,
            18.02199935913086,
            18.04199981689453,
            18.04800033569336,
            18.045000076293945,
            18.059999465942383,
            18.062000274658203,
            18.058000564575195,
            18.0310001373291,
            18.041000366210938,
            18.20599937438965,
            17.993000030517578,
            18.030000686645508,
            17.996000289916992,
            18.06599998474121,
            18.030000686645508,
            18.05900001525879,
            18.024999618530273,
            18.05500030517578,
            17.98900032043457,
            18.017000198364258,
            17.950000762939453,
            17.996999740600586,
            18.03499984741211,
            17.98900032043457,
            17.986000061035156,
            18.020999908447266,
            18.075000762939453,
        ];
        let ts: TimeSeries<f32> = TimeSeries::new(&x[..], &y[..], None);
        let actual = fe.eval(ts);
        assert!(actual.iter().all(|x| x.is_finite()));
    }

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
        magnitude_percentage_ratio_plateau,
        [Box::new(MagnitudePercentageRatio::default())],
        [0.0],
        [0.0; 10],
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
        median_buffer_range_percentage_plateau,
        [Box::new(MedianBufferRangePercentage::default())],
        [0.0],
        [0.0; 10],
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
    fn periodogram_plateau() {
        let fe = FeatureExtractor {
            features: vec![Box::new(Periodogram::default())],
        };
        let x = linspace(0.0_f32, 1.0, 100);
        let y = [0.0_f32; 100];
        let ts = TimeSeries::new(&x[..], &y[..], None);
        let desired = vec![0.0, 0.0];
        let actual = fe.eval(ts);
        assert_eq!(desired, actual);
    }

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

    feature_test!(
        stetson_k_plateau,
        [Box::new(StetsonK::new())],
        [0.0],
        [1.0; 100], // isn't used
        [1.0; 100],
        Some(&[1.0; 100]),
    );
}
