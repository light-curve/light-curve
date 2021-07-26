use crate::float_trait::Float;
use crate::sorted_array::SortedArray;
use conv::{ConvAsUtil, ConvUtil, RoundToNearest};
use enum_dispatch::enum_dispatch;
use itertools::Itertools;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Derive Nyquist frequency from time series
///
/// Nyquist frequency for unevenly time series is not well-defined. Here we define it as
/// $\pi / \delta t$, where $\delta t$ is some typical interval between consequent observations
#[enum_dispatch]
trait NyquistFreqTrait: Send + Sync + Clone + Debug {
    fn nyquist_freq<T: Float>(&self, t: &[T]) -> T;
}

#[enum_dispatch(NyquistFreqTrait)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum NyquistFreq {
    Average(AverageNyquistFreq),
    Median(MedianNyquistFreq),
    Quantile(QuantileNyquistFreq),
}

/// $\Delta t = \mathrm{duration} / (N - 1)$ is the mean time interval between observations
///
/// The denominator is $(N-1)$ for compatibility with Nyquist frequency for uniform grid. Note that
/// in literature definition of "average Nyquist" frequency usually differ and place $N$ to the
/// denominator
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Average")]
pub struct AverageNyquistFreq;

impl NyquistFreqTrait for AverageNyquistFreq {
    fn nyquist_freq<T: Float>(&self, t: &[T]) -> T {
        let n = t.len();
        T::PI() * (n - 1).value_as().unwrap() / (t[n - 1] - t[0])
    }
}

fn diff<T: Float>(x: &[T]) -> Vec<T> {
    x.iter().tuple_windows().map(|(&a, &b)| b - a).collect()
}

/// $\Delta t$ is the median time interval between observations
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Median")]
pub struct MedianNyquistFreq;

impl NyquistFreqTrait for MedianNyquistFreq {
    fn nyquist_freq<T: Float>(&self, t: &[T]) -> T {
        let sorted_dt: SortedArray<_> = diff(t).into();
        let dt = sorted_dt.median();
        T::PI() / dt
    }
}

/// $\Delta t$ is the $q$th quantile of time intervals between subsequent observations
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Quantile")]
pub struct QuantileNyquistFreq {
    pub quantile: f32,
}

impl NyquistFreqTrait for QuantileNyquistFreq {
    fn nyquist_freq<T: Float>(&self, t: &[T]) -> T {
        let sorted_dt: SortedArray<_> = diff(t).into();
        let dt = sorted_dt.ppf(self.quantile);
        T::PI() / dt
    }
}

#[derive(Clone, Debug)]
pub struct FreqGrid<T> {
    pub step: T,
    pub size: usize,
}

impl<T> FreqGrid<T>
where
    T: Float,
{
    pub fn from_t(t: &[T], resolution: f32, max_freq_factor: f32, nyquist: NyquistFreq) -> Self {
        assert!(resolution.is_sign_positive() && resolution.is_finite());

        let sizef = t.len().value_as::<T>().unwrap();
        let duration = t[t.len() - 1] - t[0];
        let step = T::two() * T::PI() * (sizef - T::one())
            / (sizef * resolution.value_as::<T>().unwrap() * duration);
        let max_freq = nyquist.nyquist_freq(t) * max_freq_factor.value_as::<T>().unwrap();
        let size = (max_freq / step).approx_by::<RoundToNearest>().unwrap();
        Self { step, size }
    }
}
