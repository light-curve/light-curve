use crate::float_trait::Float;
use crate::statistics::Statistics;
use conv::ConvUtil;

/// Derive Nyquist frequency from time series
///
/// Nyquist frequency for unevenly time series is not well-defined. Here we define it as
/// $\pi / \delta t$, where $\delta t$ is some typical interval between consequent observations
pub trait NyquistFreq<T>: Send + Sync {
    fn nyquist_freq(&self, t: &[T]) -> T;
}

/// $\Delta t = \mathrm{duration} / (N - 1)$ is the mean time interval between observations,
/// the denominator is $(N-1)$ for compatibility with Nyquist frequency for uniform grid. Note that
/// in literature definition of "average Nyquist" frequency usually differ and place $N$ to the
/// denominator
pub struct AverageNyquistFreq;

impl<T: Float> NyquistFreq<T> for AverageNyquistFreq {
    fn nyquist_freq(&self, t: &[T]) -> T {
        let n = t.len();
        T::PI() * (n - 1).value_as().unwrap() / (t[n - 1] - t[0])
    }
}

fn diff<T: Float>(x: &[T]) -> Vec<T> {
    (1..x.len()).map(|i| x[i] - x[i - 1]).collect()
}

/// $\Delta t$ is the median time interval between observations
pub struct MedianNyquistFreq;

impl<T: Float> NyquistFreq<T> for MedianNyquistFreq {
    fn nyquist_freq(&self, t: &[T]) -> T {
        let dt = diff(t).median();
        T::PI() / dt
    }
}

pub struct QuantileNyquistFreq {
    pub quantile: f32,
}

impl<T: Float> NyquistFreq<T> for QuantileNyquistFreq {
    fn nyquist_freq(&self, t: &[T]) -> T {
        let dt = diff(t).ppf(self.quantile);
        T::PI() / dt
    }
}

pub struct FreqGrid<T> {
    pub step: T,
    pub size: usize,
}
