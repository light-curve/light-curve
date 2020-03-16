use crate::float_trait::Float;
use crate::recurrent_sin_cos::RecurrentSinCos;
use crate::statistics::Statistics;
use crate::time_series::TimeSeries;
use conv::{ConvAsUtil, ConvUtil, RoundToNearest};

#[derive(Clone)]
struct PeriodogramSums<T> {
    m_sin: T,
    m_cos: T,
    sin2: T,
}

impl<T: Float> Default for PeriodogramSums<T> {
    fn default() -> Self {
        Self {
            m_sin: T::zero(),
            m_cos: T::zero(),
            sin2: T::zero(),
        }
    }
}

struct SinCosOmegaTau<T> {
    sin_cos_2omega_x: Vec<RecurrentSinCos<T>>,
}

impl<T: Float> SinCosOmegaTau<T> {
    fn new(freq0: T, t: &[T]) -> Self {
        let sin_cos_2omega_x = t
            .iter()
            .map(|&x| RecurrentSinCos::new(T::two() * freq0 * x))
            .collect();
        Self { sin_cos_2omega_x }
    }
}

impl<T: Float> Iterator for SinCosOmegaTau<T> {
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        let mut sum_sin = T::zero();
        let mut sum_cos = T::zero();
        for s_c in self.sin_cos_2omega_x.iter_mut() {
            let (sin, cos) = s_c.next().unwrap();
            sum_sin += sin;
            sum_cos += cos;
        }
        let cos2 = sum_cos / T::hypot(sum_sin, sum_cos);
        let sin = T::signum(sum_sin) * T::sqrt(T::half() * (T::one() - cos2));
        let cos = T::sqrt(T::half() * (T::one() + cos2));
        Some((sin, cos))
    }
}

pub trait NyquistFreq<T>: Send + Sync {
    fn nyquist_freq(&self, t: &[T]) -> T;
}

/// $\Delta\t = \mathrm{duration} / N$ is the mean time interval between observations,
/// denominator is not $(N-1)$ according to literature definition of "average Nyquist" frequency
#[derive(Clone)]
pub struct AverageNyquistFreq;

impl<T: Float> NyquistFreq<T> for AverageNyquistFreq {
    fn nyquist_freq(&self, t: &[T]) -> T {
        let n = t.len();
        T::PI() * n.value_as().unwrap() / (t[n - 1] - t[0])
    }
}

fn diff<T: Float>(x: &[T]) -> Vec<T> {
    (1..x.len()).map(|i| x[i] - x[i - 1]).collect()
}

/// $\Delta t$ is the median time interval between observations
#[derive(Clone)]
pub struct MedianNyquistFreq;

impl<T: Float> NyquistFreq<T> for MedianNyquistFreq {
    fn nyquist_freq(&self, t: &[T]) -> T {
        let dt = diff(t).median();
        T::PI() / dt
    }
}

#[derive(Clone)]
pub struct QuantileNyquistFreq {
    pub quantile: f32,
}

impl<T: Float> NyquistFreq<T> for QuantileNyquistFreq {
    fn nyquist_freq(&self, t: &[T]) -> T {
        let dt = diff(t).ppf(self.quantile);
        T::PI() / dt
    }
}

/// Lamb-Scargle periodogram calculator on uniform frequency grid
///
/// Frequencies are given by $\omega = \{\min\omega..\max\omega\}$: $N_\omega$ nodes with step
/// $\Delta\omega$: $\min\omega = \Delta\omega$, $\max\omega = N_\omega \Delta\omega$.
///
/// Parameters of the grid can be derived from time series properties: typical time interval
/// $\delta t$ and duration of observation. The idea is to set maximum frequency to Nyquist
/// value $\pi / \Delta t$ and minimum frequency to $2\pi / \mathrm{duration}$, while `nyquist` and
/// `resolution` factors are used to widen this interval:
/// $$
/// \max\omega = N_\omega \Delta\omega = \pi \frac{\mathrm{nyquist}}{\Delta t},
/// $$
/// $$
/// \min\omega = \Delta\omega = \frac{2\pi}{\mathrm{resolution} \times \mathrm{duration}}.
/// $$
pub struct Periodogram<T> {
    delta_freq: T,
    size: usize,
}

impl<T> Periodogram<T>
where
    T: Float,
{
    pub fn new(delta_freq: T, size: usize) -> Self {
        assert!(delta_freq.is_sign_positive() && delta_freq.is_finite());
        assert!(size > 0);
        Self { delta_freq, size }
    }

    #[allow(clippy::borrowed_box)] // https://github.com/rust-lang/rust-clippy/issues/4305
    pub fn from_t(t: &[T], resolution: f32, nyquist: &Box<dyn NyquistFreq<T>>) -> Self {
        assert!(resolution.is_sign_positive() && resolution.is_finite());

        let duration = t[t.len() - 1] - t[0];
        let delta_freq = T::two() * T::PI() / (resolution.value_as::<T>().unwrap() * duration);
        let max_freq = nyquist.nyquist_freq(t);
        let size = (max_freq / delta_freq)
            .approx_by::<RoundToNearest>()
            .unwrap();
        Self::new(delta_freq, size)
    }

    pub fn freq(&self) -> Vec<T> {
        (1..=self.size)
            .map(|i| self.delta_freq * i.value_as::<T>().unwrap())
            .collect()
    }

    pub fn power(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let m_mean = ts.m.get_mean();

        let sin_cos_omega_tau = SinCosOmegaTau::new(self.delta_freq, ts.t.sample);
        let mut sin_cos_omega_x: Vec<_> =
            ts.t.sample
                .iter()
                .map(|&x| RecurrentSinCos::new(self.delta_freq * x))
                .collect();

        sin_cos_omega_tau
            .take(self.size)
            .map(|(sin_omega_tau, cos_omega_tau)| {
                let mut sum_m_sin = T::zero();
                let mut sum_m_cos = T::zero();
                let mut sum_sin2 = T::zero();
                for (s_c_omega_x, &y) in sin_cos_omega_x.iter_mut().zip(ts.m.sample.iter()) {
                    let (sin_omega_x, cos_omega_x) = s_c_omega_x.next().unwrap();
                    // sini and cosine of omega * (x - tau)
                    let sin = sin_omega_x * cos_omega_tau - cos_omega_x * sin_omega_tau;
                    let cos = cos_omega_x * cos_omega_tau + sin_omega_x * sin_omega_tau;
                    sum_m_sin += (y - m_mean) * sin;
                    sum_m_cos += (y - m_mean) * cos;
                    sum_sin2 += sin.powi(2);
                }
                let sum_cos2 = ts.lenf() - sum_sin2;

                if (sum_m_sin.is_zero() & sum_sin2.is_zero())
                    | (sum_m_cos.is_zero() & sum_cos2.is_zero())
                    | ts.m.get_std().is_zero()
                {
                    T::zero()
                } else {
                    T::half() * (sum_m_sin.powi(2) / sum_sin2 + sum_m_cos.powi(2) / sum_cos2)
                        / ts.m.get_std().powi(2)
                }
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use light_curve_common::{all_close, linspace};

    #[test]
    fn test_sin() {
        const OMEGA_SIN: f64 = 0.07;
        const N: usize = 100;
        let t = linspace(0.0, 99.0, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA_SIN * x)).collect();
        let mut ts = TimeSeries::new(&t[..], &m[..], None);
        let periodogram = Periodogram::new(OMEGA_SIN, 1);
        all_close(
            &[periodogram.power(&mut ts)[0] * 2.0 / (N as f64 - 1.0)],
            &[1.0],
            1.0 / (N as f64),
        );

        // import numpy as np
        // from scipy.signal import lombscargle
        //
        // t = np.arange(100)
        // m = np.sin(0.07 * t)
        // y = (m - m.mean()) / m.std(ddof=1)
        // freq = np.linspace(0.01, 0.05, 5)
        // print(lombscargle(t, y, freq, precenter=True, normalize=False))

        let periodogram = Periodogram::new(0.01, 5);
        all_close(&linspace(0.01, 0.05, 5), &periodogram.freq(), 1e-12);
        let desired = [
            16.99018018,
            18.57722516,
            21.96049738,
            28.15056806,
            36.66519435,
        ];
        let actual = periodogram.power(&mut ts);
        all_close(&actual[..], &desired[..], 1e-6);
    }
}
