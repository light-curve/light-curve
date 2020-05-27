use crate::fft::{AlignedAllocable, AlignedVec, Fft, FftwFloat, Plan, R2CPlan};
use crate::float_trait::Float;
use crate::recurrent_sin_cos::RecurrentSinCos;
use crate::statistics::Statistics;
use crate::time_series::TimeSeries;
use conv::{ConvAsUtil, ConvUtil, RoundToNearest};
use num_complex::Complex;
use std::cell::RefCell;

thread_local! {
    static FFT32: RefCell<Fft<f32>> = RefCell::new(Fft::<f32>::new());
    static FFT64: RefCell<Fft<f64>> = RefCell::new(Fft::<f64>::new());
}

pub trait FloatSupportedByFft: FftwFloat {
    fn fft(x: AlignedVec<Self>) -> AlignedVec<Complex<Self>>;
}

macro_rules! float_supported_by_fft {
    ($float: ty, $fft: expr) => {
        impl FloatSupportedByFft for $float {
            fn fft(x: AlignedVec<Self>) -> AlignedVec<Complex<Self>> {
                $fft.with(|cell| cell.borrow_mut().fft(x).unwrap())
            }
        }
    };
}

float_supported_by_fft!(f32, FFT32);
float_supported_by_fft!(f64, FFT64);

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
/// \max\omega = N_\omega \Delta\omega = \frac{\pi}{\Delta t},
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

        let sizef = t.len().value_as::<T>().unwrap();
        let duration = t[t.len() - 1] - t[0];
        let delta_freq = T::two() * T::PI() * (sizef - T::one())
            / (sizef * resolution.value_as::<T>().unwrap() * duration);
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

    pub fn power_direct(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
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

impl<T> Periodogram<T>
where
    T: Float + FloatSupportedByFft,
    Complex<T>: AlignedAllocable,
    Plan<T, Complex<T>, T::Plan>: R2CPlan<Real = T, Complex = Complex<T>>,
{
    fn spread(v: &mut AlignedVec<T>, x: T, y: T) {
        let x_lo = x.floor();
        let x_hi = x.ceil();
        let i_lo: usize = x_lo.approx_by::<RoundToNearest>().unwrap() % v.len();
        let i_hi: usize = x_hi.approx_by::<RoundToNearest>().unwrap() % v.len();

        if i_lo == i_hi {
            v[i_lo] += y;
            return;
        }

        let alpha = (x - x_lo) / (x_hi - x_lo);
        v[i_lo] = (T::one() - alpha) * y;
        v[i_hi] = alpha * y;
    }

    fn zero_aligned_vec(size: usize) -> AlignedVec<T> {
        let mut av = AlignedVec::new(size);
        for x in av.iter_mut() {
            *x = T::zero();
        }
        av
    }

    fn spread_arrays_for_fft(
        &self,
        ts: &mut TimeSeries<T>,
        size: usize,
    ) -> (AlignedVec<T>, AlignedVec<T>) {
        let mut mh = Self::zero_aligned_vec(size);
        let mut m2 = Self::zero_aligned_vec(size);

        let spread_dt = T::two() * T::PI() / self.delta_freq / size.value_as::<T>().unwrap();
        let t0 = ts.t.sample[0];
        let m_mean = ts.m.get_mean();

        for (&t, &m) in ts.t.sample.iter().zip(ts.m.sample.iter()) {
            let x = (t - t0) / spread_dt;
            Self::spread(&mut mh, x, m - m_mean);
            let double_x = T::two() * x;
            Self::spread(&mut m2, double_x, T::one());
        }

        (mh, m2)
    }

    fn sum_sin_cos_fft(
        &self,
        ts: &mut TimeSeries<T>,
        size: usize,
    ) -> (AlignedVec<Complex<T>>, AlignedVec<Complex<T>>) {
        let (m_for_sch, m_for_sc2) = self.spread_arrays_for_fft(ts, size);
        let sum_sin_cos_h = T::fft(m_for_sch);
        let sum_sin_cos_2 = T::fft(m_for_sc2);
        (sum_sin_cos_h, sum_sin_cos_2)
    }

    pub fn power_fft(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let m_std2 = ts.m.get_std().powi(2);
        let spread_size = self.size.next_power_of_two() << 1;

        if m_std2.is_zero() {
            return vec![T::zero(); spread_size >> 1];
        }

        let (sum_sin_cos_h, sum_sin_cos_2) = self.sum_sin_cos_fft(ts, spread_size);

        let spread_size_f = spread_size.value_as::<T>().unwrap();

        sum_sin_cos_h
            .iter()
            .zip(sum_sin_cos_2.iter())
            .skip(1) // skip zero frequency
            .map(|(sch, sc2)| {
                let sum_cos_h = sch.re;
                let sum_sin_h = -sch.im;
                let sum_cos_2 = sc2.re;
                let sum_sin_2 = -sc2.im;

                let cos_2wtau = if T::is_zero(&sum_cos_2) && T::is_zero(&sum_sin_2) {
                    // Set tau to zero
                    T::one()
                } else {
                    sum_cos_2 / T::hypot(sum_cos_2, sum_sin_2)
                };

                let cos_wtau = T::sqrt(T::half() * (T::one() + cos_2wtau));
                let sin_wtau = T::signum(sum_sin_2) * T::sqrt(T::half() * (T::one() - cos_2wtau));

                let sum_h_cos = sum_cos_h * cos_wtau + sum_sin_h * sin_wtau;
                let sum_h_sin = sum_sin_h * cos_wtau - sum_cos_h * sin_wtau;

                let sum_cos2_wt_tau =
                    T::half() * (spread_size_f + sum_cos_2 * cos_wtau + sum_sin_2 * sin_wtau);
                let sum_sin2_wt_tau = spread_size_f - sum_cos2_wt_tau;

                let frac_cos = if T::is_zero(&sum_cos2_wt_tau) {
                    T::zero()
                } else {
                    sum_h_cos.powi(2) / sum_cos2_wt_tau
                };
                let frac_sin = if T::is_zero(&sum_sin2_wt_tau) {
                    T::zero()
                } else {
                    sum_h_sin.powi(2) / sum_sin2_wt_tau
                };

                let sum_frac = if T::is_zero(&frac_cos) {
                    T::two() * frac_sin
                } else if T::is_zero(&frac_sin) {
                    T::two() * frac_cos
                } else {
                    frac_sin + frac_cos
                };

                T::half() / m_std2 * sum_frac
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
    use rand::prelude::*;

    #[test]
    fn sin() {
        const OMEGA_SIN: f64 = 0.07;
        const N: usize = 100;
        let t = linspace(0.0, 99.0, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA_SIN * x)).collect();
        let mut ts = TimeSeries::new(&t[..], &m[..], None);
        let periodogram = Periodogram::new(OMEGA_SIN, 1);
        all_close(
            &[periodogram.power_direct(&mut ts)[0] * 2.0 / (N as f64 - 1.0)],
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
        let actual = periodogram.power_direct(&mut ts);
        all_close(&actual[..], &desired[..], 1e-6);
    }

    #[test]
    fn zero_aligned_vec() {
        const N: usize = 32;
        let av = Periodogram::<f64>::zero_aligned_vec(N);
        assert_eq!(&av[..], &[0.0; N]);
    }

    #[test]
    fn spread_arrays_for_fft_one_to_one() {
        const N: usize = 32;

        let mut rng = StdRng::seed_from_u64(0);

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<f64> = (0..N).map(|_| rng.gen()).collect();
        let mut ts = TimeSeries::new(&t[..], &m[..], None);

        let nyquist: Box<dyn NyquistFreq<f64>> = Box::new(AverageNyquistFreq);
        let periodogram = Periodogram::from_t(&t, 1.0, &nyquist);
        let (mh, m2) = periodogram.spread_arrays_for_fft(&mut ts, N);

        let desired_mh: Vec<_> = m.iter().map(|&x| x - ts.m.get_mean()).collect();
        all_close(&mh, &desired_mh, 1e-10);

        let desired_m2: Vec<_> = (0..N).map(|i| ((i + 1) % 2 * 2) as f64).collect();
        assert_eq!(&m2[..], &desired_m2[..]);
    }

    #[test]
    fn spread_arrays_for_fft_one_to_one_resolution() {
        const N: usize = 8;
        const RESOLUTION: usize = 4;

        let mut rng = StdRng::seed_from_u64(0);

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<f64> = (0..N).map(|_| rng.gen()).collect();
        let mut ts = TimeSeries::new(&t[..], &m[..], None);

        let nyquist: Box<dyn NyquistFreq<f64>> = Box::new(AverageNyquistFreq);
        let periodogram = Periodogram::from_t(&t, RESOLUTION as f32, &nyquist);
        let (mh, m2) = periodogram.spread_arrays_for_fft(&mut ts, RESOLUTION * N);

        let desired_mh: Vec<_> = m.iter().map(|&x| x - ts.m.get_mean()).collect();
        all_close(&mh[..N], &desired_mh, 1e-10);
        assert_eq!(&mh[N..], &[0.0; (RESOLUTION - 1) * N]);

        let desired_m2: Vec<_> = (0..2 * N).map(|i| ((i + 1) % 2) as f64).collect();
        assert_eq!(&m2[..2 * N], &desired_m2[..]);
        assert_eq!(&m2[2 * N..], &[0.0; (RESOLUTION - 2) * N]);
    }

    #[test]
    fn direct_vs_fft_one_to_one() {
        const OMEGA: f64 = 0.472;
        const N: usize = 64;
        const RESOLUTION: f32 = 1.0;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA * x)).collect();
        let mut ts = TimeSeries::new(&t[..], &m[..], None);
        let nyquist: Box<dyn NyquistFreq<f64>> = Box::new(AverageNyquistFreq);
        let periodogram = Periodogram::from_t(&t, RESOLUTION, &nyquist);

        let direct = periodogram.power_direct(&mut ts);
        let fft = periodogram.power_fft(&mut ts);
        all_close(
            &fft[..periodogram.size - 1],
            &direct[..periodogram.size - 1],
            1e-8,
        );
    }

    #[test]
    fn direct_vs_fft_uniform_sin_cos() {
        const OMEGA1: f64 = 0.472;
        const OMEGA2: f64 = 1.222;
        const AMPLITUDE2: f64 = 2.0;
        const N: usize = 100;
        const RESOLUTION: f32 = 4.0;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t
            .iter()
            .map(|&x| f64::sin(OMEGA1 * x) + AMPLITUDE2 * f64::cos(OMEGA2 * x))
            .collect();
        let mut ts = TimeSeries::new(&t[..], &m[..], None);
        let nyquist: Box<dyn NyquistFreq<f64>> = Box::new(AverageNyquistFreq);
        let periodogram = Periodogram::from_t(&t, RESOLUTION, &nyquist);

        let direct = periodogram.power_direct(&mut ts);
        let fft = periodogram.power_fft(&mut ts);

        assert_eq!(
            &fft.peak_indices_reverse_sorted()[..2],
            &direct.peak_indices_reverse_sorted()[..2]
        );
    }

    #[test]
    fn direct_vs_fft_unevenly_sin_cos() {
        const OMEGA1: f64 = 0.472;
        const OMEGA2: f64 = 1.222;
        const AMPLITUDE2: f64 = 2.0;
        const NOISE_AMPLITUDE: f64 = 1.0;
        const N: usize = 100;
        const RESOLUTION: f32 = 6.0;

        let mut rng = StdRng::seed_from_u64(0);
        let t = (0..N)
            .map(|_| rng.gen::<f64>() * (N - 1) as f64)
            .collect::<Vec<_>>()
            .sorted();
        let m: Vec<_> = t
            .iter()
            .map(|&x| {
                f64::sin(OMEGA1 * x)
                    + AMPLITUDE2 * f64::cos(OMEGA2 * x)
                    + NOISE_AMPLITUDE * rng.gen::<f64>()
            })
            .collect();
        let mut ts = TimeSeries::new(&t[..], &m[..], None);
        let nyquist: Box<dyn NyquistFreq<f64>> = Box::new(MedianNyquistFreq);
        let periodogram = Periodogram::from_t(&t, RESOLUTION, &nyquist);

        let direct = periodogram.power_direct(&mut ts);
        let fft = periodogram.power_fft(&mut ts);

        assert_eq!(
            &fft.peak_indices_reverse_sorted()[..2],
            &direct.peak_indices_reverse_sorted()[..2]
        );
    }
}
