use crate::float_trait::Float;
use crate::time_series::TimeSeries;
use conv::{ConvAsUtil, ConvUtil, RoundToNearest};

mod fft;
use fft::{AlignedAllocable, AlignedVec, Fft, FftwFloat, Plan, R2CPlan};

mod fft_thread_local;
pub use fft_thread_local::FloatSupportedByFft;

mod freq;
use freq::FreqGrid;
pub use freq::{AverageNyquistFreq, MedianNyquistFreq, NyquistFreq, QuantileNyquistFreq};

mod power;
use power::PeriodogramPower;

mod power_fft;
use power_fft::PeriodogramPowerFft;

mod power_direct;
use power_direct::PeriodogramPowerDirect;

mod recurrent_sin_cos;

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
    freq_grid: FreqGrid<T>,
    periodogram_power: Box<dyn PeriodogramPower<T>>,
}

impl<T> Periodogram<T>
where
    T: Float,
{
    pub fn new(delta_freq: T, size: usize) -> Self {
        assert!(delta_freq.is_sign_positive() && delta_freq.is_finite());
        assert!(size > 0);
        let freq_grid = FreqGrid {
            step: delta_freq,
            size,
        };

        static FFT_MIN_SIZE: usize = 1 << 7;
        let periodogram_power: Box<dyn PeriodogramPower<T>> = if size < FFT_MIN_SIZE {
            Box::new(PeriodogramPowerDirect {})
        } else {
            Box::new(PeriodogramPowerFft {})
        };

        Self {
            freq_grid,
            periodogram_power,
        }
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

    pub fn freq(&self, i: usize) -> T {
        self.freq_grid.step * (i + 1).value_as::<T>().unwrap()
    }

    pub fn power(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        self.periodogram_power.power(&self.freq_grid, ts)
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
        let av = Periodogram::<f64>::zeroed_aligned_vec(N);
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
