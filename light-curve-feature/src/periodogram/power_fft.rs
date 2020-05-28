use crate::float_trait::Float;
use crate::periodogram::fft::*;
use crate::periodogram::freq::FreqGrid;
use crate::periodogram::power::*;
use crate::time_series::TimeSeries;
use conv::{ConvAsUtil, ConvUtil, RoundToNearest};
use num_complex::Complex;

pub struct PeriodogramPowerFft {}

impl PeriodogramPowerFft {
    pub fn init_thread_local_fft_plan<T: Float>(n: usize) {
        T::init_fft_plan(n);
    }
}

impl<T> PeriodogramPower<T> for PeriodogramPowerFft
where
    T: Float,
{
    fn power(&self, freq: &FreqGrid<T>, ts: &mut TimeSeries<T>) -> Vec<T> {
        let spread_freq = FreqGrid {
            step: freq.step,
            size: freq.size.next_power_of_two(),
        };

        let m_std2 = ts.m.get_std().powi(2);

        if m_std2.is_zero() {
            return vec![T::zero(); spread_freq.size];
        }

        let (sum_sin_cos_h, sum_sin_cos_2) = sum_sin_cos(&spread_freq, ts);

        let spread_size = (spread_freq.size << 1).value_as::<T>().unwrap();

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
                    T::half() * (spread_size + sum_cos_2 * cos_wtau + sum_sin_2 * sin_wtau);
                let sum_sin2_wt_tau = spread_size - sum_cos2_wt_tau;

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

fn spread<T: Float>(v: &mut AlignedVec<T>, x: T, y: T) {
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

fn zeroed_aligned_vec<T: Float>(size: usize) -> AlignedVec<T> {
    let mut av = AlignedVec::new(size);
    for x in av.iter_mut() {
        *x = T::zero();
    }
    av
}

fn spread_arrays_for_fft<T: Float>(
    freq: &FreqGrid<T>,
    ts: &mut TimeSeries<T>,
) -> (AlignedVec<T>, AlignedVec<T>) {
    let size = freq.size << 1;

    let mut mh = zeroed_aligned_vec(size);
    let mut m2 = zeroed_aligned_vec(size);

    let spread_dt = T::two() * T::PI() / freq.step / size.value_as::<T>().unwrap();
    let t0 = ts.t.sample[0];
    let m_mean = ts.m.get_mean();

    for (&t, &m) in ts.t.sample.iter().zip(ts.m.sample.iter()) {
        let x = (t - t0) / spread_dt;
        spread(&mut mh, x, m - m_mean);
        let double_x = T::two() * x;
        spread(&mut m2, double_x, T::one());
    }

    (mh, m2)
}

fn sum_sin_cos<T: Float>(
    freq: &FreqGrid<T>,
    ts: &mut TimeSeries<T>,
) -> (AlignedVec<Complex<T>>, AlignedVec<Complex<T>>) {
    let (m_for_sch, m_for_sc2) = spread_arrays_for_fft(freq, ts);
    let sum_sin_cos_h = T::fft(m_for_sch);
    let sum_sin_cos_2 = T::fft(m_for_sc2);
    (sum_sin_cos_h, sum_sin_cos_2)
}
