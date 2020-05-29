pub use fftw::array::{AlignedAllocable, AlignedVec};
use fftw::error::Result;
pub use fftw::plan::{Plan, R2CPlan};
use fftw::plan::{Plan32, Plan64, PlanSpec};
use fftw::types::Flag;
use num_complex::Complex;
use std::collections::HashMap;

pub trait FftwFloat: AlignedAllocable {
    type Plan: PlanSpec;
}

impl FftwFloat for f32 {
    type Plan = Plan32;
}

impl FftwFloat for f64 {
    type Plan = Plan64;
}

pub struct Fft<T>
where
    T: FftwFloat,
{
    plans: HashMap<usize, Plan<T, Complex<T>, T::Plan>>,
    flags: Flag,
}

impl<T> Fft<T>
where
    T: FftwFloat,
    Complex<T>: AlignedAllocable,
    Plan<T, Complex<T>, T::Plan>: R2CPlan<Real = T, Complex = Complex<T>>,
{
    pub fn new() -> Self {
        let mut flags = Flag::Measure;
        flags.insert(Flag::DestroyInput);
        Self {
            plans: HashMap::new(),
            flags,
        }
    }

    pub fn get_plan(&mut self, n: usize) -> &mut Plan<T, Complex<T>, T::Plan> {
        let flags = self.flags;
        self.plans
            .entry(n)
            .or_insert_with(|| R2CPlan::aligned(&[n], flags).unwrap())
    }

    pub fn fft(&mut self, mut x: AlignedVec<T>) -> Result<AlignedVec<Complex<T>>> {
        let n = x.len();
        let mut y = AlignedVec::new(n / 2 + 1);
        self.get_plan(n).r2c(&mut x, &mut y)?;
        Ok(y)
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use super::*;
    use light_curve_common::all_close;
    use std::f64::consts::PI;

    #[test]
    #[allow(clippy::float_cmp)]
    fn unity() {
        const N: usize = 1024;

        let mut x = AlignedVec::new(N);
        for a in x.iter_mut() {
            *a = 1.0;
        }

        let mut fft = Fft::new();
        let y = fft.fft(x).unwrap();

        assert_eq!(y[0].re, 1024.0_f64);
        assert_eq!(y[0].im, 0.0_f64);

        let (re, im): (Vec<_>, Vec<_>) = y.iter().map(|c| (c.re, c.im)).unzip();
        all_close(&re[1..], &[0.0; 512], 1e-12);
        all_close(&im[1..], &[0.0; 512], 1e-12);
    }

    #[test]
    fn numpy_compr() {
        const N: usize = 32;

        let mut x = AlignedVec::new(N);
        for i in 0..N {
            x[i] = f64::sin(2.0 * PI * 0.27 * (i as f64));
        }

        let mut fft = Fft::new();
        let y = fft.fft(x).unwrap();

        let (actual_re, actual_im): (Vec<_>, Vec<_>) = y.iter().map(|c| (c.re, c.im)).unzip();

        // np.fft.fft(np.sin(2 * np.pi * 0.27 * np.arange(32)))[:17]
        let desired_re = [
            1.10704834,
            1.1195868,
            1.15941438,
            1.23418408,
            1.36101006,
            1.57816611,
            1.98413371,
            2.92020198,
            6.86602938,
            -11.2588103,
            -2.77097256,
            -1.50267075,
            -1.01091584,
            -0.76502572,
            -0.63191196,
            -0.56424862,
            -0.54338985,
        ];
        let desired_im = [
            0.,
            0.06794917,
            0.14051614,
            0.22370033,
            0.32725189,
            0.47044727,
            0.70062801,
            1.17923298,
            3.07385847,
            -5.4167115,
            -1.38305977,
            -0.7445412,
            -0.46825362,
            -0.30311016,
            -0.18462464,
            -0.08785979,
            0.,
        ];

        all_close(&actual_re, &desired_re, 1e-8);
        all_close(&actual_im, &desired_im, 1e-8);
    }
}
