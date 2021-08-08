pub use fftw::array::{AlignedAllocable, AlignedVec};
use fftw::error::Result;
pub use fftw::plan::{Plan, Plan32, Plan64, R2CPlan};
use fftw::types::Flag;
use num_complex::Complex;
use std::collections::HashMap;
use std::fmt;

/// Complex number trait for `fftw` crate
pub trait FftwComplex<T>: AlignedAllocable + Send
where
    T: AlignedAllocable + Send,
{
    /// Real part
    fn get_re(&self) -> T;

    /// Imaginary part
    fn get_im(&self) -> T;
}

impl FftwComplex<f32> for Complex<f32> {
    #[inline]
    fn get_re(&self) -> f32 {
        self.re
    }

    #[inline]
    fn get_im(&self) -> f32 {
        self.im
    }
}

impl FftwComplex<f64> for Complex<f64> {
    #[inline]
    fn get_re(&self) -> f64 {
        self.re
    }

    #[inline]
    fn get_im(&self) -> f64 {
        self.im
    }
}

/// Floating number trait for `fftw` crate
pub trait FftwFloat: AlignedAllocable + Send {
    type FftwComplex: FftwComplex<Self>;
    type FftwPlan: R2CPlan<Real = Self, Complex = Self::FftwComplex> + Send;
}

impl FftwFloat for f32 {
    type FftwComplex = Complex<f32>;
    type FftwPlan = Plan<f32, Complex<f32>, Plan32>;
}

impl FftwFloat for f64 {
    type FftwComplex = Complex<f64>;
    type FftwPlan = Plan<f64, Complex<f64>, Plan64>;
}

pub struct Fft<T>
where
    T: FftwFloat,
{
    plans: HashMap<usize, T::FftwPlan>,
}

impl<T> fmt::Debug for Fft<T>
where
    T: FftwFloat,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}<{}>",
            std::any::type_name::<Self>(),
            std::any::type_name::<T>()
        )
    }
}

impl<T> Fft<T>
where
    T: FftwFloat,
{
    pub fn new() -> Self {
        Self {
            plans: HashMap::new(),
        }
    }

    fn flags(n: usize) -> Flag {
        const MAX_N_TO_MEASURE: usize = 1 << 12; // It takes ~3s to measure
        let mut flag = Flag::DESTROYINPUT;
        if n <= MAX_N_TO_MEASURE {
            flag.insert(Flag::MEASURE);
        } else {
            flag.insert(Flag::ESTIMATE);
        }
        flag
    }

    pub fn get_plan(&mut self, n: usize) -> &mut T::FftwPlan {
        self.plans
            .entry(n)
            .or_insert_with(|| R2CPlan::aligned(&[n], Self::flags(n)).unwrap())
    }

    pub fn fft(&mut self, x: &mut AlignedVec<T>, y: &mut AlignedVec<T::FftwComplex>) -> Result<()> {
        let n = x.len();
        self.get_plan(n).r2c(x, y)?;
        Ok(())
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

        let mut y: AlignedVec<Complex<f64>> = AlignedVec::new(N / 2 + 1);
        fft.fft(&mut x, &mut y).unwrap();

        assert_eq!(y[0].get_re(), 1024.0_f64);
        assert_eq!(y[0].get_im(), 0.0_f64);

        let (re, im): (Vec<_>, Vec<_>) = y.iter().map(|c| (c.get_re(), c.get_im())).unzip();
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
        let mut y = AlignedVec::new(N / 2 + 1);
        fft.fft(&mut x, &mut y).unwrap();

        let (actual_re, actual_im): (Vec<_>, Vec<_>) =
            y.iter().map(|c| (c.get_re(), c.get_im())).unzip();

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
