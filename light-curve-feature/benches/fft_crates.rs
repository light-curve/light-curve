use chfft::RFft1D;
use core::fmt;
use core::fmt::Debug;
use criterion::{black_box, Criterion};
use fftw::array::{AlignedAllocable, AlignedVec};
use fftw::plan::{Plan, Plan32, Plan64, PlanSpec, R2CPlan};
use fftw::types::Flag;
use num_traits::{Float, FloatConst, NumAssign};
use rand::prelude::*;
use rustfft::algorithm::Radix4;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{FFTnum, FFT};
use std::{any, collections::HashMap};

trait Fft<T>: Debug {
    fn run(&mut self, a: &[T]) -> (T, T);
}

struct RustFft<T> {
    radix: HashMap<usize, Radix4<T>>,
}

impl<T> RustFft<T>
where
    T: FFTnum,
{
    fn new(n: &[usize]) -> Self {
        Self {
            radix: n.iter().map(|&i| (i, Radix4::new(i, false))).collect(),
        }
    }
}

impl<T> Debug for RustFft<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("rustfft").finish()
    }
}

impl<T> Fft<T> for RustFft<T>
where
    T: FFTnum,
{
    fn run(&mut self, a: &[T]) -> (T, T) {
        let mut t: Vec<_> = a
            .iter()
            .map(|&x| Complex {
                re: x,
                im: T::zero(),
            })
            .collect();
        let mut f = vec![Complex::zero(); a.len()];
        self.radix.get(&a.len()).unwrap().process(&mut t, &mut f);
        // let norm = T::one() / T::sqrt(f.len());
        // for y in f.mut_iter() {
        //     *y *= norm;
        // }
        f.iter()
            .take(a.len())
            .fold((T::zero(), T::zero()), |acc, c| {
                (acc.0 + c.re, acc.1 + c.im)
            })
    }
}

struct ChFft<T> {
    rfft1d: HashMap<usize, RFft1D<T>>,
}

impl<T> ChFft<T>
where
    T: Float + FloatConst + NumAssign,
{
    fn new(n: &[usize]) -> Self {
        Self {
            rfft1d: n.iter().map(|&i| (i, RFft1D::new(i))).collect(),
        }
    }
}

impl<T> Debug for ChFft<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("chfft").finish()
    }
}

impl<T> Fft<T> for ChFft<T>
where
    T: Float + FloatConst + NumAssign,
{
    fn run(&mut self, a: &[T]) -> (T, T) {
        let t: Vec<_> = a.to_vec();
        let f = self.rfft1d.get_mut(&a.len()).unwrap().forward(&t[..]);
        f.iter().fold((T::zero(), T::zero()), |acc, c| {
            (acc.0 + c.re, acc.1 + c.im)
        })
    }
}

pub trait FloatSupportedByFftwPlan {
    type Plan: PlanSpec;
}

impl FloatSupportedByFftwPlan for f32 {
    type Plan = Plan32;
}

impl FloatSupportedByFftwPlan for f64 {
    type Plan = Plan64;
}

struct Fftw<T>
where
    T: FloatSupportedByFftwPlan,
{
    r2cplan: HashMap<usize, Plan<T, Complex<T>, T::Plan>>,
}

impl<T> Fftw<T>
where
    T: FloatSupportedByFftwPlan + AlignedAllocable,
    Complex<T>: AlignedAllocable,
    Plan<T, Complex<T>, T::Plan>: R2CPlan<Real = T, Complex = Complex<T>>,
{
    fn new(n: &[usize]) -> Self {
        let mut flags = Flag::PATIENT;
        flags.insert(Flag::DESTROYINPUT);
        Self {
            r2cplan: n
                .iter()
                .map(|&i| (i, R2CPlan::aligned(&[i], flags).unwrap()))
                .collect(),
        }
    }
}

impl<T> Debug for Fftw<T>
where
    T: FloatSupportedByFftwPlan,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("fftw").finish()
    }
}

impl<T> Fft<T> for Fftw<T>
where
    T: FloatSupportedByFftwPlan + AlignedAllocable,
    Complex<T>: AlignedAllocable,
    Plan<T, Complex<T>, T::Plan>: R2CPlan<Real = T, Complex = Complex<T>>,
{
    fn run(&mut self, a: &[T]) -> (T, T) {
        let mut t = AlignedVec::new(a.len());
        for i in 0..a.len() {
            t[i] = a[i];
        }
        let mut f = AlignedVec::new(a.len() / 2 + 1);
        self.r2cplan
            .get_mut(&a.len())
            .unwrap()
            .r2c(&mut t, &mut f)
            .unwrap();
        f.iter().fold((T::zero(), T::zero()), |acc, c| {
            (acc.0 + c.re, acc.1 + c.im)
        })
    }
}

trait Series<T>: Debug {
    fn series(&self, n: usize) -> Vec<T>;
}

#[derive(Debug)]
struct Ones {}

impl<T> Series<T> for Ones
where
    T: FFTnum,
{
    fn series(&self, n: usize) -> Vec<T> {
        vec![T::one(); n]
    }
}

#[derive(Debug)]
struct Randoms {}

impl<T> Series<T> for Randoms
where
    T: FFTnum,
    rand::distributions::Standard: Distribution<T>,
{
    fn series(&self, n: usize) -> Vec<T> {
        let mut rng = StdRng::seed_from_u64(0);
        (0..n).map(|_| rng.gen::<T>()).collect()
    }
}

pub fn bench_fft<T>(c: &mut Criterion)
where
    T: fmt::Display
        + FFTnum
        + Float
        + FloatConst
        + NumAssign
        + FloatSupportedByFftwPlan
        + AlignedAllocable,
    Complex<T>: AlignedAllocable,
    Plan<T, Complex<T>, T::Plan>: R2CPlan<Real = T, Complex = Complex<T>>,
    Vec<T>: fmt::Debug,
    rand::distributions::Standard: Distribution<T>,
{
    let counts: Vec<_> = (8..=20).step_by(4).map(|i| 1_usize << i).collect();
    let series: Vec<Box<dyn Series<T>>> = vec![Box::new(Ones {}), Box::new(Randoms {})];
    let mut ffts: Vec<Box<dyn Fft<T>>> = vec![
        Box::new(RustFft::new(&counts)),
        Box::new(ChFft::new(&counts)),
        Box::new(Fftw::new(&counts)),
    ];

    for &n in counts.iter() {
        for s in series.iter() {
            let x = s.series(n);
            for fft in ffts.iter_mut() {
                c.bench_function(
                    format!("FFT - {:?}, {:?}[{}; {}]", fft, s, n, any::type_name::<T>()).as_str(),
                    |b| b.iter(|| fft.run(black_box(&x))),
                );
                // let res = fft.run(&x);
                // println!(
                //     "FFT - {:?}, {:?}[{}; {}] = ({}, {})",
                //     fft,
                //     s,
                //     n,
                //     T::type_name(),
                //     res.0,
                //     res.1
                // );
                // println!("{:?}", x);
            }
        }
    }
}
