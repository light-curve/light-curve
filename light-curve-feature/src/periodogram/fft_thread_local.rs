use crate::periodogram::fft::{AlignedVec, Fft, FftwFloat};
use num_complex::Complex;
use std::cell::RefCell;

thread_local! {
    static FFT32: RefCell<Fft<f32>> = RefCell::new(Fft::<f32>::new());
    static FFT64: RefCell<Fft<f64>> = RefCell::new(Fft::<f64>::new());
}

pub trait FloatSupportedByFft: FftwFloat {
    fn fft(x: AlignedVec<Self>) -> AlignedVec<Complex<Self>>;
    fn init_fft_plan(n: usize);
}

macro_rules! float_supported_by_fft {
    ($float: ty, $fft: expr) => {
        impl FloatSupportedByFft for $float {
            fn fft(x: AlignedVec<Self>) -> AlignedVec<Complex<Self>> {
                $fft.with(|cell| cell.borrow_mut().fft(x).unwrap())
            }

            fn init_fft_plan(n: usize) {
                $fft.with(|cell| {
                    cell.borrow_mut().get_plan(n);
                })
            }
        }
    };
}

float_supported_by_fft!(f32, FFT32);
float_supported_by_fft!(f64, FFT64);
