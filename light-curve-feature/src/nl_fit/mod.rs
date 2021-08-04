pub mod curve_fit;
pub use curve_fit::{CurveFitAlgorithm, CurveFitResult, CurveFitTrait};

pub mod data;

#[cfg(feature = "gsl")]
pub mod lmsder;
#[cfg(feature = "gsl")]
pub use lmsder::LmsderCurveFit;

pub mod mcmc;
pub use mcmc::McmcCurveFit;

#[cfg(test)]
pub trait HyperdualFloat: hyperdual::Float {}
#[cfg(test)]
impl<T> HyperdualFloat for T where T: hyperdual::Float {}
#[cfg(not(test))]
pub trait HyperdualFloat: crate::Float {}
#[cfg(not(test))]
impl<T> HyperdualFloat for T where T: crate::Float {}

pub trait F64LikeFloat:
    HyperdualFloat + std::ops::AddAssign<Self> + std::ops::MulAssign<Self>
{
}

impl<T> F64LikeFloat for T where
    T: HyperdualFloat + std::ops::AddAssign<Self> + std::ops::MulAssign<Self>
{
}
