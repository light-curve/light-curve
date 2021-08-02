pub mod curve_fit;
pub use curve_fit::{CurveFitAlgorithm, CurveFitResult, CurveFitTrait};

pub mod data;

#[cfg(feature = "gsl")]
pub mod lmsder;
#[cfg(feature = "gsl")]
pub use lmsder::LmsderCurveFit;

pub mod mcmc;
pub use mcmc::McmcCurveFit;
