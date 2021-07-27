pub mod curve_fit;
pub use curve_fit::{CurveFitAlgorithm, CurveFitResult, CurveFitTrait};

pub mod data;

pub mod lmsder;
pub use lmsder::LmsderCurveFit;
