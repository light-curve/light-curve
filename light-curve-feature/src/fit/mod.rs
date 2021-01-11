#[cfg(feature = "gsl")]
mod curve_fit;
#[cfg(feature = "gsl")]
pub use curve_fit::{curve_fit, CurveFitResult};

#[cfg(feature = "gsl")]
pub mod data;

#[cfg(feature = "gsl")]
mod nls;

mod straight_line;
pub use straight_line::{fit_straight_line, StraightLineFitterResult};
