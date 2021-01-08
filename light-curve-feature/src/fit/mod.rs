#[cfg(feature = "gsl")]
mod curve_fit;

mod nls;

mod straight_line;
pub use straight_line::{fit_straight_line, StraightLineFitterResult};
