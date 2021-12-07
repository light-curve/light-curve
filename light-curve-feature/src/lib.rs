#![doc = include_str!("../README.md")]

#[cfg(test)]
#[macro_use]
mod tests;

#[macro_use]
mod macros;

mod evaluator;
pub use evaluator::{EvaluatorInfoTrait, FeatureEvaluator, FeatureNamesDescriptionsTrait};

mod error;
pub use error::EvaluatorError;

mod extractor;
pub use extractor::FeatureExtractor;

mod feature;
pub use feature::Feature;

pub mod features;
pub use features::antifeatures;
pub use features::*;

mod float_trait;
pub use float_trait::Float;

mod lnerfc;

mod nl_fit;
#[cfg(feature = "gsl")]
pub use nl_fit::LmsderCurveFit;
pub use nl_fit::{CurveFitAlgorithm, McmcCurveFit};

#[doc(hidden)]
pub mod periodogram;
pub use periodogram::recurrent_sin_cos::RecurrentSinCos;
pub use periodogram::{
    AverageNyquistFreq, MedianNyquistFreq, NyquistFreq, PeriodogramPower, PeriodogramPowerDirect,
    PeriodogramPowerFft, QuantileNyquistFreq,
};

pub mod prelude;

mod sorted_array;

mod straight_line_fit;
#[doc(hidden)]
pub use straight_line_fit::fit_straight_line;

mod peak_indices;
#[doc(hidden)]
pub use peak_indices::peak_indices;

mod time_series;
pub use time_series::{DataSample, TimeSeries};

mod types;
