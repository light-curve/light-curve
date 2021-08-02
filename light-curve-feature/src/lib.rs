//!
//! `light-curve-feature` is a part of [`light-curve`](https://docs.rs/light-curve) family that
//! implements extraction of numerous light curve features used in astrophysics.
//!
//! ```
//! use light_curve_feature::*;
//!
//! // Let's find amplitude and reduced Chi-squared of the light curve
//! let fe = FeatureExtractor::<_, Feature<_>>::new(vec![Amplitude::default().into(), ReducedChi2::default().into()]);
//! // Define light curve
//! let time = [0.0, 1.0, 2.0, 3.0, 4.0];
//! let magn = [-1.0, 2.0, 1.0, 3.0, 4.5];
//! let weights = [5.0, 10.0, 2.0, 10.0, 5.0]; // inverse squared magnitude errors
//! let mut ts = TimeSeries::new(&time, &magn, &weights);
//! // Get results and print
//! let result = fe.eval(&mut ts)?;
//! let names = fe.get_names();
//! println!("{:?}", names.iter().zip(result.iter()).collect::<Vec<_>>());
//! # Ok::<(), EvaluatorError>(())
//! ```

#[cfg(test)]
#[macro_use]
pub mod tests;

#[macro_use]
mod macros;

mod evaluator;
pub use evaluator::FeatureEvaluator;
pub use features::antifeatures;

mod error;
pub use error::EvaluatorError;

mod extractor;
pub use extractor::FeatureExtractor;

pub mod feature;
pub use feature::Feature;

pub mod features;
pub use features::*;

mod float_trait;
pub use float_trait::Float;

mod lnerfc;

#[cfg(feature = "gsl")]
mod nl_fit;

pub mod periodogram;
pub use periodogram::recurrent_sin_cos::RecurrentSinCos;
pub use periodogram::{
    AverageNyquistFreq, MedianNyquistFreq, NyquistFreq, PeriodogramPower, PeriodogramPowerDirect,
    PeriodogramPowerFft, QuantileNyquistFreq,
};

pub mod sorted_array;

pub mod straight_line_fit;

pub mod peak_indices;

pub mod time_series;
pub use time_series::TimeSeries;

mod types;
