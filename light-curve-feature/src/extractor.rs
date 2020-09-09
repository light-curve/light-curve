use crate::evaluator::{FeatureEvaluator, VecFE};
use crate::float_trait::Float;
use crate::time_series::TimeSeries;

/// Constructs a `FeatureExtractor` object from a list of objects that implement `FeatureEvaluator`
/// ```
/// use light_curve_feature::*;
///
/// let fe = feat_extr!(BeyondNStd::new(1.0), Cusum::default());
/// ```
#[macro_export]
macro_rules! feat_extr{
    ( $( $x: expr ),* $(,)? ) => {
        FeatureExtractor::new(
            vec![$(
                Box::new($x),
            )*]
        )
    }
}

/// The engine that extracts features one by one
#[derive(Clone)]
pub struct FeatureExtractor<T: Float> {
    features: VecFE<T>,
}

impl<T> FeatureExtractor<T>
where
    T: Float,
{
    pub fn new(features: VecFE<T>) -> Self {
        Self { features }
    }

    /// Copy of the feature vector
    pub fn clone_features(&self) -> VecFE<T> {
        self.features.clone()
    }

    pub fn add_feature(&mut self, feature: Box<dyn FeatureEvaluator<T>>) {
        self.features.push(feature);
    }
}

impl<T> FeatureEvaluator<T> for FeatureExtractor<T>
where
    T: Float,
{
    /// Get a vector of computed features.
    /// The length of the returned vector is guaranteed to be the same as returned by `get_names()`
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        self.features.iter().flat_map(|x| x.eval(ts)).collect()
    }

    /// Get a vector of feature names.
    /// The length of the returned vector is guaranteed to be the same as returned by `eval()`
    fn get_names(&self) -> Vec<&str> {
        self.features.iter().flat_map(|x| x.get_names()).collect()
    }

    /// Total number of features
    fn size_hint(&self) -> usize {
        self.features.iter().map(|x| x.size_hint()).sum()
    }

    /// Minimum time series length
    fn min_ts_length(&self) -> usize {
        self.features
            .iter()
            .map(|x| x.min_ts_length())
            .max()
            .unwrap_or(0)
    }
}
