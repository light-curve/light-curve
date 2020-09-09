use crate::float_trait::Float;
use crate::time_series::TimeSeries;

use dyn_clonable::*;

/// The trait each feature should implement
#[clonable]
pub trait FeatureEvaluator<T: Float>: Send + Sync + Clone {
    /// Should return the vector of feature values. The length and feature order should
    /// correspond to `get_names()` output
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T>;

    /// Should return the vector of feature names. The length and feature order should
    /// correspond to `eval()` output
    fn get_names(&self) -> Vec<&str>;

    /// Should return the size of vectors returned by `eval()` and `get_names()`
    fn size_hint(&self) -> usize;
}

pub type VecFE<T> = Vec<Box<dyn FeatureEvaluator<T>>>;
