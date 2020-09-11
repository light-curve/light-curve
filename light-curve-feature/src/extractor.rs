use crate::error::EvaluatorError;
use crate::evaluator::*;
use crate::float_trait::Float;
use crate::time_series::TimeSeries;

/// The engine that extracts features one by one
#[derive(Clone, Debug)]
pub struct FeatureExtractor<T: Float> {
    info: EvaluatorInfo,
    features: VecFE<T>,
}

impl<T> FeatureExtractor<T>
where
    T: Float,
{
    pub fn new(features: VecFE<T>) -> Self {
        let info = EvaluatorInfo {
            size: features.iter().map(|x| x.size_hint()).sum(),
            min_ts_length: features
                .iter()
                .map(|x| x.min_ts_length())
                .max()
                .unwrap_or(0),
            t_required: features.iter().any(|x| x.is_t_required()),
            m_required: features.iter().any(|x| x.is_m_required()),
            w_required: features.iter().any(|x| x.is_w_required()),
            sorting_required: features.iter().any(|x| x.is_sorting_required()),
        };
        Self { info, features }
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
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let mut vec = Vec::with_capacity(self.size_hint());
        for x in self.features.iter() {
            vec.extend(x.eval(ts)?);
        }
        Ok(vec)
    }

    fn eval_or_fill(&self, ts: &mut TimeSeries<T>, fill_value: T) -> Vec<T> {
        self.features
            .iter()
            .flat_map(|x| x.eval_or_fill(ts, fill_value))
            .collect()
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &self.info
    }

    /// Get a vector of feature names.
    /// The length of the returned vector is guaranteed to be the same as returned by `eval()`
    fn get_names(&self) -> Vec<&str> {
        self.features.iter().flat_map(|x| x.get_names()).collect()
    }
}
