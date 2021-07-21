use crate::error::EvaluatorError;
use crate::evaluator::*;
use crate::float_trait::Float;
use crate::time_series::TimeSeries;

use serde::ser::SerializeStruct;
use serde::Serializer;
use std::marker::PhantomData;

/// The engine that extracts features one by one
#[derive(Clone, Debug)]
pub struct FeatureExtractor<T, F> {
    info: Box<EvaluatorInfo>,
    features: Vec<F>,
    phantom: PhantomData<T>,
}

impl<T, F> FeatureExtractor<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    pub fn new(features: Vec<F>) -> Self {
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
        }
        .into();
        Self {
            info,
            features,
            phantom: PhantomData,
        }
    }

    pub fn get_features(&self) -> &Vec<F> {
        &self.features
    }

    pub fn add_feature(&mut self, feature: F) {
        self.features.push(feature);
    }
}

impl<T, F> FeatureEvaluator<T> for FeatureExtractor<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
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

    /// Get feature names
    fn get_names(&self) -> Vec<&str> {
        self.features.iter().flat_map(|x| x.get_names()).collect()
    }

    /// Get feature descriptions
    fn get_descriptions(&self) -> Vec<&str> {
        self.features
            .iter()
            .flat_map(|x| x.get_descriptions())
            .collect()
    }
}

impl<T, F> Serialize for FeatureExtractor<T, F>
where
    F: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("FeatureExtractor", 1)?;
        state.serialize_field("features", &self.features)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_test::{assert_ser_tokens, Token};

    #[test]
    fn serialization_empty() {
        let fe: FeatureExtractor<f64, crate::Feature<_>> = FeatureExtractor::new(vec![]);
        assert_ser_tokens(
            &fe,
            &[
                //
                Token::Struct {
                    len: 1,
                    name: "FeatureExtractor",
                },
                //
                Token::String("features"),
                Token::Seq { len: Some(0) },
                Token::SeqEnd,
                //
                Token::StructEnd,
            ],
        )
    }
}
