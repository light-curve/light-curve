use crate::evaluator::*;

macro_const! {
    const DOC: &'static str = r#"
Number of observations

$$
N
$$

- Depends on: nothing
- Minimum number of observations: **0**
- Number of features: **1**
"#;
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Deserialize, Serialize, JsonSchema)]
pub struct ObservationCount {}

impl ObservationCount {
    pub fn new() -> Self {
        Self {}
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    OBSERVATION_COUNT_INFO,
    ObservationCount,
    size: 1,
    min_ts_length: 0,
    t_required: false,
    m_required: false,
    w_required: false,
    sorting_required: false,
);

impl FeatureNamesDescriptionsTrait for ObservationCount {
    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_observation_count"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["observation count"]
    }
}

impl<T> FeatureEvaluator<T> for ObservationCount
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.lenf()])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(ObservationCount);

    feature_test!(
        observation_count,
        [ObservationCount::new()],
        [5.0],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
    );
}
