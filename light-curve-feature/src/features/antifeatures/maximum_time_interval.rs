use crate::evaluator::*;
use itertools::Itertools;

macro_const! {
    const DOC: &str = r#"
Maximum time interval between consequent observations

$$
\max{(t_{i+1} - t_i)}
$$

- Depends on: **time**
- Minimum number of observations: **2**
- Number of features: **1**
"#;
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Deserialize, Serialize, JsonSchema)]
pub struct MaximumTimeInterval {}

impl MaximumTimeInterval {
    pub fn new() -> Self {
        Self {}
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    MAXIMUM_TIME_INTERVAL_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: true,
    m_required: false,
    w_required: false,
    sorting_required: true,
);

impl<T> FeatureEvaluator<T> for MaximumTimeInterval
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let dt =
            ts.t.as_slice()
                .iter()
                .tuple_windows()
                .map(|(&a, &b)| b - a)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
        Ok(vec![dt])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &MAXIMUM_TIME_INTERVAL_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_maximum_time_interval"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["maximum time interval between consequent observations"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(MaximumTimeInterval);

    feature_test!(
        maximum_time_interval,
        [MaximumTimeInterval::new()],
        [9.0],
        [0.0_f32, 0.5, 0.6, 1.6, 10.6],
    );
}
