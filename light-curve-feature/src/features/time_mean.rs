use crate::evaluator::*;

macro_const! {
    const DOC: &'static str = r#"
Mean time

$$
\langle t \rangle \equiv \frac1{N} \sum_i {t_i}.
$$

Note: highly cadence-dependent feature.

- Depends on: **time**
- Minimum number of observations: **1**
- Number of features: **1**
"#;
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Deserialize, Serialize, JsonSchema)]
pub struct TimeMean {}

impl TimeMean {
    pub fn new() -> Self {
        Self {}
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    TIME_MEAN_INFO,
    TimeMean,
    size: 1,
    min_ts_length: 1,
    t_required: true,
    m_required: false,
    w_required: false,
    sorting_required: false,
);
impl FeatureNamesDescriptionsTrait for TimeMean {
    fn get_names(&self) -> Vec<&str> {
        vec!["time_mean"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["mean of time moments"]
    }
}
impl<T> FeatureEvaluator<T> for TimeMean
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.t.get_mean()])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(TimeMean);

    feature_test!(
        time_mean,
        [TimeMean::new()],
        [14.0],
        [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0, 6.0, 99.0],
    );
}
