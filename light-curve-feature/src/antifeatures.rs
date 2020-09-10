use crate::error::EvaluatorError;
use crate::evaluator::*;
use crate::float_trait::Float;
use crate::time_series::TimeSeries;

use lazy_static::lazy_static;

#[derive(Clone, Default)]
pub struct Duration {}

impl Duration {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    DURATION_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: true,
    m_required: false,
    w_required: false,
    sorting_required: true,
);

impl<T> FeatureEvaluator<T> for Duration
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        check_ts_length(self, ts)?;
        Ok(vec![ts.t.sample[ts.lenu() - 1] - ts.t.sample[0]])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &DURATION_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_duration"]
    }
}

#[derive(Clone, Default)]
pub struct MaximumTimeInterval {}

impl MaximumTimeInterval {
    pub fn new() -> Self {
        Self {}
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
        check_ts_length(self, ts)?;
        let dt = (0..ts.lenu() - 1)
            .map(|i| ts.t.sample[i + 1] - ts.t.sample[i])
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
}

#[derive(Clone, Default)]
pub struct MinimumTimeInterval {}

impl MinimumTimeInterval {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    MINIMUM_TIME_INTERVAL_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: true,
    m_required: false,
    w_required: false,
    sorting_required: true,
);

impl<T> FeatureEvaluator<T> for MinimumTimeInterval
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        check_ts_length(self, ts)?;
        let dt = (0..ts.lenu() - 1)
            .map(|i| ts.t.sample[i + 1] - ts.t.sample[i])
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        Ok(vec![dt])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &MINIMUM_TIME_INTERVAL_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_minimum_time_interval"]
    }
}

#[derive(Clone, Default)]
pub struct ObservationCount {}

impl ObservationCount {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    OBSERVATION_COUNT_INFO,
    size: 1,
    min_ts_length: 0,
    t_required: false,
    m_required: false,
    w_required: false,
    sorting_required: false,
);

impl<T> FeatureEvaluator<T> for ObservationCount
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        check_ts_length(self, ts)?;
        Ok(vec![ts.lenf()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &OBSERVATION_COUNT_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_observation_count"]
    }
}

#[derive(Clone, Default)]
pub struct TimeMean {}

impl TimeMean {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    TIME_MEAN_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: true,
    m_required: false,
    w_required: false,
    sorting_required: false,
);

impl<T> FeatureEvaluator<T> for TimeMean
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        check_ts_length(self, ts)?;
        Ok(vec![ts.t.get_mean()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &TIME_MEAN_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_time_mean"]
    }
}

#[derive(Clone, Default)]
pub struct TimeStandardDeviation {}

impl TimeStandardDeviation {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    TIME_STANDARD_DEVIATION_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: true,
    m_required: false,
    w_required: false,
    sorting_required: false,
);

impl<T> FeatureEvaluator<T> for TimeStandardDeviation
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        check_ts_length(self, ts)?;
        Ok(vec![ts.t.get_std()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &TIME_STANDARD_DEVIATION_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_time_standard_deviation"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    use crate::extractor::FeatureExtractor;

    use light_curve_common::all_close;

    feature_test!(
        duration,
        [Box::new(Duration::new())],
        [4.0],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
    );

    feature_test!(
        maximum_time_interval,
        [Box::new(MaximumTimeInterval::new())],
        [9.0],
        [0.0_f32, 0.5, 0.6, 1.6, 10.6],
    );

    feature_test!(
        minimum_time_interval,
        [Box::new(MinimumTimeInterval::new())],
        [0.1],
        [0.0_f32, 0.5, 0.6, 1.6, 10.6],
    );

    feature_test!(
        observation_count,
        [Box::new(ObservationCount::new())],
        [5.0],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
    );

    feature_test!(
        time_mean,
        [Box::new(TimeMean::new())],
        [14.0],
        [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0, 6.0, 99.0],
    );

    feature_test!(
        time_standard_deviation,
        [Box::new(TimeStandardDeviation::new())],
        [1.5811388300841898],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
    );
}
