use crate::evaluator::FeatureEvaluator;
use crate::float_trait::Float;
use crate::time_series::TimeSeries;

#[derive(Clone, Default)]
pub struct Duration {}

impl Duration {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for Duration
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        vec![ts.t.sample[ts.lenu() - 1] - ts.t.sample[0]]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_duration"]
    }

    fn size_hint(&self) -> usize {
        1
    }

    fn min_ts_length(&self) -> usize {
        1
    }
}

#[derive(Clone, Default)]
pub struct MaximumTimeInterval {}

impl MaximumTimeInterval {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for MaximumTimeInterval
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        assert!(ts.lenu() >= 2);
        let dt = (0..ts.lenu() - 1)
            .map(|i| ts.t.sample[i + 1] - ts.t.sample[i])
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        vec![dt]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_maximum_time_interval"]
    }

    fn size_hint(&self) -> usize {
        1
    }

    fn min_ts_length(&self) -> usize {
        2
    }
}

#[derive(Clone, Default)]
pub struct MinimumTimeInterval {}

impl MinimumTimeInterval {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for MinimumTimeInterval
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        assert!(ts.lenu() >= 2);
        let dt = (0..ts.lenu() - 1)
            .map(|i| ts.t.sample[i + 1] - ts.t.sample[i])
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        vec![dt]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_minimum_time_interval"]
    }

    fn size_hint(&self) -> usize {
        1
    }

    fn min_ts_length(&self) -> usize {
        2
    }
}

#[derive(Clone, Default)]
pub struct ObservationCount {}

impl ObservationCount {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for ObservationCount
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        vec![ts.lenf()]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_observation_count"]
    }

    fn size_hint(&self) -> usize {
        1
    }

    fn min_ts_length(&self) -> usize {
        0
    }
}

#[derive(Clone, Default)]
pub struct TimeMean {}

impl TimeMean {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for TimeMean
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        vec![ts.t.get_mean()]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_time_mean"]
    }

    fn size_hint(&self) -> usize {
        1
    }

    fn min_ts_length(&self) -> usize {
        1
    }
}

#[derive(Clone, Default)]
pub struct TimeStandardDeviation {}

impl TimeStandardDeviation {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for TimeStandardDeviation
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        vec![ts.t.get_std()]
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_time_standard_deviation"]
    }

    fn size_hint(&self) -> usize {
        1
    }

    fn min_ts_length(&self) -> usize {
        2
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
