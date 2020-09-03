use crate::evaluator::FeatureEvaluator;
use crate::float_trait::Float;
use crate::time_series::TimeSeries;

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
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    use crate::extractor::FeatureExtractor;

    use light_curve_common::all_close;

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
