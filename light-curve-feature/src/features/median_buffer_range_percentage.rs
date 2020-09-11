use crate::evaluator::*;

use conv::ConvUtil;

/// Fraction of observations inside $\mathrm{Median}(m) \pm q \times \mathrm{Median}(m)$ interval
///
/// $$
/// \mathrm{median~buffer~range}~q~\mathrm{percentage} \equiv \frac{\sum\_i I\_{|m - \mathrm{Median}(m)| < q\\,\mathrm{Median}(m)}(m_i)}{N},
/// $$
/// where $I$ is the [indicator function](https://en.wikipedia.org/wiki/Indicator_function),
/// and $N$ is the number of observations.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
#[derive(Clone)]
pub struct MedianBufferRangePercentage<T>
where
    T: Float,
{
    quantile: T,
    name: String,
}

lazy_info!(
    MEDIAN_BUFFER_RANGE_PERCENTAGE_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl<T> MedianBufferRangePercentage<T>
where
    T: Float,
{
    pub fn new(quantile: T) -> Self {
        assert!(quantile > T::zero(), "Quanitle should be positive");
        Self {
            quantile,
            name: format!(
                "median_buffer_range_percentage_{:.0}",
                T::hundred() * quantile
            ),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

impl<T> Default for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(0.1_f32.value_as::<T>().unwrap())
    }
}

impl<T> FeatureEvaluator<T> for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_median = ts.m.get_median();
        let threshold = self.quantile * m_median;
        Ok(vec![
            ts.m.sample
                .iter()
                .cloned()
                .filter(|&y| T::abs(y - m_median) < threshold)
                .count()
                .value_as::<T>()
                .unwrap()
                / ts.lenf(),
        ])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &MEDIAN_BUFFER_RANGE_PERCENTAGE_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(
        median_buffer_range_percentage_info,
        MedianBufferRangePercentage::default()
    );

    feature_test!(
        median_buffer_range_percentage,
        [
            Box::new(MedianBufferRangePercentage::default()),
            Box::new(MedianBufferRangePercentage::new(0.1)), // should be the same
            Box::new(MedianBufferRangePercentage::new(0.2)),
        ],
        [0.555555555, 0.555555555, 0.777777777],
        [1.0_f32, 41.0, 49.0, 49.0, 50.0, 51.0, 52.0, 58.0, 100.0],
    );

    feature_test!(
        median_buffer_range_percentage_plateau,
        [Box::new(MedianBufferRangePercentage::default())],
        [0.0],
        [0.0; 10],
    );
}
