use crate::evaluator::*;

use conv::ConvUtil;

macro_const! {
    const DOC: &str = r#"
Fraction of observations inside $\mathrm{Median}(m) \pm q \times (\max(m) - \min(m)) / 2$ interval

- Depends on: **magnitude**
- Minimum number of observations: **1**
- Number of features: **1**

D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)   
"#;
}

#[doc = DOC!()]
#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    into = "MedianBufferRangePercentageParameters<T>",
    from = "MedianBufferRangePercentageParameters<T>",
    bound = "T: Float"
)]
pub struct MedianBufferRangePercentage<T> {
    quantile: T,
    name: String,
    description: String,
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
            description: format!(
                "fraction of observations which magnitudes differ from median by no more than \
                {:.3e} of amplitude",
                quantile,
            ),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[inline]
    pub fn default_quantile() -> T {
        0.1_f32.value_as::<T>().unwrap()
    }
}

impl<T> MedianBufferRangePercentage<T> {
    pub fn doc() -> &'static str {
        DOC
    }
}

impl<T> Default for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_quantile())
    }
}

impl<T> FeatureEvaluator<T> for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_median = ts.m.get_median();
        let amplitude = T::half() * (ts.m.get_max() - ts.m.get_min());
        let threshold = self.quantile * amplitude;
        let count_under = ts.m.sample.fold(0, |count, &m| {
            let under = T::abs(m - m_median) < threshold;
            count + (under as u32)
        });
        Ok(vec![count_under.value_as::<T>().unwrap() / ts.lenf()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &MEDIAN_BUFFER_RANGE_PERCENTAGE_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![self.description.as_str()]
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "MedianBufferRangePercentage")]
struct MedianBufferRangePercentageParameters<T> {
    quantile: T,
}

impl<T> From<MedianBufferRangePercentage<T>> for MedianBufferRangePercentageParameters<T> {
    fn from(f: MedianBufferRangePercentage<T>) -> Self {
        Self {
            quantile: f.quantile,
        }
    }
}

impl<T> From<MedianBufferRangePercentageParameters<T>> for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn from(p: MedianBufferRangePercentageParameters<T>) -> Self {
        Self::new(p.quantile)
    }
}

impl<T> JsonSchema for MedianBufferRangePercentage<T>
where
    T: Float,
{
    json_schema!(MedianBufferRangePercentageParameters<T>, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    use serde_test::{assert_tokens, Token};

    check_feature!(MedianBufferRangePercentage<f64>);

    feature_test!(
        median_buffer_range_percentage,
        [
            MedianBufferRangePercentage::default(),
            MedianBufferRangePercentage::new(0.1), // should be the same
            MedianBufferRangePercentage::new(0.2),
        ],
        [0.5555555555555556, 0.5555555555555556, 0.7777777777777778],
        [1.0_f32, 41.0, 49.0, 49.0, 50.0, 51.0, 52.0, 58.0, 100.0],
    );

    feature_test!(
        median_buffer_range_percentage_plateau,
        [MedianBufferRangePercentage::default()],
        [0.0],
        [0.0; 10],
    );

    #[test]
    fn serialization() {
        const QUANTILE: f64 = 0.432;
        let median_buffer_range_percentage = MedianBufferRangePercentage::new(QUANTILE);
        assert_tokens(
            &median_buffer_range_percentage,
            &[
                Token::Struct {
                    len: 1,
                    name: "MedianBufferRangePercentage",
                },
                Token::String("quantile"),
                Token::F64(QUANTILE),
                Token::StructEnd,
            ],
        )
    }
}
