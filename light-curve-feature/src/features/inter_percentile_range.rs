use crate::evaluator::*;

macro_const! {
    const DOC: &'static str = r#"
Inter-percentile range

$$
\mathrm{inter-percetile range} \equiv Q(1 - p) - Q(p),
$$
where $Q(p)$ is the $p$th quantile of the magnitude distribution.

Special cases are [the interquartile range](https://en.wikipedia.org/wiki/Interquartile_range)
which is inter-percentile range for $p = 0.25$ and
[the interdecile range](https://en.wikipedia.org/wiki/Interdecile_range) which is
inter-percentile range for $p = 0.1$.

- Depends on: **magnitude**
- Minimum number of observations: **1**
- Number of features: **1**
"#;
}

#[doc = DOC!()]
#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    from = "InterPercentileRangeParameters",
    into = "InterPercentileRangeParameters"
)]
pub struct InterPercentileRange {
    quantile: f32,
    name: String,
    description: String,
}

lazy_info!(
    INTER_PERCENTILE_RANGE_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl InterPercentileRange {
    pub fn new(quantile: f32) -> Self {
        assert!(
            (quantile > 0.0) && (quantile < 0.5),
            "Quanitle should be in range (0.0, 0.5)"
        );
        Self {
            quantile,
            name: format!("inter_percentile_range_{:.0}", 100.0 * quantile),
            description: format!(
                "range between {:.3e}% and {:.3e}% magnitude percentiles",
                100.0 * quantile,
                100.0 * (1.0 - quantile)
            ),
        }
    }

    #[inline]
    pub fn default_quantile() -> f32 {
        0.25
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

impl Default for InterPercentileRange {
    fn default() -> Self {
        Self::new(Self::default_quantile())
    }
}

impl<T> FeatureEvaluator<T> for InterPercentileRange
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let ppf_low = ts.m.get_sorted().ppf(self.quantile);
        let ppf_high = ts.m.get_sorted().ppf(1.0 - self.quantile);
        let value = ppf_high - ppf_low;
        Ok(vec![value])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &INTER_PERCENTILE_RANGE_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![self.description.as_str()]
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "InterPercentileRange")]
struct InterPercentileRangeParameters {
    quantile: f32,
}

impl From<InterPercentileRange> for InterPercentileRangeParameters {
    fn from(f: InterPercentileRange) -> Self {
        Self {
            quantile: f.quantile,
        }
    }
}

impl From<InterPercentileRangeParameters> for InterPercentileRange {
    fn from(p: InterPercentileRangeParameters) -> Self {
        Self::new(p.quantile)
    }
}

impl JsonSchema for InterPercentileRange {
    json_schema!(InterPercentileRangeParameters, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    use serde_test::{assert_tokens, Token};

    check_feature!(InterPercentileRange);

    feature_test!(
        inter_percentile_range,
        [
            InterPercentileRange::default(),
            InterPercentileRange::new(0.25), // should be the same
            InterPercentileRange::new(0.1),
        ],
        [50.0, 50.0, 80.0],
        linspace(0.0, 99.0, 100),
    );

    #[test]
    fn serialization() {
        const QUANTILE: f32 = 0.256;
        let beyond_n_std = InterPercentileRange::new(QUANTILE);
        assert_tokens(
            &beyond_n_std,
            &[
                Token::Struct {
                    len: 1,
                    name: "InterPercentileRange",
                },
                Token::String("quantile"),
                Token::F32(QUANTILE),
                Token::StructEnd,
            ],
        )
    }
}
