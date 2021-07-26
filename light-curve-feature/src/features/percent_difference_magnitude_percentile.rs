use crate::evaluator::*;

/// Ratio of $p$th inter-percentile range to the median
///
/// $$
/// p\mathrm{~percent~difference~magnitude~percentile} \equiv \frac{Q(1-p) - Q(p)}{\mathrm{Median}(m)}.
/// $$
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    into = "PercentDifferenceMagnitudePercentileParameters",
    from = "PercentDifferenceMagnitudePercentileParameters"
)]
pub struct PercentDifferenceMagnitudePercentile {
    quantile: f32,
    name: String,
    description: String,
}

lazy_info!(
    PERCENT_DIFFERENCE_MAGNITUDE_PERCENTILE_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl PercentDifferenceMagnitudePercentile {
    pub fn new(quantile: f32) -> Self {
        assert!(
            (quantile > 0.0) && (quantile < 0.5),
            "quantiles should be between zero and half"
        );
        Self {
            quantile,
            name: format!(
                "percent_difference_magnitude_percentile_{:.0}",
                100.0 * quantile
            ),
            description: format!(
                "ratio of inter-percentile {:.3e}% - {:.3e}% range of magnitude to its mdeian",
                100.0 * quantile,
                100.0 * (1.0 - quantile),
            ),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[inline]
    pub fn default_quantile() -> f32 {
        0.05
    }
}

impl Default for PercentDifferenceMagnitudePercentile {
    fn default() -> Self {
        Self::new(Self::default_quantile())
    }
}

impl<T> FeatureEvaluator<T> for PercentDifferenceMagnitudePercentile
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let nominator =
            ts.m.get_sorted().ppf(1.0 - self.quantile) - ts.m.get_sorted().ppf(self.quantile);
        let denominator = ts.m.get_median();
        if nominator.is_zero() & denominator.is_zero() {
            Err(EvaluatorError::ZeroDivision("median magnitude is zero"))
        } else {
            Ok(vec![nominator / denominator])
        }
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &PERCENT_DIFFERENCE_MAGNITUDE_PERCENTILE_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![self.description.as_str()]
    }
}

#[derive(Deserialize, Serialize, JsonSchema)]
#[serde(rename = "PercentDifferenceMagnitudePercentile")]
struct PercentDifferenceMagnitudePercentileParameters {
    quantile: f32,
}

impl From<PercentDifferenceMagnitudePercentile> for PercentDifferenceMagnitudePercentileParameters {
    fn from(f: PercentDifferenceMagnitudePercentile) -> Self {
        Self {
            quantile: f.quantile,
        }
    }
}

impl From<PercentDifferenceMagnitudePercentileParameters> for PercentDifferenceMagnitudePercentile {
    fn from(p: PercentDifferenceMagnitudePercentileParameters) -> Self {
        Self::new(p.quantile)
    }
}

impl JsonSchema for PercentDifferenceMagnitudePercentile {
    json_schema!(PercentDifferenceMagnitudePercentileParameters, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    use serde_test::{assert_tokens, Token};

    check_feature!(PercentDifferenceMagnitudePercentile);

    feature_test!(
        percent_difference_magnitude_percentile,
        [
            PercentDifferenceMagnitudePercentile::default(),
            PercentDifferenceMagnitudePercentile::new(0.05), // should be the same
            PercentDifferenceMagnitudePercentile::new(0.1),
        ],
        [4.85, 4.85, 4.6],
        [
            80.0_f32, 13.0, 20.0, 20.0, 75.0, 25.0, 100.0, 1.0, 2.0, 3.0, 7.0, 30.0, 5.0, 9.0,
            10.0, 70.0, 80.0, 92.0, 97.0, 17.0
        ],
    );

    #[test]
    fn serialization() {
        const QUANTILE: f32 = 0.017;
        let percent_difference_magnitude_percentile =
            PercentDifferenceMagnitudePercentile::new(QUANTILE);
        assert_tokens(
            &percent_difference_magnitude_percentile,
            &[
                Token::Struct {
                    len: 1,
                    name: "PercentDifferenceMagnitudePercentile",
                },
                Token::String("quantile"),
                Token::F32(QUANTILE),
                Token::StructEnd,
            ],
        )
    }
}
