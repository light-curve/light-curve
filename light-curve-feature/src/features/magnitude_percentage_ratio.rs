use crate::evaluator::*;

macro_const! {
    const DOC: &str = r#"
Magnitude percentage ratio

$$
\mathrm{magnitude~}q\mathrm{~to~}n\mathrm{~ratio} \equiv \frac{Q(1-n) - Q(n)}{Q(1-d) - Q(d)},
$$
where $n$ and $d$ denotes user defined percentage, $Q$ is the quantile function of magnitude
distribution.

- Depends on: **magnitude**
- Minimum number of observations: **1**
- Number of features: **1**

D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)    
"#;
}

#[doc = DOC!()]
#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(
    from = "MagnitudePercentageRatioParameters",
    into = "MagnitudePercentageRatioParameters"
)]
pub struct MagnitudePercentageRatio {
    quantile_numerator: f32,
    quantile_denominator: f32,
    name: String,
    description: String,
}

lazy_info!(
    MAGNITUDE_PERCENTAGE_RATIO_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl MagnitudePercentageRatio {
    pub fn new(quantile_numerator: f32, quantile_denominator: f32) -> Self {
        assert!(
            (quantile_numerator > 0.0)
                && (quantile_numerator < 0.5)
                && (quantile_denominator > 0.0)
                && (quantile_denominator < 0.5),
            "quantiles should be between zero and half"
        );
        Self {
            quantile_numerator,
            quantile_denominator,
            name: format!(
                "magnitude_percentage_ratio_{:.0}_{:.0}",
                100.0 * quantile_numerator,
                100.0 * quantile_denominator
            ),
            description: format!(
                "ratio of {:.3e}% - {:.3e}% and {:.3e}% - {:.3e}% percentile ranges of magnitude \
                sample",
                100.0 * quantile_numerator,
                100.0 * (1.0 - quantile_numerator),
                100.0 * quantile_denominator,
                100.0 * (1.0 - quantile_denominator),
            ),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[inline]
    pub fn default_quantile_numerator() -> f32 {
        0.4
    }

    #[inline]
    pub fn default_quantile_denominator() -> f32 {
        0.05
    }

    pub fn doc() -> &'static str {
        DOC
    }
}
impl Default for MagnitudePercentageRatio {
    fn default() -> Self {
        Self::new(
            Self::default_quantile_numerator(),
            Self::default_quantile_denominator(),
        )
    }
}

impl<T> FeatureEvaluator<T> for MagnitudePercentageRatio
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_sorted = ts.m.get_sorted();
        let numerator =
            m_sorted.ppf(1.0 - self.quantile_numerator) - m_sorted.ppf(self.quantile_numerator);
        let denumerator =
            m_sorted.ppf(1.0 - self.quantile_denominator) - m_sorted.ppf(self.quantile_denominator);
        if numerator.is_zero() & denumerator.is_zero() {
            Err(EvaluatorError::FlatTimeSeries)
        } else {
            Ok(vec![numerator / denumerator])
        }
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &MAGNITUDE_PERCENTAGE_RATIO_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![self.description.as_str()]
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "MagnitudePercentageRatio")]
struct MagnitudePercentageRatioParameters {
    quantile_numerator: f32,
    quantile_denominator: f32,
}

impl From<MagnitudePercentageRatio> for MagnitudePercentageRatioParameters {
    fn from(f: MagnitudePercentageRatio) -> Self {
        Self {
            quantile_numerator: f.quantile_numerator,
            quantile_denominator: f.quantile_denominator,
        }
    }
}

impl From<MagnitudePercentageRatioParameters> for MagnitudePercentageRatio {
    fn from(p: MagnitudePercentageRatioParameters) -> Self {
        Self::new(p.quantile_numerator, p.quantile_denominator)
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    use serde_test::{assert_tokens, Token};

    check_feature!(MagnitudePercentageRatio);

    feature_test!(
        magnitude_percentage_ratio,
        [
            MagnitudePercentageRatio::default(),
            MagnitudePercentageRatio::new(0.4, 0.05), // should be the same
            MagnitudePercentageRatio::new(0.2, 0.05),
            MagnitudePercentageRatio::new(0.4, 0.1),
        ],
        [0.12886598, 0.12886598, 0.7628866, 0.13586957],
        [
            80.0_f32, 13.0, 20.0, 20.0, 75.0, 25.0, 100.0, 1.0, 2.0, 3.0, 7.0, 30.0, 5.0, 9.0,
            10.0, 70.0, 80.0, 92.0, 97.0, 17.0
        ],
    );

    #[test]
    fn magnitude_percentage_ratio_plateau() {
        let eval = MagnitudePercentageRatio::default();
        let x = [0.0; 10];
        let mut ts = TimeSeries::new_without_weight(&x, &x);
        assert_eq!(eval.eval(&mut ts), Err(EvaluatorError::FlatTimeSeries));
    }

    #[test]
    fn serialization() {
        const QUANTILE_NUMERATOR: f32 = 0.256;
        const QUANTILE_DENOMINATOR: f32 = 0.128;

        let beyond_n_std = MagnitudePercentageRatio::new(QUANTILE_NUMERATOR, QUANTILE_DENOMINATOR);
        assert_tokens(
            &beyond_n_std,
            &[
                Token::Struct {
                    len: 2,
                    name: "MagnitudePercentageRatio",
                },
                Token::String("quantile_numerator"),
                Token::F32(QUANTILE_NUMERATOR),
                Token::String("quantile_denominator"),
                Token::F32(QUANTILE_DENOMINATOR),
                Token::StructEnd,
            ],
        )
    }
}
