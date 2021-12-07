use crate::evaluator::*;

macro_const! {
    const DOC: &'static str = r#"
Reduced $\chi^2$ of magnitude measurements

$$
\mathrm{reduced~}\chi^2 \equiv \frac1{N-1} \sum_i\left(\frac{m_i - \bar{m}}{\delta\_i}\right)^2,
$$
where $N$ is the number of observations,
and $\bar{m}$ is the weighted mean magnitude.

- Depends on: **magnitude**, **magnitude error**
- Minimum number of observations: **2**
- Number of features: **1**

This is a good measure of variability which takes into account observations uncertainties.

[Wikipedia](https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic)
"#;
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReducedChi2 {}

lazy_info!(
    REDUCED_CHI2_INFO,
    ReducedChi2,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: true,
    sorting_required: false,
);

impl ReducedChi2 {
    pub fn new() -> Self {
        Self {}
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for ReducedChi2 {
    fn get_names(&self) -> Vec<&str> {
        vec!["chi2"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["reduced chi2 as a goodness of constant fit with respect to observation errors"]
    }
}

impl<T> FeatureEvaluator<T> for ReducedChi2
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.get_m_reduced_chi2()])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(ReducedChi2);

    feature_test!(
        reduced_chi2,
        [ReducedChi2::default()],
        [2.192592592592593],
        [0.0_f64; 10], // isn't used
        [1.0, 2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, -2.0, 0.0],
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    );
}
