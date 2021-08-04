use crate::evaluator::*;

macro_const! {
    const DOC: &'static str = r#"
Half amplitude of magnitude

$$
\mathrm{amplitude} \equiv \frac{\left( \max{(m)} - \min{(m)} \right)}{2}
$$

- Depends on: **magnitude**
- Minimum number of observations: **1**
- Number of features: **1**    
"#;
}

#[doc = DOC!()]
/// ### Example
/// ```
/// use light_curve_feature::*;
///
/// let amplitude = Amplitude::default();
/// let time = [0.0, 1.0];  // Doesn't depend on time
/// let magn = [0.0, 2.0];
/// let mut ts = TimeSeries::new_without_weight(&time[..], &magn[..]);
/// assert_eq!(vec![1.0], amplitude.eval(&mut ts).unwrap());
/// ```
#[derive(Clone, Default, Debug, Deserialize, Serialize, JsonSchema)]
pub struct Amplitude {}

impl Amplitude {
    pub fn new() -> Self {
        Self {}
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    AMPLITUDE_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl<T> FeatureEvaluator<T> for Amplitude
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![T::half() * (ts.m.get_max() - ts.m.get_min())])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &AMPLITUDE_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["amplitude"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["half of the interval between maximum and minimum magnitude"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(Amplitude);

    feature_test!(amplitude, [Amplitude::new()], [1.0], [0.0_f32, 1.0, 2.0],);
}
