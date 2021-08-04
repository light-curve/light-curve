use crate::evaluator::*;

use conv::ConvUtil;

macro_const! {
    const DOC: &str = r#"
Fraction of observations beyond $n\,\sigma\_m$ from the mean magnitude $\langle m \rangle$

$$
\mathrm{beyond}~n\,\sigma\_m \equiv \frac{\sum\_i I\_{|m - \langle m \rangle| > n\,\sigma\_m}(m_i)}{N},
$$
where $I$ is the [indicator function](https://en.wikipedia.org/wiki/Indicator_function),
$N$ is the number of observations,
$\langle m \rangle$ is the mean magnitude
and $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.

- Depends on: **magnitude**
- Minimum number of observations: **2**
- Number of features: **1**

D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
"#;
}

#[doc = DOC!()]
/// ### Example
/// ```
/// use light_curve_feature::*;
/// use light_curve_common::all_close;
/// use std::f64::consts::SQRT_2;
///
/// let fe = FeatureExtractor::new(vec![BeyondNStd::default(), BeyondNStd::new(2.0)]);
/// let time = [0.0; 21];  // Doesn't depend on time
/// let mut magn = vec![0.0; 17];
/// magn.extend_from_slice(&[SQRT_2, -SQRT_2, 2.0 * SQRT_2, -2.0 * SQRT_2]);
/// let mut ts = TimeSeries::new_without_weight(&time[..], &magn[..]);
/// assert_eq!(0.0, ts.m.get_mean());
/// assert!((1.0 - ts.m.get_std()).abs() < 1e-15);
/// assert_eq!(vec![4.0 / 21.0, 2.0 / 21.0], fe.eval(&mut ts).unwrap());
/// ```
#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    from = "BeyondNStdParameters<T>",
    into = "BeyondNStdParameters<T>",
    bound = "T: Float"
)]
pub struct BeyondNStd<T> {
    nstd: T,
    name: String,
    description: String,
}

impl<T> BeyondNStd<T>
where
    T: Float,
{
    pub fn new(nstd: T) -> Self {
        assert!(nstd > T::zero(), "nstd should be positive");
        Self {
            nstd,
            name: format!("beyond_{:.0}_std", nstd),
            description: format!(
                "fraction of observations which magnitudes are beyond {:.3e} standard deviations \
                from the mean magnitude",
                nstd
            ),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[inline]
    pub fn default_nstd() -> T {
        T::one()
    }
}

impl<T> BeyondNStd<T> {
    pub fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    BEYOND_N_STD_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl<T> Default for BeyondNStd<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_nstd())
    }
}

impl<T> FeatureEvaluator<T> for BeyondNStd<T>
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_mean = ts.m.get_mean();
        let threshold = ts.m.get_std() * self.nstd;
        let count_beyond = ts.m.sample.fold(0, |count, &m| {
            let beyond = T::abs(m - m_mean) > threshold;
            count + (beyond as u32)
        });
        Ok(vec![count_beyond.value_as::<T>().unwrap() / ts.lenf()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &BEYOND_N_STD_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![self.description.as_str()]
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "BeyondNStd")]
struct BeyondNStdParameters<T> {
    nstd: T,
}

impl<T> From<BeyondNStd<T>> for BeyondNStdParameters<T> {
    fn from(f: BeyondNStd<T>) -> Self {
        Self { nstd: f.nstd }
    }
}

impl<T> From<BeyondNStdParameters<T>> for BeyondNStd<T>
where
    T: Float,
{
    fn from(p: BeyondNStdParameters<T>) -> Self {
        Self::new(p.nstd)
    }
}

impl<T> JsonSchema for BeyondNStd<T>
where
    T: Float,
{
    json_schema!(BeyondNStdParameters<T>, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    use serde_test::{assert_tokens, Token};

    check_feature!(BeyondNStd<f64>);

    feature_test!(
        beyond_n_std,
        [
            BeyondNStd::default(),
            BeyondNStd::new(1.0), // should be the same as the previous one
            BeyondNStd::new(2.0),
        ],
        [0.2, 0.2, 0.0],
        [1.0_f32, 2.0, 3.0, 4.0, 100.0],
    );

    #[test]
    fn serialization() {
        const NSTD: f64 = 2.34;
        let beyond_n_std = BeyondNStd::new(NSTD);
        assert_tokens(
            &beyond_n_std,
            &[
                Token::Struct {
                    len: 1,
                    name: "BeyondNStd",
                },
                Token::String("nstd"),
                Token::F64(NSTD),
                Token::StructEnd,
            ],
        )
    }
}
