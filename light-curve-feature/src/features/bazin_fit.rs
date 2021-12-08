use crate::evaluator::*;
use crate::nl_fit::{
    data::NormalizedData, evaluator::*, CurveFitAlgorithm, CurveFitResult, CurveFitTrait,
    LikeFloat, LnPrior, McmcCurveFit,
};

use conv::ConvUtil;

const NPARAMS: usize = 5;

macro_const! {
    const DOC: &str = r#"
Bazin function fit

Five fit parameters and goodness of fit (reduced $\chi^2$) of the Bazin function developed for
core-collapsed supernovae:

$$
f(t) = A \frac{ \mathrm{e}^{ -(t-t_0)/\tau_\mathrm{fall} } }{ 1 + \mathrm{e}^{ -(t - t_0) / \tau_\mathrm{rise} } } + B.
$$

Note, that the Bazin function is developed to be used with fluxes, not magnitudes. Also note a typo
in the Eq. (1) of the original paper, the minus sign is missed in the "rise" exponent.

- Depends on: **time**, **magnitude**, **magnitude error**
- Minimum number of observations: **6**
- Number of features: **6**

Bazin et al. 2009 [DOI:10.1051/0004-6361/200911847](https://doi.org/10.1051/0004-6361/200911847)
"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct BazinFit {
    algorithm: CurveFitAlgorithm,
    ln_prior: LnPrior,
}

impl BazinFit {
    /// New [BazinFit] instance
    ///
    /// `algorithm` specifies which optimization method is used, it is an instance of the
    /// [CurveFitAlgorithm], currently supported algorithms are [MCMC](McmcCurveFit) and
    /// [LMSDER](crate::nl_fit::LmsderCurveFit) (a Levenberg–Marquard algorithm modification,
    /// requires `gsl` Cargo feature).
    ///
    /// `ln_prior` is an instance of [LnPrior] and specifies the natural logarithm of the prior to
    /// use. Some curve-fit algorithms doesn't support this and ignores the prior
    pub fn new(algorithm: CurveFitAlgorithm, ln_prior: LnPrior) -> Self {
        Self {
            algorithm,
            ln_prior,
        }
    }

    /// Default [McmcCurveFit] for [BazinFit]
    #[inline]
    pub fn default_algorithm() -> CurveFitAlgorithm {
        McmcCurveFit::new(
            McmcCurveFit::default_niterations(),
            McmcCurveFit::default_fine_tuning_algorithm(),
        )
        .into()
    }

    /// Default [LnPrior] for [BazinFit]
    #[inline]
    pub fn default_ln_prior() -> LnPrior {
        LnPrior::none()
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

impl Default for BazinFit {
    fn default() -> Self {
        Self::new(Self::default_algorithm(), Self::default_ln_prior())
    }
}

lazy_info!(
    BAZIN_FIT_INFO,
    BazinFit,
    size: NPARAMS + 1,
    min_ts_length: NPARAMS + 1,
    t_required: true,
    m_required: true,
    w_required: true,
    sorting_required: true, // improve reproducibility
);

struct Params<'a, T> {
    internal: &'a [T; NPARAMS],
    external: [T; NPARAMS],
}

impl<'a, T> Params<'a, T>
where
    T: LikeFloat,
{
    #[inline]
    fn a(&self) -> T {
        self.external[0]
    }

    #[inline]
    fn sgn_a(&self) -> T {
        self.internal[0].signum()
    }

    #[inline]
    fn b(&self) -> T {
        self.external[1]
    }

    #[inline]
    fn t0(&self) -> T {
        self.external[2]
    }

    #[inline]
    fn tau_rise(&self) -> T {
        self.external[3]
    }

    #[inline]
    fn sgn_tau_rise(&self) -> T {
        self.internal[3].signum()
    }

    #[inline]
    fn tau_fall(&self) -> T {
        self.external[4]
    }

    #[inline]
    fn sgn_tau_fall(&self) -> T {
        self.internal[4].signum()
    }
}

impl<T, U> FitModelTrait<T, U> for BazinFit
where
    T: Float + Into<U>,
    U: LikeFloat,
{
    fn model(t: T, param: &[U]) -> U
    where
        T: Float + Into<U>,
        U: LikeFloat,
    {
        let t: U = t.into();
        let x = param
            .try_into()
            .expect("input param slice has wrong length");
        let x = Params {
            internal: &x,
            external: Self::internal_to_dimensionless(&x),
        };
        let minus_dt = x.t0() - t;
        x.b()
            + x.a() * U::exp(minus_dt / x.tau_fall()) / (U::exp(minus_dt / x.tau_rise()) + U::one())
    }
}

impl<T> FitFunctionTrait<T, NPARAMS> for BazinFit where T: Float {}

impl<T> FitDerivalivesTrait<T> for BazinFit
where
    T: Float,
{
    fn derivatives(t: T, param: &[T], jac: &mut [T]) {
        let x = param
            .try_into()
            .expect("input param slice has wrong length");
        let x = Params {
            internal: &x,
            external: Self::internal_to_dimensionless(&x),
        };
        let minus_dt = x.t0() - t;
        let exp_rise = T::exp(minus_dt / x.tau_rise());
        let frac = T::exp(minus_dt / x.tau_fall()) / (T::one() + exp_rise);
        let exp_1p_exp_rise = (T::one() + exp_rise.recip()).recip();
        // a
        jac[0] = x.sgn_a() * frac;
        // b
        jac[1] = T::one();
        // t0
        jac[2] = x.a() * frac * (x.tau_fall().recip() - exp_1p_exp_rise / x.tau_rise());
        // tau_rise
        jac[3] =
            x.sgn_tau_rise() * x.a() * minus_dt * frac / x.tau_rise().powi(2) * exp_1p_exp_rise;
        // tau_fall
        jac[4] = -x.sgn_tau_fall() * x.a() * minus_dt * frac / x.tau_fall().powi(2);
    }
}

impl<T> FitInitsBoundsTrait<T, NPARAMS> for BazinFit
where
    T: Float,
{
    fn init_and_bounds_from_ts(ts: &mut TimeSeries<T>) -> ([f64; NPARAMS], [(f64, f64); NPARAMS]) {
        let t_min: f64 = ts.t.get_min().value_into().unwrap();
        let t_max: f64 = ts.t.get_max().value_into().unwrap();
        let t_amplitude = t_max - t_min;
        let t_peak: f64 = ts.get_t_max_m().value_into().unwrap();
        let m_min: f64 = ts.m.get_min().value_into().unwrap();
        let m_max: f64 = ts.m.get_max().value_into().unwrap();
        let m_amplitude = m_max - m_min;

        let a_init = 0.5 * m_amplitude;
        let a_bound = (0.0, 100.0 * m_amplitude);

        let b_init = m_min;
        let b_bound = (m_min - 100.0 * m_amplitude, m_max + 100.0 * m_amplitude);

        let t0_init = t_peak;
        let t0_bound = (t_min - 10.0 * t_amplitude, t_max + 10.0 * t_amplitude);

        let rise_init = 0.5 * t_amplitude;
        let rise_bound = (0.0, 10.0 * t_amplitude);

        let fall_init = 0.5 * t_amplitude;
        let fall_bound = (0.0, 10.0 * t_amplitude);
        (
            [a_init, b_init, t0_init, rise_init, fall_init],
            [a_bound, b_bound, t0_bound, rise_bound, fall_bound],
        )
    }
}

impl FitParametersOriginalDimLessTrait<NPARAMS> for BazinFit {
    fn orig_to_dimensionless(
        norm_data: &NormalizedData<f64>,
        orig: &[f64; NPARAMS],
    ) -> [f64; NPARAMS] {
        [
            norm_data.m_to_norm_scale(orig[0]), // A amplitude
            norm_data.m_to_norm(orig[1]),       // c baseline
            norm_data.t_to_norm(orig[2]),       // t_0 reference_time
            norm_data.t_to_norm_scale(orig[3]), // tau_rise rise time
            norm_data.t_to_norm_scale(orig[4]), // tau_fall fall time
        ]
    }

    fn dimensionless_to_orig(
        norm_data: &NormalizedData<f64>,
        norm: &[f64; NPARAMS],
    ) -> [f64; NPARAMS] {
        [
            norm_data.m_to_orig_scale(norm[0]), // A amplitude
            norm_data.m_to_orig(norm[1]),       // c baseline
            norm_data.t_to_orig(norm[2]),       // t_0 reference_time
            norm_data.t_to_orig_scale(norm[3]), // tau_rise rise time
            norm_data.t_to_orig_scale(norm[4]), // tau_fall fall time
        ]
    }
}

impl<U> FitParametersInternalDimlessTrait<U, NPARAMS> for BazinFit
where
    U: LikeFloat,
{
    fn dimensionless_to_internal(params: &[U; NPARAMS]) -> [U; NPARAMS] {
        *params
    }

    fn internal_to_dimensionless(params: &[U; NPARAMS]) -> [U; NPARAMS] {
        [
            params[0].abs(),
            params[1],
            params[2],
            params[3].abs(),
            params[4].abs(),
        ]
    }
}

impl FitParametersInternalExternalTrait<NPARAMS> for BazinFit {}

impl FitFeatureEvaluatorGettersTrait for BazinFit {
    fn get_algorithm(&self) -> &CurveFitAlgorithm {
        &self.algorithm
    }

    fn get_ln_prior(&self) -> &LnPrior {
        &self.ln_prior
    }
}

impl FeatureNamesDescriptionsTrait for BazinFit {
    fn get_names(&self) -> Vec<&str> {
        vec![
            "bazin_fit_amplitude",
            "bazin_fit_baseline",
            "bazin_fit_reference_time",
            "bazin_fit_rise_time",
            "bazin_fit_fall_time",
            "bazin_fit_reduced_chi2",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "half amplitude of the Bazin function (A)",
            "baseline of the Bazin function (B)",
            "reference time of the Bazin fit (t0)",
            "rise time of the Bazin function (tau_rise)",
            "fall time of the Bazin function (tau_fall)",
            "Bazin fit quality (reduced chi2)",
        ]
    }
}

impl<T> FeatureEvaluator<T> for BazinFit
where
    T: Float,
{
    fit_eval!();
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;
    use crate::LmsderCurveFit;
    use crate::TimeSeries;

    use approx::assert_relative_eq;
    use hyperdual::Hyperdual;

    check_feature!(BazinFit);

    feature_test!(
        bazin_fit_plateau,
        [BazinFit::default()],
        [0.0, 0.0, 10.0, 5.0, 5.0, 0.0], // initial model parameters and zero chi2
        linspace(0.0, 10.0, 11),
        [0.0; 11],
    );

    fn bazin_fit_noisy(eval: BazinFit) {
        const N: usize = 50;

        let mut rng = StdRng::seed_from_u64(0);

        let param_true = [1e4, 1e3, 30.0, 10.0, 30.0];

        let t = linspace(0.0, 100.0, N);
        let model: Vec<_> = t.iter().map(|&x| BazinFit::model(x, &param_true)).collect();
        let m: Vec<_> = model
            .iter()
            .map(|&y| {
                let std = f64::sqrt(y);
                let err = std * rng.sample::<f64, _>(StandardNormal);
                y + err
            })
            .collect();
        let w: Vec<_> = model.iter().copied().map(f64::recip).collect();
        println!("{:?}\n{:?}\n{:?}\n{:?}", t, model, m, w);
        let mut ts = TimeSeries::new(&t, &m, &w);

        // curve_fit(lambda t, a, b, t0, rise, fall: b + a * np.exp(-(t-t0)/fall) / (1 + np.exp(-(t-t0) / rise)), xdata=t, ydata=m, sigma=np.array(w)**-0.5, p0=[1e4, 1e3, 30, 10, 30])
        let desired = [
            9.89658673e+03,
            1.11312724e+03,
            3.06401284e+01,
            9.75027284e+00,
            2.86714363e+01,
        ];

        let values = eval.eval(&mut ts).unwrap();
        assert_relative_eq!(&values[..5], &desired[..], max_relative = 0.01);
    }

    #[test]
    fn bazin_fit_noisy_lmsder() {
        bazin_fit_noisy(BazinFit::new(
            LmsderCurveFit::new(9).into(),
            LnPrior::none(),
        ));
    }

    #[test]
    fn bazin_fit_noizy_mcmc_plus_lmsder() {
        let lmsder = LmsderCurveFit::new(8);
        let mcmc = McmcCurveFit::new(128, Some(lmsder.into()));
        bazin_fit_noisy(BazinFit::new(mcmc.into(), LnPrior::none()));
    }

    #[test]
    fn bazin_fit_derivatives() {
        const REPEAT: usize = 10;

        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..REPEAT {
            let t = 10.0 * rng.gen::<f64>();

            let param: Vec<_> = (0..5).map(|_| rng.gen::<f64>() - 0.5).collect();
            let actual = {
                let mut jac = [0.0; 5];
                BazinFit::derivatives(t, &param, &mut jac);
                jac
            };

            let desired: Vec<_> = {
                let param: Vec<Hyperdual<f64, 6>> = param
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let mut x = Hyperdual::from_real(x);
                        x[i + 1] = 1.0;
                        x
                    })
                    .collect();
                let result = BazinFit::model(t, &param);
                (1..=5).map(|i| result[i]).collect()
            };

            assert_relative_eq!(&actual[..], &desired[..], epsilon = 1e-9);
        }
    }
}
