use crate::evaluator::*;
use crate::nl_fit::{
    data::NormalizedData, CurveFitAlgorithm, CurveFitResult, CurveFitTrait, F64LikeFloat,
    McmcCurveFit,
};

use conv::ConvUtil;

macro_const! {
    const DOC: &str = r#"
Villar function fit

Seven fit parameters and goodness of fit (reduced $\chi^2$) of the Villar function developed for
supernovae classification:

<span>
$$
f(t) = c + \frac{A}{ 1 + \exp{\frac{-(t - t_0)}{\tau_\mathrm{rise}}}}  \left\{ \begin{array}{ll} 1 - \frac{\nu (t - t_0)}{\gamma}, &t < t_0 + \gamma \\ (1 - \nu) \exp{\frac{-(t-t_0-\gamma)}{\tau_\mathrm{fall}}}, &t \geq t_0 + \gamma \end{array} \right.
$$
</span>
where $A, \gamma, \tau_\mathrm{rise}, \tau_\mathrm{fall} > 0$, $\nu \in [0; 1)$.

Here we introduce a new dimensionless parameter $\nu$ instead of the plateau slope $\beta$ from the
orioginal paper: $\nu \equiv -\beta \gamma / A$.

Note, that the Villar function is developed to be used with fluxes, not magnitudes.

- Depends on: **time**, **magnitude**, **magnitude error**
- Minimum number of observations: **8**
- Number of features: **8**

Villar et al. 2019 [DOI:10.3847/1538-4357/ab418c](https://doi.org/10.3847/1538-4357/ab418c)
"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct VillarFit {
    algorithm: CurveFitAlgorithm,
}

impl VillarFit {
    /// New [VillarFit] instance
    ///
    /// `algorithm` specifies which optimization method is used, it is an instance of the
    /// [CurveFitAlgorithm], currently supported algorithms are [MCMC](McmcCurveFit) and
    /// [LMSDER](crate::nl_fit::LmsderCurveFit) (a Levenberg–Marquard algorithm modification,
    /// requires `gsl` Cargo feature).
    pub fn new(algorithm: CurveFitAlgorithm) -> Self {
        Self { algorithm }
    }

    /// [VillarFit] with the default [McmcCurveFit]
    #[inline]
    pub fn default_algorithm() -> CurveFitAlgorithm {
        McmcCurveFit::new(McmcCurveFit::default_niterations(), None).into()
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

impl Default for VillarFit {
    fn default() -> Self {
        Self::new(Self::default_algorithm())
    }
}

lazy_info!(
    VILLAR_FIT_INFO,
    size: 8,
    min_ts_length: 8,
    t_required: true,
    m_required: true,
    w_required: true,
    sorting_required: true, // improve reproducibility
);

impl VillarFit {
    pub fn f<T>(t: T, values: &[T]) -> T
    where
        T: Float,
    {
        let mut param = values[..7].to_owned();
        // beta to b
        param[5] = T::half() * (T::ln_1p(param[5]) - T::ln(T::one() - param[5]));
        Self::model(t, &param)
    }

    fn model<T, U>(t: T, param: &[U]) -> U
    where
        T: Into<U>,
        U: F64LikeFloat,
    {
        let t: U = t.into();
        let x = Params { storage: param };
        x.c() + x.a() * x.rise(t) * x.plateau(t) * x.fall(t)
    }

    fn derivatives<T>(t: T, param: &[T], jac: &mut [T])
    where
        T: Float,
    {
        let x = Params { storage: param };
        let dt = x.dt(t);
        let t1 = x.t1();
        let nu = x.nu();
        let plateau = x.plateau(t);
        let rise = x.rise(t);
        let fall = x.fall(t);
        let is_rise = t <= t1;
        let f_minus_c = x.a() * plateau * rise * fall;

        // A
        jac[0] = x.sgn_a() * plateau * rise * fall;
        // c
        jac[1] = T::one();
        // t0
        jac[2] = x.a()
            * rise
            * fall
            * (-(T::one() - rise) * plateau / x.tau_rise()
                + if is_rise {
                    nu / x.gamma()
                } else {
                    plateau / x.tau_fall()
                });
        // tau_rise
        jac[3] = -x.sgn_tau_rise() * f_minus_c * (T::one() - rise) * dt / x.tau_rise().powi(2);
        // tau_fall
        jac[4] = if is_rise {
            T::zero()
        } else {
            x.sgn_tau_fall() * f_minus_c * (dt - x.gamma()) / x.tau_fall().powi(2)
        };
        // b = 1/2 * ln [(1 - nu) / (1 + nu)]
        jac[5] = -x.sgn_b()
            * (T::one() - nu.powi(2))
            * x.a()
            * rise
            * fall
            * if is_rise { dt / x.gamma() } else { T::one() };
        // gamma
        jac[6] = x.sgn_gamma()
            * if is_rise {
                x.a() * rise * fall * nu * dt / x.gamma().powi(2)
            } else {
                f_minus_c / x.tau_fall()
            };
    }

    fn init_and_bounds_from_ts<T: Float>(ts: &mut TimeSeries<T>) -> ([f64; 7], [(f64, f64); 7]) {
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

        // t0 is not a peak time, but something before the peak
        let t0_init = t_peak;
        let t0_bound = (t_min - 20.0 * t_amplitude, t_max + 10.0 * t_amplitude);

        let tau_rise_init = 0.5 * t_amplitude;
        let tau_rise_bound = (0.0, 10.0 * t_amplitude);

        let tau_fall_init = 0.5 * t_amplitude;
        let tau_fall_bound = (0.0, 10.0 * t_amplitude);

        let nu_init = 0.0;
        let nu_bound = (0.0, 1.0);

        let gamma_init = 0.1 * t_amplitude;
        let gamma_bound = (0.0, 10.0 * t_amplitude);

        (
            [
                a_init,
                b_init,
                t0_init,
                tau_rise_init,
                tau_fall_init,
                nu_init,
                gamma_init,
            ],
            [
                a_bound,
                b_bound,
                t0_bound,
                tau_rise_bound,
                tau_fall_bound,
                nu_bound,
                gamma_bound,
            ],
        )
    }
}

impl<T> FeatureEvaluator<T> for VillarFit
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;

        let norm_data = NormalizedData::<f64>::from_ts(ts);

        let (x0, bound) = {
            let (mut x0, mut bound) = Self::init_and_bounds_from_ts(ts);

            // A amplitude
            x0[0] = norm_data.m_to_norm_scale(x0[0]);
            bound[0].0 = norm_data.m_to_norm_scale(bound[0].0);
            bound[0].1 = norm_data.m_to_norm_scale(bound[0].1);

            // c baseline
            x0[1] = norm_data.m_to_norm(x0[1]);
            bound[1].0 = norm_data.m_to_norm(bound[1].0);
            bound[1].1 = norm_data.m_to_norm(bound[1].1);

            // t_0 reference time
            x0[2] = norm_data.t_to_norm(x0[2]);
            bound[2].0 = norm_data.t_to_norm(bound[2].0);
            bound[2].1 = norm_data.t_to_norm(bound[2].1);

            // tau_rise rise time
            x0[3] = norm_data.t_to_norm_scale(x0[3]);
            bound[3].0 = norm_data.t_to_norm_scale(bound[3].0);
            bound[3].1 = norm_data.t_to_norm_scale(bound[3].1);

            // tau_fall fall time
            x0[4] = norm_data.t_to_norm_scale(x0[4]);
            bound[4].0 = norm_data.t_to_norm_scale(bound[4].0);
            bound[4].1 = norm_data.t_to_norm_scale(bound[4].1);

            // nu = beta gamma / A relative plateau amplitude
            // dimensionless, do nothing

            // gamma plateau duration
            x0[6] = norm_data.t_to_norm_scale(x0[6]);
            bound[6].0 = norm_data.t_to_norm_scale(bound[6].0);
            bound[6].1 = norm_data.t_to_norm_scale(bound[6].1);

            (x0, bound)
        };

        let result = {
            let CurveFitResult {
                x, reduced_chi2, ..
            } = self.algorithm.curve_fit(
                norm_data.data.clone(),
                &x0,
                &bound,
                Self::model::<f64, f64>,
                Self::derivatives::<f64>,
            );
            let x = Params { storage: &x };
            let mut result = vec![0.0; 8];
            result[0] = norm_data.m_to_orig_scale(x.a()); // A amplitude
            result[1] = norm_data.m_to_orig(x.c()); // c offset
            result[2] = norm_data.t_to_orig(x.t0()); // t_0 reference time
            result[3] = norm_data.t_to_orig_scale(x.tau_rise()); // tau_rise rise time
            result[4] = norm_data.t_to_orig_scale(x.tau_fall()); // tau_fall fall time
            result[5] = x.nu(); // nu relative plateau amplitude
            result[6] = norm_data.t_to_orig_scale(x.gamma()); // gamma plateau duration
            result[7] = reduced_chi2;
            result
                .iter()
                .map(|&x| x.approx_as::<T>().unwrap())
                .collect()
        };

        Ok(result)
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &VILLAR_FIT_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec![
            "villar_fit_amplitude",
            "villar_fit_baseline",
            "villar_fit_reference_time",
            "villar_fit_rise_time",
            "villar_fit_fall_time",
            "villar_fit_plateau_rel_amplitude",
            "villar_fit_plateau_duration",
            "villar_fit_reduced_chi2",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "half amplitude of the Villar function (A)",
            "baseline of the Villar function (c)",
            "reference time of the Villar function (t_0)",
            "rise time of the Villar function (tau_rise)",
            "decline time of the Villar function (tau_fall)",
            "relative plateau amplitude of the Villar function (nu = beta gamma / A)",
            "plateau duration of the Villar function (gamma)",
            "Villar fit quality (reduced chi2)",
        ]
    }
}

struct Params<'a, T> {
    storage: &'a [T],
}

impl<'a, T> Params<'a, T>
where
    T: F64LikeFloat,
{
    #[inline]
    fn a(&self) -> T {
        self.storage[0].abs()
    }

    #[inline]
    fn sgn_a(&self) -> T {
        self.storage[0].signum()
    }

    #[inline]
    fn c(&self) -> T {
        self.storage[1]
    }

    #[inline]
    fn t0(&self) -> T {
        self.storage[2]
    }

    #[inline]
    fn tau_rise(&self) -> T {
        self.storage[3].abs()
    }

    #[inline]
    fn sgn_tau_rise(&self) -> T {
        self.storage[3].signum()
    }

    #[inline]
    fn tau_fall(&self) -> T {
        self.storage[4].abs()
    }

    #[inline]
    fn sgn_tau_fall(&self) -> T {
        self.storage[4].signum()
    }

    /// $\nu = 2 logistic(2|b|) - 1$
    /// Properties:
    /// dnu/db = 4 logistic(2|b|) (1 - logistic(2|b|)) sgn(b) = (1 - nu^2) sgn(b)
    /// $\nu = 0$ <=> $b = 0$, for this point $d\nu/db = 1$
    /// $\nu = 1$ <=> $|b| = inf$
    #[inline]
    fn nu(&self) -> T {
        T::two() * T::logistic(T::two() * self.storage[5].abs()) - T::one()
    }

    #[inline]
    fn sgn_b(&self) -> T {
        self.storage[5].signum()
    }

    #[inline]
    fn gamma(&self) -> T {
        self.storage[6].abs()
    }

    #[inline]
    fn sgn_gamma(&self) -> T {
        self.storage[6].signum()
    }

    #[inline]
    fn t1(&self) -> T {
        self.t0() + self.gamma()
    }

    #[inline]
    fn dt(&self, t: T) -> T {
        t - self.t0()
    }

    #[inline]
    fn rise(&self, t: T) -> T {
        T::logistic(self.dt(t) / self.tau_rise())
    }

    #[inline]
    fn plateau(&self, t: T) -> T {
        T::one() - self.nu() * T::min(self.dt(t) / self.gamma(), T::one())
    }

    #[inline]
    fn fall(&self, t: T) -> T {
        let t1 = self.t1();
        if t <= t1 {
            T::one()
        } else {
            T::exp(-(t - t1) / self.tau_fall())
        }
    }
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

    check_feature!(VillarFit);

    feature_test!(
        villar_fit_plateau,
        [VillarFit::default()],
        [0.0, 0.0, 10.0, 5.0, 5.0, 0.0, 1.0, 0.0], // initial model parameters and zero chi2
        linspace(0.0, 10.0, 11),
        [0.0; 11],
    );

    fn villar_fit_noisy(eval: VillarFit) {
        const N: usize = 50;

        let mut rng = StdRng::seed_from_u64(0);

        let param_true = [1e4, 1e3, 20.0, 5.0, 30.0, 0.3, 30.0];

        let t = linspace(0.0, 100.0, N);
        let model: Vec<_> = t.iter().map(|&x| VillarFit::f(x, &param_true)).collect();
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

        // curve_fit(lambda t, a, c, t0, tau_rise, tau_fall, nu, gamma: c + a * (1 - nu * np.where(t - t0 < gamma, (t - t0) / gamma, 1)) / (1 + np.exp(-(t-t0) / tau_rise)) * np.where(t > t0 + gamma, np.exp(-(t-t0-gamma) / tau_fall), 1.0), xdata=t, ydata=m, sigma=np.array(w)**-0.5, p0=[1e4, 1e3, 20, 5, 30, 0.3, 30])
        let desired = [
            1.00899754e+04,
            1.00689083e+03,
            2.02385802e+01,
            5.04283765e+00,
            3.00679883e+01,
            3.00400783e-01,
            2.94149214e+01,
        ];

        let values = eval.eval(&mut ts).unwrap();
        assert_relative_eq!(&values[..7], &desired[..], max_relative = 0.01);
    }

    #[test]
    fn villar_fit_noisy_lmsder() {
        villar_fit_noisy(VillarFit::new(LmsderCurveFit::new(6).into()));
    }

    #[test]
    fn villar_fit_noizy_mcmc_plus_lmsder() {
        let lmsder = LmsderCurveFit::new(1);
        let mcmc = McmcCurveFit::new(2048, Some(lmsder.into()));
        villar_fit_noisy(VillarFit::new(mcmc.into()));
    }

    #[test]
    fn villar_fit_derivatives() {
        const REPEAT: usize = 10;

        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..REPEAT {
            let t = 10.0 * rng.gen::<f64>();

            let param: Vec<_> = (0..7).map(|_| rng.gen::<f64>() - 0.5).collect();
            println!("{:?}", param);
            let actual = {
                let mut jac = [0.0; 7];
                VillarFit::derivatives(t, &param, &mut jac);
                jac
            };

            let desired: Vec<_> = {
                let param: Vec<Hyperdual<f64, 8>> = param
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let mut x = Hyperdual::from_real(x);
                        x[i + 1] = 1.0;
                        x
                    })
                    .collect();
                let result = VillarFit::model(t, &param);
                (1..=7).map(|i| result[i]).collect()
            };

            assert_relative_eq!(&actual[..], &desired[..], epsilon = 1e-9);
        }
    }
}
