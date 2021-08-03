use crate::evaluator::*;
use crate::nl_fit::{
    data::NormalizedData, CurveFitAlgorithm, CurveFitResult, CurveFitTrait, F64LikeFloat,
    McmcCurveFit,
};

use conv::ConvUtil;

/// Villar fit
///
/// Five fit parameters and goodness of fit (reduced $\Chi^2$) of the Villar function developed for
/// supernovae classification:
/// $$
/// f(t) = c + \\frac{ A + \\beta (t - t_0) }{ 1 + \\exp{\\frac{-(t - t_0)}{\\tau_\\mathrm{rise}}}}  \\left\\{ \\begin{split} &1, t < t_0 + \\gamma \\\\ &\\exp{\\frac{-(t-t_0-\\gamma)}{\\tau_\\mathrm{fall}}}, t \\geq t_0 + \\gamma \\end{split} \\right.
/// $$
///
/// Note, that the Villar function is developed to use with fluxes, not magnitudes.
///
/// Optimization is done using specified `algorithm` which is an instance of the
/// [CurveFitAlgorithm], currently supported algorithms are [MCMC](McmcCurveFit) and
/// [LMSDER](crate::nl_fit::LmsderCurveFit) (a Levenberg–Marquard algorithm modification, requires
/// `gsl` Cargo feature).
///
/// - Depends on: **time**, **magnitude**, **magnitude error**
/// - Minimum number of observations: **8**
/// - Number of features: **8**
///
/// Villar et al. 2019 [DOI:10.3847/1538-4357/ab418c](https://doi.org/10.3847/1538-4357/ab418c)
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct VillarFit {
    algorithm: CurveFitAlgorithm,
}

impl VillarFit {
    pub fn new(algorithm: CurveFitAlgorithm) -> Self {
        Self { algorithm }
    }

    /// [VillarFit] with the default [McmcCurveFit]
    #[inline]
    pub fn default_algorithm() -> CurveFitAlgorithm {
        McmcCurveFit::new(McmcCurveFit::default_niterations(), None).into()
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
    fn model<T, U>(t: T, param: &[U]) -> U
    where
        T: Into<U>,
        U: F64LikeFloat,
    {
        let t: U = t.into();
        let x = Params { storage: param };
        let dt = x.dt(t);
        let t1 = x.t1();
        let mut f = (x.a() + x.beta() * dt) / (U::one() + U::exp(-dt / x.tau_rise()));
        if t > t1 {
            f *= U::exp(-(t - t1) / x.tau_fall());
        }
        f += x.c();
        f
    }

    fn derivatives<T>(t: T, param: &[T], jac: &mut [T])
    where
        T: Float,
    {
        let x = Params { storage: param };
        let dt = x.dt(t);
        let t1 = x.t1();
        let exp_rise = T::exp(-dt / x.tau_rise());
        let rise = T::recip(T::one() + exp_rise);
        let is_fall = t > t1;
        let fall = if is_fall {
            T::exp(-(t - t1) / x.tau_fall())
        } else {
            T::one()
        };
        let plateau = x.a() + x.beta() * dt;
        let f_minus_c = plateau * rise * fall;

        // A
        jac[0] = rise * fall;
        // c
        jac[1] = T::one();
        // t0
        jac[2] = {
            let mut df_dt0 = -x.beta() * rise - plateau * rise.powi(2) * exp_rise / x.tau_rise();
            if is_fall {
                df_dt0 = df_dt0 * fall + f_minus_c / x.tau_fall();
            }
            df_dt0
        };
        // tau_rise
        jac[3] = -f_minus_c * rise * dt * exp_rise / x.tau_rise().powi(2);
        // tau_fall
        jac[4] = if is_fall {
            f_minus_c * (dt - x.gamma()) / x.tau_fall().powi(2)
        } else {
            T::zero()
        };
        // beta
        jac[5] = dt * rise * fall;
        // gamma
        jac[6] = if is_fall {
            f_minus_c / x.tau_fall()
        } else {
            T::zero()
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

        let t0_init = t_peak;
        let t0_bound = (t_min - 10.0 * t_amplitude, t_max + 10.0 * t_amplitude);

        let tau_rise_init = 0.5 * t_amplitude;
        let tau_rise_bound = (0.0, 10.0 * t_amplitude);

        let tau_fall_init = 0.5 * t_amplitude;
        let tau_fall_bound = (0.0, 10.0 * t_amplitude);

        let beta_init = -0.1 * m_amplitude / t_amplitude;
        let beta_bound = (-100.0 * m_amplitude / t_amplitude, 0.0);

        let gamma_init = 0.1 * t_amplitude;
        let gamma_bound = (0.0, 10.0 * t_amplitude);

        (
            [
                a_init,
                b_init,
                t0_init,
                tau_rise_init,
                tau_fall_init,
                beta_init,
                gamma_init,
            ],
            [
                a_bound,
                b_bound,
                t0_bound,
                tau_rise_bound,
                tau_fall_bound,
                beta_bound,
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

            // amplitude
            x0[0] = norm_data.m_to_norm_scale(x0[0]);
            bound[0].0 = norm_data.m_to_norm_scale(bound[0].0);
            bound[0].1 = norm_data.m_to_norm_scale(bound[0].1);

            // baseline
            x0[1] = norm_data.m_to_norm(x0[1]);
            bound[1].0 = norm_data.m_to_norm(bound[1].0);
            bound[1].1 = norm_data.m_to_norm(bound[1].1);

            // peak time
            x0[2] = norm_data.t_to_norm(x0[2]);
            bound[2].0 = norm_data.t_to_norm(bound[2].0);
            bound[2].1 = norm_data.t_to_norm(bound[2].1);

            // rise time
            x0[3] = norm_data.t_to_norm_scale(x0[3]);
            bound[3].0 = norm_data.t_to_norm_scale(bound[3].0);
            bound[3].1 = norm_data.t_to_norm_scale(bound[3].1);

            // fall time
            x0[4] = norm_data.t_to_norm_scale(x0[4]);
            bound[4].0 = norm_data.t_to_norm_scale(bound[4].0);
            bound[4].1 = norm_data.t_to_norm_scale(bound[4].1);

            // plateau slope
            x0[5] = norm_data.slope_to_norm(x0[5]);
            bound[5].0 = norm_data.slope_to_norm(x0[5]);
            bound[5].1 = norm_data.slope_to_norm(x0[5]);

            // plateau duration
            x0[6] = norm_data.t_to_norm_scale(x0[6]);
            bound[6].0 = norm_data.t_to_norm_scale(bound[6].0);
            bound[6].1 = norm_data.t_to_norm_scale(bound[6].1);

            (x0, bound)
        };

        let result = {
            let CurveFitResult {
                mut x,
                reduced_chi2,
                ..
            } = self.algorithm.curve_fit(
                norm_data.data.clone(),
                &x0,
                &bound,
                Self::model::<f64, f64>,
                Self::derivatives::<f64>,
            );
            x[0] = norm_data.m_to_orig_scale(x[0]); // amplitude
            x[1] = norm_data.m_to_orig(x[1]); // offset
            x[2] = norm_data.t_to_orig(x[2]); // peak time
            x[3] = norm_data.t_to_orig_scale(x[3]); // rise time
            x[4] = norm_data.t_to_orig_scale(x[4]); // fall time
            x[5] = norm_data.slope_to_orig(x[5]); // plateau slope
            x[6] = norm_data.t_to_orig_scale(x[6]); // plateau duration
            x.push(reduced_chi2);
            x.iter().map(|&x| x.approx_as::<T>().unwrap()).collect()
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
            "villar_fit_peak_time",
            "villar_fit_rise_time",
            "villar_fit_fall_time",
            "villar_fit_plateau_slope",
            "villar_fit_plateau_duration",
            "villar_fit_reduced_chi2",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "half amplitude of the Villar function (A)",
            "baseline of the Villar function (c)",
            "peak time of the Villar function (t_0)",
            "rise time of the Villar function (tau_rise)",
            "decline time of the Villar function (tau_fall)",
            "plateau slope of the Villar function (beta)",
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
    T: Copy + std::ops::Add<T, Output = T> + std::ops::Sub<T, Output = T>,
{
    #[inline]
    fn a(&self) -> T {
        self.storage[0]
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
        self.storage[3]
    }

    #[inline]
    fn tau_fall(&self) -> T {
        self.storage[4]
    }

    #[inline]
    fn beta(&self) -> T {
        self.storage[5]
    }

    #[inline]
    fn gamma(&self) -> T {
        self.storage[6]
    }

    #[inline]
    fn t1(&self) -> T {
        self.t0() + self.gamma()
    }

    #[inline]
    fn dt(&self, t: T) -> T {
        t - self.t0()
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
    use hyperdual::{Hyperdual, U8};

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

        let param_true = [1e4, 1e3, 30.0, 10.0, 30.0, -2e3 / 20.0, 20.0];

        let t = linspace(0.0, 100.0, N);
        let model: Vec<_> = t
            .iter()
            .map(|&x| VillarFit::model(x, &param_true))
            .collect();
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

        // curve_fit(lambda t, a, c, t0, tau_rise, tau_fall, beta, gamma: c + (a + beta * (t - t0)) / (1 + np.exp(-(t-t0) / tau_rise)) * np.where(t > t0 + gamma, np.exp(-(t-t0-gamma) / tau_fall), 1.0), xdata=t, ydata=m, sigma=np.array(w)**-0.5, p0=[1e4, 1e3, 30, 10, 30, -2e3/20, 20])
        let desired = [
            9.96814599e+03,
            1.07192395e+03,
            3.02341667e+01,
            9.71799521e+00,
            3.00318779e+01,
            -1.03862245e+02,
            1.96983192e+01,
        ];

        let values = eval.eval(&mut ts).unwrap();
        assert_relative_eq!(&values[..7], &desired[..], max_relative = 0.01);
    }

    #[test]
    fn villar_fit_noisy_lmsder() {
        villar_fit_noisy(VillarFit::new(LmsderCurveFit::new(7).into()));
    }

    #[test]
    fn villar_fit_noizy_mcmc_plus_lmsder() {
        let lmsder = LmsderCurveFit::new(7);
        let mcmc = McmcCurveFit::new(128, Some(lmsder.into()));
        villar_fit_noisy(VillarFit::new(mcmc.into()));
    }

    #[test]
    fn villar_fit_derivatives() {
        const REPEAT: usize = 10;

        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..REPEAT {
            let t = 10.0 * rng.gen::<f64>();

            let param: Vec<_> = (0..7).map(|_| rng.gen::<f64>()).collect();
            println!("{:?}", param);
            let actual = {
                let mut jac = [0.0; 7];
                VillarFit::derivatives(t, &param, &mut jac);
                jac
            };

            let desired: Vec<_> = {
                let param: Vec<Hyperdual<f64, U8>> = param
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
