use crate::float_trait::Float;
use crate::nl_fit::curve_fit::{CurveFitAlgorithm, CurveFitResult, CurveFitTrait};
use crate::nl_fit::data::Data;

use emcee::{EnsembleSampler, Guess, Prob};
use emcee_rand::{distributions::IndependentSample, *};
use itertools::Itertools;
use ndarray::Zip;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::rc::Rc;

/// MCMC sampler for non-linear least squares
///
/// First it generates `4 * dimension_number` walkers from the given initial guess using the
/// Standard distribution with `sigma = 0.1`, next it samples `niterations` guesses for each walker
/// and chooses guess corresponding to the minimum sum of squared deviations (maximum likelihood).
/// Optionally, if `fine_tuning_algorithm` is `Some`, it sends this best guess to the next
/// optimization as an initial guess and returns its result
///
/// This method supports both boundaries and priors while doesn't use the function derivatives if
/// not required by `fine_tuning_algorithm`
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Mcmc")]
pub struct McmcCurveFit {
    pub niterations: u32,
    pub fine_tuning_algorithm: Option<Box<CurveFitAlgorithm>>,
}

impl McmcCurveFit {
    pub fn new(niterations: u32, fine_tuning_algorithm: Option<CurveFitAlgorithm>) -> Self {
        Self {
            niterations,
            fine_tuning_algorithm: fine_tuning_algorithm.map(|x| x.into()),
        }
    }

    #[inline]
    pub fn default_niterations() -> u32 {
        128
    }

    #[inline]
    pub fn default_fine_tuning_algorithm() -> Option<CurveFitAlgorithm> {
        None
    }
}

impl Default for McmcCurveFit {
    fn default() -> Self {
        Self::new(
            Self::default_niterations(),
            Self::default_fine_tuning_algorithm(),
        )
    }
}

impl CurveFitTrait for McmcCurveFit {
    fn curve_fit<F, DF, LP, const NPARAMS: usize>(
        &self,
        ts: Rc<Data<f64>>,
        x0: &[f64; NPARAMS],
        bounds: &[(f64, f64); NPARAMS],
        model: F,
        derivatives: DF,
        ln_prior: LP,
    ) -> CurveFitResult<f64>
    where
        F: 'static + Clone + Fn(f64, &[f64; NPARAMS]) -> f64,
        DF: 'static + Clone + Fn(f64, &[f64; NPARAMS], &mut [f64; NPARAMS]),
        LP: Clone + Fn(&[f64; NPARAMS]) -> f64,
    {
        const NWALKERS_PER_DIMENSION: usize = 4;
        let nwalkers = NWALKERS_PER_DIMENSION * NPARAMS;
        let nsamples = ts.t.len();

        let lnlike = {
            let ts = ts.clone();
            let model = model.clone();
            move |guess: &Guess| {
                let mut residual = 0.0;
                let params = slice_to_array(&guess.values);
                Zip::from(&ts.t)
                    .and(&ts.m)
                    .and(&ts.inv_err)
                    .for_each(|&t, &m, &inv_err| {
                        residual += (inv_err * (model(t, &params) - m)).powi(2);
                    });
                -0.5 * residual as f32
            }
        };
        let lnprior = {
            let ln_prior = ln_prior.clone();
            move |guess: &Guess| {
                let params = slice_to_array(&guess.values);
                ln_prior(&params) as f32
            }
        };

        let x0_f32 = slice_to_array(x0);
        let bounds_f32 = {
            let mut array = [(0.0, 0.0); NPARAMS];
            for (input, output) in bounds.iter().zip_eq(array.iter_mut()) {
                *output = (input.0 as f32, input.1 as f32)
            }
            array
        };

        let initial_guesses =
            generate_initial_guesses(nwalkers, &x0_f32, &bounds_f32, &mut StdRng::from_seed(&[]));
        let initial_lnprob = lnlike(&initial_guesses[0]);
        let emcee_model = EmceeModel {
            ln_like: lnlike,
            ln_prior: lnprior,
            bounds: &bounds_f32,
        };
        let mut sampler = EnsembleSampler::new(nwalkers, NPARAMS, &emcee_model).unwrap();
        sampler.seed(&[]);

        let (best_x, best_lnprob) = {
            let (mut best_x, mut best_lnprob) = (initial_guesses[0].values.clone(), initial_lnprob);
            let _ = sampler.sample(&initial_guesses, self.niterations as usize, |step| {
                for (pos, &lnprob) in step.pos.iter().zip(step.lnprob.iter()) {
                    if lnprob > best_lnprob {
                        best_x = pos.values.clone();
                        best_lnprob = lnprob;
                    }
                }
            });
            (
                best_x.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
                best_lnprob as f64,
            )
        };

        match self.fine_tuning_algorithm.as_ref() {
            Some(algo) => algo.curve_fit(
                ts,
                &best_x.try_into().unwrap(),
                bounds,
                model,
                derivatives,
                ln_prior,
            ),
            None => CurveFitResult {
                x: best_x,
                reduced_chi2: -best_lnprob / ((nsamples - NPARAMS) as f64),
                success: true,
            },
        }
    }
}

#[inline]
fn slice_to_array<T, U, const NPARAMS: usize>(sl: &[T]) -> [U; NPARAMS]
where
    T: Float,
    U: Float,
{
    let mut array = [U::zero(); NPARAMS];
    for (input, output) in sl.iter().zip_eq(array.iter_mut()) {
        *output = num_traits::cast::cast(*input).unwrap();
    }
    array
}

#[inline]
fn generate_initial_guesses<R, const NPARAMS: usize>(
    nwalkers: usize,
    x0: &[f32; NPARAMS],
    bounds: &[(f32, f32); NPARAMS],
    rng: &mut R,
) -> Vec<Guess>
where
    R: Rng,
{
    const STD: f64 = 0.1;

    // First guess is the user-defined initial guess
    std::iter::once(x0.to_vec())
        // Next nwalkers-1 guesses
        .chain((1..nwalkers).map(|_| {
            // Iterate components of user-defined initial guess and boundary conditions
            x0.iter()
                .zip(bounds.iter())
                .map(|(component, (left, right))| {
                    assert!(
                        *left <= *right,
                        "Left boundary is larger than right one: {} > {}",
                        *left,
                        *right,
                    );
                    assert!(
                        (*left <= *component) && (*component <= *right),
                        "Initial guess is not between boundaries: {} not in [{}, {}]",
                        *component,
                        *left,
                        *right,
                    );
                    if (right - left) < f32::EPSILON {
                        return *component;
                    }
                    let std = f64::min(STD, (*right - *left) as f64);
                    let normal_distr = distributions::Normal::new((*component) as f64, std);
                    loop {
                        let sample = normal_distr.ind_sample(rng) as f32;
                        if (*left < sample) && (sample < *right) {
                            break sample;
                        }
                    }
                })
                .collect()
        }))
        .map(|v| Guess { values: v })
        .collect()
}

struct EmceeModel<'b, F, LP> {
    ln_like: F,
    ln_prior: LP,
    bounds: &'b [(f32, f32)],
}

impl<'b, F, LP> Prob for EmceeModel<'b, F, LP>
where
    F: Fn(&Guess) -> f32,
    LP: Fn(&Guess) -> f32,
{
    fn lnlike(&self, params: &Guess) -> f32 {
        (self.ln_like)(params)
    }

    fn lnprior(&self, params: &Guess) -> f32 {
        for (&p, &(lower, upper)) in params.values.iter().zip(self.bounds.iter()) {
            if p < lower || p > upper {
                return f32::NEG_INFINITY;
            }
        }
        (self.ln_prior)(params)
    }
}
