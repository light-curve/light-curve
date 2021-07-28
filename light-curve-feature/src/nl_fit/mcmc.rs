use crate::nl_fit::curve_fit::{CurveFitResult, CurveFitTrait};
use crate::nl_fit::data::Data;

use emcee::{EnsembleSampler, Guess, Prob};
use emcee_rand::*;
use ndarray::Zip;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::rc::Rc;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Mcmc")]
pub struct McmcCurveFit {}

impl CurveFitTrait for McmcCurveFit {
    fn curve_fit<F, DF>(
        &self,
        ts: Rc<Data<f64>>,
        x0: &[f64],
        bounds: &[(f64, f64)],
        model: F,
        _derivatives: DF,
    ) -> CurveFitResult<f64>
    where
        F: 'static + Clone + Fn(f64, &[f64]) -> f64,
        DF: 'static + Clone + Fn(f64, &[f64], &mut [f64]),
    {
        const NWALKERS_PER_DIMENSION: usize = 4;
        const NITERATIONS: usize = 128;
        let ndims = x0.len();
        let nwalkers = NWALKERS_PER_DIMENSION * ndims;
        let nsamples = ts.t.len();

        let lnlike = {
            move |params: &Guess| {
                let mut residual = 0.0;
                let params = params.values.iter().map(|&x| x as f64).collect::<Vec<_>>();
                Zip::from(&ts.t)
                    .and(&ts.m)
                    .and(&ts.inv_err)
                    .for_each(|&t, &m, &inv_err| {
                        residual += (inv_err * (model(t, &params) - m)).powi(2);
                    });
                -residual as f32
            }
        };

        let initial_guess = Guess::new(&x0.iter().map(|&x| x as f32).collect::<Vec<_>>());
        let initial_lnprob = lnlike(&initial_guess);
        let model = EmceeModel {
            func: lnlike,
            bounds: bounds
                .iter()
                .map(|&(lower, upper)| (lower as f32, upper as f32))
                .collect(),
        };
        let mut sampler = EnsembleSampler::new(nwalkers, ndims, &model).unwrap();
        sampler.seed(&[]);

        let (best_x, best_lnprob) = {
            let (mut best_x, mut best_lnprob) = (initial_guess.values.clone(), initial_lnprob);
            let initial_guesses =
                initial_guess.create_initial_guess_with_rng(nwalkers, &mut StdRng::from_seed(&[]));
            let _ = sampler.sample(&initial_guesses, NITERATIONS, |step| {
                for (pos, &lnprob) in step.pos.iter().zip(step.lnprob.iter()) {
                    if lnprob > best_lnprob {
                        best_x = pos.values.clone();
                        best_lnprob = lnprob;
                    }
                }
            });
            (best_x, best_lnprob)
        };

        CurveFitResult {
            x: best_x.into_iter().map(|x| x as f64).collect(),
            reduced_chi2: (-best_lnprob / ((nsamples - ndims) as f32)) as f64,
            success: true,
        }
    }
}

struct EmceeModel<F> {
    func: F,
    bounds: Vec<(f32, f32)>,
}

impl<F> Prob for EmceeModel<F>
where
    F: Fn(&Guess) -> f32,
{
    fn lnlike(&self, params: &Guess) -> f32 {
        (self.func)(params)
    }

    fn lnprior(&self, params: &Guess) -> f32 {
        for (&p, &(lower, upper)) in params.values.iter().zip(self.bounds.iter()) {
            if p < lower || p > upper {
                return -f32::INFINITY;
            }
        }
        0.0
    }
}
