use crate::nl_fit::prior::ln_prior_1d::{LnPrior1D, LnPrior1DTrait};

use enum_dispatch::enum_dispatch;
use itertools::Itertools;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[enum_dispatch(LnPrior)]
pub trait LnPriorTrait: Clone + Debug + Serialize + DeserializeOwned {
    fn ln_prior<const NPARAMS: usize>(&self, params: &[f64; NPARAMS]) -> f64;
}

/// Natural logarithm of prior for non-linear curve-fit problem
#[enum_dispatch]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum LnPrior {
    None(NoneLnPrior),
    IndComponents(IndComponentsLnPrior),
}

impl LnPrior {
    pub fn none() -> Self {
        NoneLnPrior {}.into()
    }

    pub fn ind_components(components: Vec<LnPrior1D>) -> Self {
        IndComponentsLnPrior { components }.into()
    }

    pub fn into_func<const NPARAMS: usize>(
        self,
    ) -> impl 'static + Clone + Fn(&[f64; NPARAMS]) -> f64 {
        move |params| self.ln_prior(params)
    }

    pub fn into_func_with_transformation<'a, F, const NPARAMS: usize>(
        self,
        transform: F,
    ) -> impl 'a + Clone + Fn(&[f64; NPARAMS]) -> f64
    where
        F: 'a + Clone + Fn(&[f64; NPARAMS]) -> [f64; NPARAMS],
    {
        move |params| self.ln_prior(&transform(params))
    }

    pub fn as_func<const NPARAMS: usize>(&self) -> impl '_ + Fn(&[f64; NPARAMS]) -> f64 {
        |params| self.ln_prior(params)
    }

    pub fn as_func_with_transformation<'a, F, const NPARAMS: usize>(
        &'a self,
        transform: F,
    ) -> impl 'a + Clone + Fn(&[f64; NPARAMS]) -> f64
    where
        F: 'a + Clone + Fn(&[f64; NPARAMS]) -> [f64; NPARAMS],
    {
        move |params| self.ln_prior(&transform(params))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct NoneLnPrior {}

impl LnPriorTrait for NoneLnPrior {
    fn ln_prior<const NPARAMS: usize>(&self, _params: &[f64; NPARAMS]) -> f64 {
        0.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct IndComponentsLnPrior {
    pub components: Vec<LnPrior1D>,
}

impl LnPriorTrait for IndComponentsLnPrior {
    fn ln_prior<const NPARAMS: usize>(&self, params: &[f64; NPARAMS]) -> f64 {
        params
            .iter()
            .zip_eq(self.components.iter())
            .map(|(&x, ln_prior)| ln_prior.ln_prior_1d(x))
            .sum()
    }
}
