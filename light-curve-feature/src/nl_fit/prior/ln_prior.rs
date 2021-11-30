use crate::nl_fit::prior::ln_prior_1d::{LnPrior1D, LnPrior1DTrait};

use enum_dispatch::enum_dispatch;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[enum_dispatch]
pub trait LnPriorTrait: Clone + Debug + Serialize + DeserializeOwned {
    fn ln_prior(&self, params: &[f64]) -> f64;
}

/// Natural logarithm of prior for non-linear curve-fit problem
#[enum_dispatch(LnPrioirTrait)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum LnPrior {
    IndComponents(IndComponentsLnPrior),
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct IndComponentsLnPrior {
    components: Vec<LnPrior1D>,
}

impl LnPriorTrait for IndComponentsLnPrior {
    fn ln_prior(&self, params: &[f64]) -> f64 {
        params
            .iter()
            .zip(self.components.iter())
            .map(|(&x, ln_prior)| ln_prior.ln_prior_1d(x))
            .sum()
    }
}
