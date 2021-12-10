use crate::nl_fit::prior::ln_prior_1d::{LnPrior1D, LnPrior1DTrait};

use enum_dispatch::enum_dispatch;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[enum_dispatch]
pub trait LnPriorTrait<const NPARAMS: usize>: Clone + Debug + Serialize + DeserializeOwned {
    fn ln_prior(&self, params: &[f64; NPARAMS]) -> f64;
}

/// Natural logarithm of prior for non-linear curve-fit problem
#[enum_dispatch(LnPriorTrait<NPARAMS>)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum LnPrior<const NPARAMS: usize> {
    None(NoneLnPrior),
    IndComponents(IndComponentsLnPrior<NPARAMS>),
}

impl<const NPARAMS: usize> LnPrior<NPARAMS> {
    pub fn none() -> Self {
        NoneLnPrior {}.into()
    }

    pub fn ind_components(components: [LnPrior1D; NPARAMS]) -> Self {
        IndComponentsLnPrior { components }.into()
    }

    pub fn into_func(self) -> impl 'static + Clone + Fn(&[f64; NPARAMS]) -> f64 {
        move |params| self.ln_prior(params)
    }

    pub fn into_func_with_transformation<'a, F>(
        self,
        transform: F,
    ) -> impl 'a + Clone + Fn(&[f64; NPARAMS]) -> f64
    where
        F: 'a + Clone + Fn(&[f64; NPARAMS]) -> [f64; NPARAMS],
    {
        move |params| self.ln_prior(&transform(params))
    }

    pub fn as_func(&self) -> impl '_ + Fn(&[f64; NPARAMS]) -> f64 {
        |params| self.ln_prior(params)
    }

    pub fn as_func_with_transformation<'a, F>(
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

impl<const NPARAMS: usize> LnPriorTrait<NPARAMS> for NoneLnPrior {
    fn ln_prior(&self, _params: &[f64; NPARAMS]) -> f64 {
        0.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    into = "IndComponentsLnPriorSerde",
    try_from = "IndComponentsLnPriorSerde"
)]
pub struct IndComponentsLnPrior<const NPARAMS: usize> {
    pub components: [LnPrior1D; NPARAMS],
}

impl<const NPARAMS: usize> LnPriorTrait<NPARAMS> for IndComponentsLnPrior<NPARAMS> {
    fn ln_prior(&self, params: &[f64; NPARAMS]) -> f64 {
        params
            .iter()
            .zip(self.components.iter())
            .map(|(&x, ln_prior)| ln_prior.ln_prior_1d(x))
            .sum()
    }
}

impl<const NPARAMS: usize> JsonSchema for IndComponentsLnPrior<NPARAMS> {
    fn is_referenceable() -> bool {
        false
    }

    fn schema_name() -> String {
        IndComponentsLnPriorSerde::schema_name()
    }

    fn json_schema(gen: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        IndComponentsLnPriorSerde::json_schema(gen)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "IndComponentsLnPrior")]
struct IndComponentsLnPriorSerde {
    components: Vec<LnPrior1D>,
}

impl<const NPARAMS: usize> From<IndComponentsLnPrior<NPARAMS>> for IndComponentsLnPriorSerde {
    fn from(value: IndComponentsLnPrior<NPARAMS>) -> Self {
        Self {
            components: value.components.into(),
        }
    }
}

impl<const NPARAMS: usize> TryFrom<IndComponentsLnPriorSerde> for IndComponentsLnPrior<NPARAMS> {
    type Error = &'static str;

    fn try_from(value: IndComponentsLnPriorSerde) -> Result<Self, Self::Error> {
        Ok(Self {
            components: value
                .components
                .try_into()
                .map_err(|_| "wrong size of the IndComponentsLnPrior.components")?,
        })
    }
}
