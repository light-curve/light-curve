use crate::nl_fit::data::Data;
#[cfg(feature = "gsl")]
use crate::nl_fit::lmsder::LmsderCurveFit;
use crate::nl_fit::mcmc::McmcCurveFit;

use enum_dispatch::enum_dispatch;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct CurveFitResult<T> {
    pub x: Vec<T>,
    pub reduced_chi2: T,
    pub success: bool,
}

#[enum_dispatch]
pub trait CurveFitTrait: Clone + Debug + Serialize + DeserializeOwned {
    fn curve_fit<F, DF>(
        &self,
        ts: Rc<Data<f64>>,
        x0: &[f64],
        model: F,
        derivatives: DF,
    ) -> CurveFitResult<f64>
    where
        F: 'static + Clone + Fn(f64, &[f64]) -> f64,
        DF: 'static + Clone + Fn(f64, &[f64], &mut [f64]);
}

#[enum_dispatch(CurveFitTrait)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum CurveFitAlgorithm {
    #[cfg(feature = "gsl")]
    Lmsder(LmsderCurveFit),
    Mcmc(McmcCurveFit),
}
