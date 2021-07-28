use crate::nl_fit::data::Data;
use crate::nl_fit::lmsder::LmsderCurveFit;
use crate::nl_fit::mcmc::McmcCurveFit;

use enum_dispatch::enum_dispatch;
use std::fmt::Debug;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct CurveFitResult<T> {
    pub x: Vec<T>,
    pub reduced_chi2: T,
    pub success: bool,
}

#[enum_dispatch]
pub trait CurveFitTrait: Clone + Debug {
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
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum CurveFitAlgorithm {
    Lmsder(LmsderCurveFit),
    Mcmc(McmcCurveFit),
}
