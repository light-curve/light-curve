use crate::float_trait::Float;
use crate::nl_fit::{data::NormalizedData, CurveFitAlgorithm, LikeFloat, LnPrior};
use crate::time_series::TimeSeries;

pub trait FitModelTrait<T, U>
where
    T: Float + Into<U>,
    U: LikeFloat,
{
    fn model(t: T, param: &[U]) -> U
    where
        T: Float + Into<U>,
        U: LikeFloat;
}

pub trait FitFunctionTrait<T: Float, const NPARAMS: usize>:
    FitModelTrait<T, T> + FitParametersInternalDimlessTrait<T, NPARAMS>
{
    fn f(t: T, values: &[T]) -> T {
        let internal = Self::dimensionless_to_internal(
            values[..NPARAMS]
                .try_into()
                .expect("values slice's length is too small"),
        );
        Self::model(t, &internal)
    }
}

pub trait FitDerivalivesTrait<T: Float> {
    fn derivatives(t: T, param: &[T], jac: &mut [T]);
}

pub trait FitInitsBoundsTrait<T: Float, const NPARAMS: usize> {
    fn init_and_bounds_from_ts(ts: &mut TimeSeries<T>) -> ([f64; NPARAMS], [(f64, f64); NPARAMS]);
}

pub trait FitParametersInternalDimlessTrait<U: LikeFloat, const NPARAMS: usize> {
    fn dimensionless_to_internal(params: &[U; NPARAMS]) -> [U; NPARAMS];

    fn internal_to_dimensionless(params: &[U; NPARAMS]) -> [U; NPARAMS];
}

pub trait FitParametersOriginalDimLessTrait<const NPARAMS: usize> {
    fn orig_to_dimensionless(
        norm_data: &NormalizedData<f64>,
        orig: &[f64; NPARAMS],
    ) -> [f64; NPARAMS];

    fn dimensionless_to_orig(
        norm_data: &NormalizedData<f64>,
        norm: &[f64; NPARAMS],
    ) -> [f64; NPARAMS];
}

pub trait FitParametersInternalExternalTrait<const NPARAMS: usize>:
    FitParametersInternalDimlessTrait<f64, NPARAMS> + FitParametersOriginalDimLessTrait<NPARAMS>
{
    fn convert_to_internal(
        norm_data: &NormalizedData<f64>,
        orig: &[f64; NPARAMS],
    ) -> [f64; NPARAMS] {
        Self::dimensionless_to_internal(&Self::orig_to_dimensionless(norm_data, orig))
    }

    fn convert_to_external(
        norm_data: &NormalizedData<f64>,
        params: &[f64; NPARAMS],
    ) -> [f64; NPARAMS] {
        Self::dimensionless_to_orig(norm_data, &Self::internal_to_dimensionless(params))
    }
}

pub trait FitFeatureEvaluatorGettersTrait {
    fn get_algorithm(&self) -> &CurveFitAlgorithm;

    fn get_ln_prior(&self) -> &LnPrior;
}
