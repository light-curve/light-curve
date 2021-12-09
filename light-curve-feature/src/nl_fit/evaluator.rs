use crate::float_trait::Float;
use crate::nl_fit::{data::NormalizedData, CurveFitAlgorithm, LikeFloat, LnPrior};
use crate::time_series::TimeSeries;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub trait FitModelTrait<T, U, const NPARAMS: usize>
where
    T: Float + Into<U>,
    U: LikeFloat,
{
    fn model(t: T, param: &[U; NPARAMS]) -> U
    where
        T: Float + Into<U>,
        U: LikeFloat;
}

pub trait FitFunctionTrait<T: Float, const NPARAMS: usize>:
    FitModelTrait<T, T, NPARAMS> + FitParametersInternalDimlessTrait<T, NPARAMS>
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

pub trait FitDerivalivesTrait<T: Float, const NPARAMS: usize> {
    fn derivatives(t: T, param: &[T; NPARAMS], jac: &mut [T; NPARAMS]);
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(
    into = "FitArraySerde<T>",
    try_from = "FitArraySerde<T>",
    bound = "T: Debug + Clone + Serialize + DeserializeOwned + JsonSchema"
)]
pub struct FitArray<T, const NPARAMS: usize>(pub [T; NPARAMS]);

impl<T, const NPARAMS: usize> JsonSchema for FitArray<T, NPARAMS>
where
    T: schemars::JsonSchema,
{
    fn is_referenceable() -> bool {
        false
    }

    fn schema_name() -> String {
        FitArraySerde::<T>::schema_name()
    }

    fn json_schema(gen: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        FitArraySerde::<T>::json_schema(gen)
    }
}

impl<T, const NPARAMS: usize> From<[T; NPARAMS]> for FitArray<T, NPARAMS> {
    fn from(item: [T; NPARAMS]) -> Self {
        Self(item)
    }
}

impl<T, const NPARAMS: usize> std::ops::Deref for FitArray<T, NPARAMS> {
    type Target = [T; NPARAMS];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const NPARAMS: usize> From<FitArray<T, NPARAMS>> for FitArray<Option<T>, NPARAMS>
where
    T: Copy,
{
    fn from(item: FitArray<T, NPARAMS>) -> Self {
        let mut opt = [None; NPARAMS];
        for (&x, y) in item.0.iter().zip(opt.iter_mut()) {
            *y = Some(x);
        }
        opt.into()
    }
}

impl<T, const NPARAMS: usize> FitArray<Option<T>, NPARAMS>
where
    T: Clone,
{
    fn unwrap_with(&self, with: &FitArray<T, NPARAMS>) -> FitArray<T, NPARAMS> {
        let mut a = with.clone();
        for (opt, x) in self.0.iter().zip(a.0.iter_mut()) {
            match opt {
                Some(value) => *x = value.clone(),
                None => {}
            }
        }
        a
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(
    rename = "FitArray",
    bound = "T: Debug + Clone + Serialize + DeserializeOwned + JsonSchema"
)]
struct FitArraySerde<T>(Vec<T>);

impl<T, const NPARAMS: usize> From<FitArray<T, NPARAMS>> for FitArraySerde<T> {
    fn from(item: FitArray<T, NPARAMS>) -> Self {
        Self(item.0.into())
    }
}

impl<T, const NPARAMS: usize> TryFrom<FitArraySerde<T>> for FitArray<T, NPARAMS> {
    type Error = &'static str;

    fn try_from(item: FitArraySerde<T>) -> Result<Self, Self::Error> {
        Ok(Self(
            item.0
                .try_into()
                .map_err(|_| "wrong size of the FitArray object")?,
        ))
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct FitInitsBoundsArrays<const NPARAMS: usize> {
    pub init: FitArray<f64, NPARAMS>,
    pub lower: FitArray<f64, NPARAMS>,
    pub upper: FitArray<f64, NPARAMS>,
}

impl<const NPARAMS: usize> FitInitsBoundsArrays<NPARAMS> {
    pub fn new(init: [f64; NPARAMS], lower: [f64; NPARAMS], upper: [f64; NPARAMS]) -> Self {
        Self {
            init: init.into(),
            lower: lower.into(),
            upper: upper.into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct OptionFitInitsBoundsArrays<const NPARAMS: usize> {
    pub init: FitArray<Option<f64>, NPARAMS>,
    pub lower: FitArray<Option<f64>, NPARAMS>,
    pub upper: FitArray<Option<f64>, NPARAMS>,
}

impl<const NPARAMS: usize> OptionFitInitsBoundsArrays<NPARAMS> {
    pub fn new(
        init: [Option<f64>; NPARAMS],
        lower: [Option<f64>; NPARAMS],
        upper: [Option<f64>; NPARAMS],
    ) -> Self {
        Self {
            init: init.into(),
            lower: lower.into(),
            upper: upper.into(),
        }
    }

    pub fn unwrap_with(&self, x: &FitInitsBoundsArrays<NPARAMS>) -> FitInitsBoundsArrays<NPARAMS> {
        FitInitsBoundsArrays {
            init: self.init.unwrap_with(&x.init),
            lower: self.lower.unwrap_with(&x.lower),
            upper: self.upper.unwrap_with(&x.upper),
        }
    }
}

impl<const NPARAMS: usize> From<FitInitsBoundsArrays<NPARAMS>>
    for OptionFitInitsBoundsArrays<NPARAMS>
{
    fn from(item: FitInitsBoundsArrays<NPARAMS>) -> Self {
        Self {
            init: item.init.into(),
            lower: item.lower.into(),
            upper: item.upper.into(),
        }
    }
}

pub trait FitInitsBoundsTrait<T: Float, const NPARAMS: usize> {
    fn init_and_bounds_from_ts(&self, ts: &mut TimeSeries<T>) -> FitInitsBoundsArrays<NPARAMS>;
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

pub trait FitFeatureEvaluatorGettersTrait<const NPARAMS: usize> {
    fn get_algorithm(&self) -> &CurveFitAlgorithm;

    fn ln_prior_from_ts<T: Float>(&self, ts: &mut TimeSeries<T>) -> LnPrior<NPARAMS>;
}
