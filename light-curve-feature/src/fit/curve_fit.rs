use crate::fit::nls::{MatrixF64, NLSFitResult, NLSProblem, Value, VectorF64};
use crate::float_trait::Float;
use crate::time_series::TimeSeries;
use conv::ConvUtil;
use std::rc::Rc;

pub struct CurveFitResult<T> {
    pub x: Vec<T>,
    pub loss: T,
    pub success: bool,
}

pub fn curve_fit<T, F, DF>(
    ts: &TimeSeries<T>,
    x: &[T],
    model: F,
    derivatives: DF,
) -> CurveFitResult<T>
where
    T: Float,
    F: 'static + Clone + Fn(f64, &[f64]) -> f64,
    DF: 'static + Clone + Fn(f64, &[f64], &mut [f64]),
{
    let x: Vec<f64> = x.iter().map(|&x| x.value_into().unwrap()).collect();
    let ts = Rc::new(ts.to_owned_f64());

    let f = {
        let ts = ts.clone();
        move |param: VectorF64, mut residual: VectorF64| {
            let param = param.as_slice().unwrap();
            for ((t, m, w), r) in ts
                .tmw_iter()
                .zip(residual.as_slice_mut().unwrap().iter_mut())
            {
                *r = w * (model(t, param) - m);
            }
            Value::Success
        }
    };
    let df = {
        let ts = ts.clone();
        move |param: VectorF64, mut jacobian: MatrixF64| {
            let param = param.as_slice().unwrap();
            let mut buffer = vec![0.0; param.len()];
            for (i, (t, w)) in ts.tw_iter().enumerate() {
                derivatives(t, param, &mut buffer);
                for (j, &jac) in buffer.iter().enumerate() {
                    jacobian.set(i, j, w * jac);
                }
            }
            Value::Success
        }
    };

    let mut problem = NLSProblem::from_f_df(ts.lenu(), x.len(), f, df);
    let result = problem.solve(VectorF64::from_slice(&x).unwrap());

    CurveFitResult {
        x: result
            .x()
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x| x.approx_as::<T>().unwrap())
            .collect(),
        loss: result.loss().approx_as::<T>().unwrap(),
        success: result.status == Value::Success,
    }
}
