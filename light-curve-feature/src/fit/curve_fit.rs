use crate::fit::data::Data;
use crate::fit::nls::{MatrixF64, NlsProblem, Value, VectorF64};

use ndarray::Zip;
use std::rc::Rc;

pub struct CurveFitResult {
    pub x: Vec<f64>,
    pub reduced_chi2: f64,
    pub success: bool,
}

pub fn curve_fit<F, DF>(ts: Rc<Data<f64>>, x0: &[f64], model: F, derivatives: DF) -> CurveFitResult
where
    F: 'static + Clone + Fn(f64, &[f64]) -> f64,
    DF: 'static + Clone + Fn(f64, &[f64], &mut [f64]),
{
    let f = {
        let ts = ts.clone();
        move |param: VectorF64, mut residual: VectorF64| {
            let param = param.as_slice().unwrap();
            Zip::from(&ts.t)
                .and(&ts.m)
                .and(&ts.inv_err)
                .and(residual.as_slice_mut().unwrap())
                .for_each(|&t, &m, &inv_err, r| {
                    *r = inv_err * (model(t, param) - m);
                });
            Value::Success
        }
    };
    let df = {
        let ts = ts.clone();
        move |param: VectorF64, mut jacobian: MatrixF64| {
            let param = param.as_slice().unwrap();
            let mut buffer = vec![0.0; param.len()];
            Zip::indexed(&ts.t)
                .and(&ts.inv_err)
                .for_each(|i, &t, &inv_err| {
                    derivatives(t, param, &mut buffer);
                    for (j, &jac) in buffer.iter().enumerate() {
                        jacobian.set(i, j, inv_err * jac);
                    }
                });
            Value::Success
        }
    };

    let mut problem = NlsProblem::from_f_df(ts.t.len(), x0.len(), f, df);
    let result = problem.solve(VectorF64::from_slice(x0).unwrap());

    CurveFitResult {
        x: result.x().as_slice().unwrap().iter().copied().collect(),
        reduced_chi2: result.loss() / ((ts.t.len() - x0.len()) as f64),
        success: result.status == Value::Success,
    }
}
