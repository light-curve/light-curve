use crate::float_trait::Float;
use conv::prelude::*;
use hyperdual::Hyperdual;
use rgsl::{
    MatrixF64, MultiFitFdfSolver, MultiFitFdfSolverType, MultiFitFunctionFdf, Value, VectorF64,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::result::Result;

pub struct NLSProblem {
    pub max_iter: usize,
    pub atol: f64,
    pub rtol: f64,
    fit_function: MultiFitFunctionFdf,
}

impl NLSProblem {
    fn new(fit_function: MultiFitFunctionFdf) -> Self {
        Self {
            max_iter: 10,
            atol: 0.0,
            rtol: 1e-4,
            fit_function,
        }
    }

    pub fn solve(&mut self, x0: &VectorF64) -> Result<VectorF64, Value> {
        let mut solver = MultiFitFdfSolver::new(
            &MultiFitFdfSolverType::lmsder(),
            self.fit_function.n,
            self.fit_function.p,
        )
        .unwrap();
        let status = solver.set(&mut self.fit_function, x0);
        if status != Value::Success {
            return Err(status);
        }

        for _ in 0..self.max_iter {
            let status = solver.iterate();
            match status {
                Value::Success | Value::ToleranceX | Value::ToleranceF | Value::ToleranceG => {}
                _ => return Err(status),
            }

            let status =
                rgsl::multifit::test_delta(&solver.dx(), &solver.x(), self.atol, self.rtol);
            match status {
                Value::Success => return Ok(solver.x().clone().unwrap()),
                Value::Continue => {}
                _ => return Err(status),
            }
        }
        println!("Out of iter loop");
        Err(Value::MaxIteration)
    }

    pub fn from_f_df_fdf<F, DF, FDF>(t_size: usize, x_size: usize, f: F, df: DF, fdf: FDF) -> Self
    where
        F: 'static + Fn(VectorF64, VectorF64) -> Value,
        DF: 'static + Fn(VectorF64, MatrixF64) -> Value,
        FDF: 'static + Fn(VectorF64, VectorF64, MatrixF64) -> Value,
    {
        let mut fit_function = MultiFitFunctionFdf::new(t_size, x_size, 0, 0);
        fit_function.f = Some(Box::new(f));
        fit_function.df = Some(Box::new(df));
        fit_function.fdf = Some(Box::new(fdf));
        Self::new(fit_function)
    }

    pub fn from_f_df<F, DF>(t_size: usize, x_size: usize, f: F, df: DF) -> Self
    where
        F: 'static + Clone + Fn(VectorF64, VectorF64) -> Value,
        DF: 'static + Clone + Fn(VectorF64, MatrixF64) -> Value,
    {
        let f_clone = f.clone();
        let df_clone = df.clone();
        let fdf = move |x: VectorF64, residual: VectorF64, jacobian: MatrixF64| {
            let result = f_clone(x.clone().unwrap(), residual);
            if result != Value::Success {
                return result;
            }
            df_clone(x, jacobian)
        };

        Self::from_f_df_fdf(t_size, x_size, f, df, fdf)
    }

    pub fn from_dual_f<F>(t_size: usize, f: F) -> Self
    where
        F: 'static
            + Clone
            + Fn(&[Hyperdual<f64, hyperdual::U4>], &mut [Hyperdual<f64, hyperdual::U4>]),
    {
        let x_size = 4 - 1;

        let result = Rc::new(RefCell::new(vec![Hyperdual::from_real(0.0); t_size]));

        let function = {
            let f = f.clone();
            let result = result.clone();
            move |param: VectorF64, mut residual: VectorF64| {
                let param = slice_to_hyperdual_vec(param.as_slice().unwrap());
                f(&param, &mut result.borrow_mut());
                for i in 0..residual.len() {
                    residual.set(i, result.borrow()[i][0]);
                }
                Value::Success
            }
        };
        let jacobian = {
            let f = f.clone();
            let result = result.clone();
            move |param: VectorF64, mut jacobian: MatrixF64| {
                let param = slice_to_hyperdual_vec(param.as_slice().unwrap());
                f(&param, &mut result.borrow_mut());
                for i in 0..jacobian.size1() {
                    for j in 0..jacobian.size2() {
                        jacobian.set(i, j, result.borrow()[i][j + 1]);
                    }
                }
                Value::Success
            }
        };
        let fdf = move |param: VectorF64, mut residual: VectorF64, mut jacobian: MatrixF64| {
            let param = slice_to_hyperdual_vec(param.as_slice().unwrap());
            f(&param, &mut result.borrow_mut());
            for i in 0..jacobian.size1() {
                residual.set(i, result.borrow()[i][0]);
                for j in 0..jacobian.size2() {
                    jacobian.set(i, j, result.borrow()[i][j + 1]);
                }
            }
            Value::Success
        };

        Self::from_f_df_fdf(t_size, x_size, function, jacobian, fdf)
    }
}

// Cannot make it dimension-generic now
// https://github.com/rust-lang/rust/issues/78220
fn slice_to_hyperdual_vec<T>(s: &[T]) -> Vec<Hyperdual<f64, hyperdual::U4>>
where
    T: Float + hyperdual::Float,
    Hyperdual<f64, hyperdual::U4>: hyperdual::Float,
{
    assert_eq!(s.len() + 1, 4, "slice size should equals hyperdual order");
    s.iter()
        .enumerate()
        .map(|(i, &x)| {
            let mut hd = Hyperdual::from_real(x.value_as::<f64>().unwrap());
            hd[i + 1] = 1.0;
            hd
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::fit::straight_line::StraightLineFitterResult;
    use itertools;
    use light_curve_common::{all_close, linspace};

    fn to_vec_f64<T: Float>(x: &[T]) -> Vec<f64> {
        x.iter().map(|&a| a.value_as::<f64>().unwrap()).collect()
    }

    fn fit_straight_line<T: Float>(
        x: &[T],
        y: &[T],
        err2: Option<&[T]>,
    ) -> StraightLineFitterResult<T> {
        let t1 = Rc::new(to_vec_f64(x));
        let t2 = t1.clone();
        let y1 = to_vec_f64(y);

        const P: usize = 2;

        let mut fitter = NLSProblem::from_f_df(
            x.len(),
            P,
            move |x, mut residual| {
                let a = x.get(0);
                let b = x.get(1);
                for i in 0..residual.len() {
                    residual.set(i, a + b * t1[i] - y1[i]);
                }
                Value::Success
            },
            move |_x, mut jacobian| {
                for i in 0..jacobian.size1() {
                    jacobian.set(i, 0, 1.0);
                    jacobian.set(i, 1, t2[i]);
                }
                Value::Success
            },
        );

        let x = fitter.solve(&VectorF64::new(P).unwrap()).unwrap();

        StraightLineFitterResult {
            slope: x.get(1).approx_as::<T>().unwrap(),
            slope_sigma2: T::nan(),
            reduced_chi2: T::nan(),
        }
    }

    #[test]
    fn straight_line() {
        let x = linspace(0.0f64, 1.0, 1000);
        let y: Vec<_> = x.iter().map(|&a| a.powi(2)).collect();
        fit_straight_line(&x, &y, None);
    }

    struct Data {
        t: Vec<f64>,
        y: Vec<f64>,
        err2: Option<Vec<f64>>,
    }

    fn func(param: &rgsl::VectorF64, residual: &mut rgsl::VectorF64, data: &Data) -> rgsl::Value {
        let a = param.get(0);
        let b = param.get(1);

        let err2_iter = match &data.err2 {
            Some(e2) => itertools::Either::Left(e2.iter()),
            None => itertools::Either::Right(std::iter::once(&1.0).cycle()),
        };

        for (i, ((&x, &y), &err2)) in data.t.iter().zip(data.y.iter()).zip(err2_iter).enumerate() {
            /* Model Yi = A * exp(-lambda * i) + b */
            let f = a + b * x;

            residual.set(i, (f - y) / err2);
        }

        rgsl::Value::Success
    }

    fn jac(_param: &rgsl::VectorF64, jacobian: &mut rgsl::MatrixF64, data: &Data) -> rgsl::Value {
        let err2_iter = match &data.err2 {
            Some(e2) => itertools::Either::Left(e2.iter()),
            None => itertools::Either::Right(std::iter::repeat(&1.0)),
        };

        for (i, (&x, &err2)) in data.t.iter().zip(err2_iter).enumerate() {
            jacobian.set(i, 0, 1.0 / err2);
            jacobian.set(i, 1, x / err2);
        }

        rgsl::Value::Success
    }

    #[test]
    fn straight_line_noise() {
        let n = 40;
        let p = 2;

        let param_init: [f64; 2] = [0.0f64, 100.0];
        let param_init = rgsl::VectorF64::from_slice(&param_init).unwrap();

        let data = Rc::new(Data {
            t: (0..n).map(|i| i as f64).collect(),
            y: (0..n).map(|i| (i * i) as f64).collect(),
            err2: None,
            // err2: Some((0..n).map(|i| ((i + 1) as f64) * 0.01).collect()),
        });

        let mut solver = {
            let data_f = data.clone();
            let data_df = data.clone();
            NLSProblem::from_f_df(
                n,
                p,
                move |param, mut residual| func(&param, &mut residual, &data_f),
                move |param, mut jacobian| jac(&param, &mut jacobian, &data_df),
            )
        };

        let param = solver.solve(&param_init).unwrap();

        all_close(param.as_slice().unwrap(), &[-247.0, 39.0], 1e-9);
    }

    #[test]
    fn parabola_dual() {
        const N: usize = 40;
        const P: usize = 3;

        let param_init = [0.5f64, 0.5f64, 0.5f64];
        let param_init = VectorF64::from_slice(&param_init).unwrap();

        let data = Rc::new(Data {
            t: linspace(0.0, 1.0, N),
            y: (0..N)
                .map(|i| (i as f64 / (N - 1) as f64).powi(2) + (i as f64 / (N - 1) as f64))
                .collect(),
            err2: Some(vec![0.01; N]),
        });

        let mut solver = NLSProblem::from_dual_f(N, move |param, result| {
            for i in 0..result.len() {
                let y = param[0] + param[1] * data.t[i] + param[2] * data.t[i].powi(2);
                result[i] = (y - data.y[i]) / data.err2.as_ref().unwrap()[i];
            }
        });
        solver.max_iter = 100;

        let param = solver.solve(&param_init).unwrap();

        all_close(param.as_slice().unwrap(), &[0.0, 1.0, 1.0], 1e-4);
    }
}
