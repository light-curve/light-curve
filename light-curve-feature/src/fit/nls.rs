use crate::float_trait::Float;
use conv::prelude::*;
use hyperdual::Hyperdual;
pub use rgsl::{MatrixF64, Value, VectorF64};
use rgsl::{MultiFitFdfSolver, MultiFitFdfSolverType, MultiFitFunctionFdf};
use std::cell::RefCell;
use std::rc::Rc;

pub struct NlsProblem {
    pub max_iter: usize,
    pub atol: f64,
    pub rtol: f64,
    fit_function: MultiFitFunctionFdf,
}

impl NlsProblem {
    fn new(fit_function: MultiFitFunctionFdf) -> Self {
        Self {
            max_iter: 10,
            atol: 0.0,
            rtol: 1e-4,
            fit_function,
        }
    }

    pub fn solve(&mut self, x0: VectorF64) -> NlsFitResult {
        let mut solver = MultiFitFdfSolver::new(
            &MultiFitFdfSolverType::lmsder(),
            self.fit_function.n,
            self.fit_function.p,
        )
        .unwrap();
        match solver.set(&mut self.fit_function, &x0) {
            Value::Success => {}
            status => return NlsFitResult { status, solver },
        }

        for _ in 0..self.max_iter {
            match solver.iterate() {
                Value::Success | Value::ToleranceX | Value::ToleranceF | Value::ToleranceG => {}
                status => return NlsFitResult { status, solver },
            }

            match rgsl::multifit::test_delta(&solver.dx(), &solver.x(), self.atol, self.rtol) {
                Value::Continue => {}
                status => return NlsFitResult { status, solver },
            }
        }
        NlsFitResult {
            status: Value::MaxIteration,
            solver,
        }
    }

    /// Construct a problem from function (f), its jacobian (df) and fdf
    ///
    /// Looks like fdf is never called
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

    #[allow(dead_code)]
    /// Create fitter from residual function of dual numbers
    ///
    /// WIP: implemented for parameter vector of length three only, stacked by
    /// https://github.com/rust-lang/rust/issues/78220
    ///
    /// Current implementation is something like twice slower than `Self::from_f_df_fdf`
    pub fn from_dual_f<RF, DF>(t_size: usize, real_f: RF, dual_f: DF) -> Self
    where
        RF: 'static + Clone + Fn(&[f64], &mut [f64]),
        DF: 'static
            + Clone
            + Fn(&[Hyperdual<f64, hyperdual::U4>], &mut [Hyperdual<f64, hyperdual::U4>]),
    {
        let x_size = 4 - 1;

        let result_dual = Rc::new(RefCell::new(vec![Hyperdual::from_real(0.0); t_size]));

        let function = {
            move |param: VectorF64, mut residual: VectorF64| {
                real_f(param.as_slice().unwrap(), residual.as_slice_mut().unwrap());
                Value::Success
            }
        };
        let jacobian = {
            let f = dual_f.clone();
            let result = result_dual.clone();
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
        let fdf = {
            let f = dual_f;
            move |param: VectorF64, mut residual: VectorF64, mut jacobian: MatrixF64| {
                let param = slice_to_hyperdual_vec(param.as_slice().unwrap());
                f(&param, &mut result_dual.borrow_mut());
                for i in 0..jacobian.size1() {
                    residual.set(i, result_dual.borrow()[i][0]);
                    for j in 0..jacobian.size2() {
                        jacobian.set(i, j, result_dual.borrow()[i][j + 1]);
                    }
                }
                Value::Success
            }
        };

        Self::from_f_df_fdf(t_size, x_size, function, jacobian, fdf)
    }
}

// Cannot make it dimension-generic for now
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

pub struct NlsFitResult {
    pub status: Value,
    solver: MultiFitFdfSolver,
}

impl NlsFitResult {
    pub fn x(&self) -> VectorF64 {
        self.solver.x()
    }

    pub fn f(&self) -> VectorF64 {
        self.solver.f()
    }

    pub fn loss(&self) -> f64 {
        self.f().as_slice().unwrap().iter().map(|x| x.powi(2)).sum()
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::fit::straight_line::StraightLineFitterResult;
    use light_curve_common::{all_close, linspace};
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    use std::ops::Mul;

    fn to_vec_f64<T: Float>(x: &[T]) -> Vec<f64> {
        x.iter().map(|&a| a.value_as::<f64>().unwrap()).collect()
    }

    fn fit_straight_line<T: Float>(
        x: &[T],
        y: &[T],
        _w: Option<&[T]>,
    ) -> StraightLineFitterResult<T> {
        let t1 = Rc::new(to_vec_f64(x));
        let t2 = t1.clone();
        let y1 = to_vec_f64(y);

        const P: usize = 2;

        let mut fitter = NlsProblem::from_f_df(
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

        let result = fitter.solve(VectorF64::new(P).unwrap());
        assert_eq!(result.status, Value::Success);
        let param = result.x();

        StraightLineFitterResult {
            slope: param.get(1).approx_as::<T>().unwrap(),
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
        err: Option<Vec<f64>>,
    }

    fn func(param: &rgsl::VectorF64, residual: &mut rgsl::VectorF64, data: &Data) -> rgsl::Value {
        let a = param.get(0);
        let b = param.get(1);

        let err2_iter = match &data.err {
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
        let err2_iter = match &data.err {
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
            err: None,
            // err2: Some((0..n).map(|i| ((i + 1) as f64) * 0.01).collect()),
        });

        let mut solver = {
            let data_f = data.clone();
            let data_df = data;
            NlsProblem::from_f_df(
                n,
                p,
                move |param, mut residual| func(&param, &mut residual, &data_f),
                move |param, mut jacobian| jac(&param, &mut jacobian, &data_df),
            )
        };

        let result = solver.solve(param_init);
        assert_eq!(result.status, Value::Success);
        let param = result.x();

        all_close(param.as_slice().unwrap(), &[-247.0, 39.0], 1e-9);
    }

    #[test]
    fn parabola_dual() {
        const N: usize = 40;

        let param_init = [0.5f64, 0.5f64, 0.5f64];
        let param_init = VectorF64::from_slice(&param_init).unwrap();

        let data = Rc::new(Data {
            t: linspace(0.0, 1.0, N),
            y: (0..N)
                .map(|i| (i as f64 / (N - 1) as f64).powi(2) + (i as f64 / (N - 1) as f64))
                .collect(),
            err: Some(vec![0.01; N]),
        });
        let data_real = data.clone();
        let data_dual = data;

        let mut fitter = NlsProblem::from_dual_f(
            N,
            move |param, result| {
                for (i, r) in result.iter_mut().enumerate() {
                    let y =
                        param[0] + param[1] * data_real.t[i] + param[2] * data_real.t[i].powi(2);
                    *r = (y - data_real.y[i]) / data_real.err.as_ref().unwrap()[i];
                }
            },
            move |param, result| {
                for (i, r) in result.iter_mut().enumerate() {
                    let y =
                        param[0] + param[1] * data_dual.t[i] + param[2] * data_dual.t[i].powi(2);
                    *r = (y - data_dual.y[i]) / data_dual.err.as_ref().unwrap()[i];
                }
            },
        );
        fitter.atol = 1e-8;
        fitter.rtol = 0.0;

        let result = fitter.solve(param_init);
        assert_eq!(result.status, Value::Success);
        let param = result.x();

        all_close(param.as_slice().unwrap(), &[0.0, 1.0, 1.0], 1e-8);
    }

    #[inline]
    fn nonlinear_func<T, U>(param: &[T], t: &U) -> T
    where
        T: hyperdual::Float + Mul<U, Output = T>,
        U: hyperdual::Float,
    {
        param[1] * T::exp(-param[0] * *t) * t.powi(2) + param[2]
    }

    #[test]
    fn nonlinear_fdf() {
        const N: usize = 300;
        const P: usize = 3;
        const NOISE: f64 = 0.5;
        const RTOL: f64 = 1e-6;

        // curve_fit(lambda x, a, b, c: b * np.exp(-a * x) * x**2 + c, xdata=t, ydata=y, p0=[1, 1, 1], xtol=1e-6)
        let desired = [0.7450598836400693, 1.981911479079224, 0.5094446163866907];

        let param_true = [0.75, 2.0, 0.5];
        let param_init = [1.0, 1.0, 1.0];
        let param_init = VectorF64::from_slice(&param_init).unwrap();

        let mut rng = StdRng::seed_from_u64(0);

        let t = linspace(0.0, 10.0, N);
        let y = t
            .iter()
            .map(|x| {
                let eps: f64 = rng.sample(StandardNormal);
                nonlinear_func(&param_true, x) + NOISE * eps
            })
            .collect();
        let data = Rc::new(Data { t, y, err: None });

        let function = {
            let data = data.clone();
            move |x: VectorF64, mut residual: VectorF64| {
                for i in 0..residual.len() {
                    residual.set(
                        i,
                        nonlinear_func(x.as_slice().unwrap(), &data.t[i]) - data.y[i],
                    );
                }
                Value::Success
            }
        };
        let jac = {
            let data = data.clone();
            move |x: VectorF64, mut jacobian: MatrixF64| {
                for i in 0..jacobian.size1() {
                    let exp_t2 = f64::exp(-x.get(0) * data.t[i]) * data.t[i].powi(2);
                    jacobian.set(i, 0, -x.get(1) * data.t[i] * exp_t2);
                    jacobian.set(i, 1, exp_t2);
                    jacobian.set(i, 2, 1.0);
                }
                Value::Success
            }
        };
        let fdf = move |x: VectorF64, mut residual: VectorF64, mut jacobian: MatrixF64| {
            for i in 0..jacobian.size1() {
                let exp_t2 = f64::exp(-x.get(0) * data.t[i]) * data.t[i].powi(2);
                residual.set(i, x.get(1) * exp_t2 + x.get(2) - data.y[i]);
                jacobian.set(i, 0, -x.get(1) * data.t[i] * exp_t2);
                jacobian.set(i, 1, exp_t2);
                jacobian.set(i, 2, 1.0);
            }
            Value::Success
        };

        let mut fitter = NlsProblem::from_f_df_fdf(N, P, function, jac, fdf);
        fitter.rtol = RTOL;

        let result = fitter.solve(param_init);
        assert_eq!(result.status, Value::Success);
        let param = result.x();

        all_close(
            param.as_slice().unwrap(),
            &param_true,
            NOISE / (N as f64).sqrt(),
        );
        all_close(param.as_slice().unwrap(), &desired, RTOL);
    }

    #[test]
    fn nonlinear_dual() {
        const N: usize = 300;
        const NOISE: f64 = 0.5;
        const RTOL: f64 = 1e-6;

        // curve_fit(lambda x, a, b, c: b * np.exp(-a * x) * x**2 + c, xdata=t, ydata=y, p0=[1, 1, 1], xtol=1e-6)
        let desired = [0.7450598836400693, 1.981911479079224, 0.5094446163866907];

        let param_true = [0.75, 2.0, 0.5];
        let param_init = [1.0, 1.0, 1.0];
        let param_init = VectorF64::from_slice(&param_init).unwrap();

        let mut rng = StdRng::seed_from_u64(0);

        let t = linspace(0.0, 10.0, N);
        let y = t
            .iter()
            .map(|x| {
                let eps: f64 = rng.sample(StandardNormal);
                nonlinear_func(&param_true, x) + NOISE * eps
            })
            .collect();
        let data = Rc::new(Data { t, y, err: None });
        let data_real = data.clone();
        let data_dual = data;

        let mut fitter = NlsProblem::from_dual_f(
            N,
            move |x: &[f64], result: &mut [f64]| {
                for (t, (&y, r)) in data_real
                    .t
                    .iter()
                    .zip(data_real.y.iter().zip(result.iter_mut()))
                {
                    *r = nonlinear_func(x, t) - y;
                }
            },
            move |x: &[Hyperdual<f64, _>], result: &mut [Hyperdual<f64, _>]| {
                for (t, (&y, r)) in data_dual
                    .t
                    .iter()
                    .zip(data_dual.y.iter().zip(result.iter_mut()))
                {
                    *r = nonlinear_func(x, t) - y;
                }
            },
        );
        fitter.rtol = RTOL;

        let result = fitter.solve(param_init);
        assert_eq!(result.status, Value::Success);
        let param = result.x();

        all_close(
            param.as_slice().unwrap(),
            &param_true,
            NOISE / (N as f64).sqrt(),
        );
        all_close(param.as_slice().unwrap(), &desired, RTOL);
    }
}
