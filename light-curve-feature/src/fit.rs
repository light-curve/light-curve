use std::collections::BTreeMap;

use conv::ConvUtil;

use crate::float_trait::Float;

pub struct FitResult<T> {
    pub values: BTreeMap<&'static str, T>,
    pub cov: BTreeMap<&'static str, BTreeMap<&'static str, T>>,
    pub reduced_chi2: T,
}

pub trait Fitter<T> {
    fn fit(&self) -> FitResult<T>;
}

macro_rules! map {
    {$($key:expr => $value:expr),+ $(,)?} => {
        {
            let mut m = BTreeMap::new();
            $(
                m.insert($key, $value);
            )+
            m
        }
    }
}

pub struct StraightLineFitResult<T> {
    slope: T,
    slope_sigma2: T,
    intercept: T,
    intercept_sigma2: T,
    cov: T,
    reduced_chi2: T,
}

pub struct StraightLineFirstCoeffs<T: Float> {
    ss: T,
    sx: T,
    sy: T,
}

impl<T> StraightLineFirstCoeffs<T>
where
    T: Float,
{
    fn new() -> Self {
        Self {
            ss: T::zero(),
            sx: T::zero(),
            sy: T::zero(),
        }
    }
}

pub struct StraightLineSecondCoeffs<T: Float> {
    st2: T,
    b_st2: T,
}

impl<T> StraightLineSecondCoeffs<T>
where
    T: Float,
{
    fn new() -> Self {
        Self {
            st2: T::zero(),
            b_st2: T::zero(),
        }
    }
}

pub trait StraightLineFitterTrait<T: Float>: Fitter<T> {
    fn first_coeffs(&self) -> StraightLineFirstCoeffs<T>;
    fn second_coeffs(&self, sxoss: T) -> StraightLineSecondCoeffs<T>;
    fn set_errors(&self, result: &mut StraightLineFitResult<T>);
}

impl<T: Float, F: StraightLineFitterTrait<T>> Fitter<T> for F {
    fn fit(&self) -> FitResult<T> {
        let c1 = self.first_coeffs();
        let sxoss = c1.sx / c1.ss;
        let c2 = self.second_coeffs(sxoss);
        let b = c2.b_st2 / c2.st2;
        let a = (c1.sy - c1.sx * b) / c1.ss;
        let sig2a = (T::one() + c1.sx.powi(2) / (c1.ss * c2.st2)) / c1.ss;
        let sig2b = T::recip(c2.st2);
        let cov = -c1.sx / (c1.ss * c2.st2);
        let mut result = StraightLineFitResult {
            slope: b,
            slope_sigma2: sig2b,
            intercept: a,
            intercept_sigma2: sig2a,
            cov,
            reduced_chi2: T::zero(),
        };
        self.set_errors(&mut result);
        FitResult {
            values: map! {"slope" => result.slope, "intercept" => result.intercept},
            cov: map! {
                "slope" => map!{"slope" => result.slope_sigma2, "intercept" => result.cov},
                "intercept" => map!{"intercept" => result.intercept_sigma2, "slope" => result.cov},
            },
            reduced_chi2: result.reduced_chi2,
        }
    }
}

pub struct StraightLineFitter<'a, 'b, T: Float> {
    x: &'a [T],
    y: &'b [T],
}

impl<'a, 'b, T> StraightLineFitter<'a, 'b, T>
where
    T: Float,
{
    pub fn new(x: &'a [T], y: &'b [T]) -> Self {
        assert_eq!(x.len(), y.len());
        Self { x, y }
    }
}

impl<'a, 'b, T> StraightLineFitterTrait<T> for StraightLineFitter<'a, 'b, T>
where
    T: Float,
{
    fn first_coeffs(&self) -> StraightLineFirstCoeffs<T> {
        let mut coeffs = StraightLineFirstCoeffs::new();
        coeffs.ss = self.x.len().value_as::<T>().unwrap();
        for (&x, &y) in self.x.iter().zip(self.y.iter()) {
            coeffs.sx += x;
            coeffs.sy += y;
        }
        coeffs
    }

    fn second_coeffs(&self, sxoss: T) -> StraightLineSecondCoeffs<T> {
        let mut coeffs = StraightLineSecondCoeffs::new();
        for (&x, &y) in self.x.iter().zip(self.y.iter()) {
            let t = x - sxoss;
            coeffs.st2 += t.powi(2);
            coeffs.b_st2 += t * y;
        }
        coeffs
    }

    fn set_errors(&self, result: &mut StraightLineFitResult<T>) {
        for (&x, &y) in self.x.iter().zip(self.y.iter()) {
            result.reduced_chi2 += (y - result.intercept - result.slope * x).powi(2);
        }
        result.reduced_chi2 /= self.x.len().value_as::<T>().unwrap() - T::two();
        result.slope_sigma2 *= result.reduced_chi2;
        result.intercept_sigma2 *= result.reduced_chi2;
    }
}

pub struct StraightLineNoisyFitter<'a, 'b, 'c, T: Float> {
    x: &'a [T],
    y: &'b [T],
    err2: &'c [T],
}

impl<'a, 'b, 'c, T> StraightLineNoisyFitter<'a, 'b, 'c, T>
where
    T: Float,
{
    pub fn new(x: &'a [T], y: &'b [T], err2: &'c [T]) -> Self {
        assert_eq!(x.len(), y.len());
        assert_eq!(y.len(), err2.len());
        Self { x, y, err2 }
    }
}

impl<'a, 'b, 'c, T> StraightLineFitterTrait<T> for StraightLineNoisyFitter<'a, 'b, 'c, T>
where
    T: Float,
{
    fn first_coeffs(&self) -> StraightLineFirstCoeffs<T> {
        let mut coeffs = StraightLineFirstCoeffs::new();
        for ((&x, &y), &err2) in self.x.iter().zip(self.y.iter()).zip(self.err2.iter()) {
            let wt = err2.recip();
            coeffs.ss += wt;
            coeffs.sx += wt * x;
            coeffs.sy += wt * y;
        }
        coeffs
    }

    fn second_coeffs(&self, sxoss: T) -> StraightLineSecondCoeffs<T> {
        let mut coeffs = StraightLineSecondCoeffs::new();
        for ((&x, &y), &err2) in self.x.iter().zip(self.y.iter()).zip(self.err2.iter()) {
            let x_sxoss = x - sxoss;
            let g = x_sxoss / err2;
            coeffs.st2 += g * x_sxoss;
            coeffs.b_st2 += g * y;
        }
        coeffs
    }

    fn set_errors(&self, result: &mut StraightLineFitResult<T>) {
        for ((&x, &y), &err2) in self.x.iter().zip(self.y.iter()).zip(self.err2.iter()) {
            result.reduced_chi2 += (y - result.intercept - result.slope * x).powi(2) / err2;
        }
        result.reduced_chi2 /= self.x.len().value_as::<T>().unwrap() - T::two();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use light_curve_common::all_close;

    #[test]
    fn straight_line_fitter() {
        let x = [0.5, 1.5, 2.5, 5.0, 7.0, 16.0];
        let y = [-1.0, 3.0, 2.0, 6.0, 10.0, 25.0];
        let names = ["intercept", "slope"];
        // scipy.optimize.curve_fit(absolute_sigma=False)
        let actual_values = [-1.3303457, 1.63021767];
        let actual_cov = [0.44109184, -0.03329066, -0.03329066, 0.0078127];
        let result = StraightLineFitter::new(&x[..], &y[..]).fit();
        let result_ref = &result;
        let desired_values: Vec<_> = names.iter().map(|&name| result.values[name]).collect();
        let desired_cov: Vec<_> = names
            .iter()
            .flat_map(|&name1| names.iter().map(move |&name2| result_ref.cov[name1][name2]))
            .collect();
        all_close(&actual_values[..], &desired_values[..], 1e-6);
        all_close(&actual_cov[..], &desired_cov[..], 1e-6);
    }

    #[test]
    fn straight_line_noisy_fitter() {
        let x = [0.5, 1.5, 2.5, 5.0, 7.0, 16.0];
        let y = [-1.0, 3.0, 2.0, 6.0, 10.0, 25.0];
        let err = [0.5, 1.0, 0.3, 0.1, 0.9, 2.5];
        let names = ["intercept", "slope"];
        // scipy.optimize.curve_fit(absolute_sigma=True)
        let actual_values = [-1.77189545, 1.60504579];
        let actual_cov = [0.20954795, -0.03651815, -0.03651815, 0.00868733];
        let actual_reduced_chi2 = [1.8057513419557492];
        let result = StraightLineNoisyFitter::new(&x[..], &y[..], &err[..]).fit();
        let result_ref = &result;
        let desired_values: Vec<_> = names.iter().map(|&name| result.values[name]).collect();
        let desired_cov: Vec<_> = names
            .iter()
            .flat_map(|&name1| names.iter().map(move |&name2| result_ref.cov[name1][name2]))
            .collect();
        all_close(&actual_values[..], &desired_values[..], 1e-6);
        all_close(&actual_cov[..], &desired_cov[..], 1e-6);
        all_close(&actual_reduced_chi2[..], &[result.reduced_chi2], 1e-6);
    }
}
