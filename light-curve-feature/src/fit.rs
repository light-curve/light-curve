use crate::float_trait::Float;
use conv::ConvUtil;

// See Press et al. sec. 15.2 Fitting Data to a Straight Line, p. 661

pub fn fit_straight_line<T: Float>(
    x: &[T],
    y: &[T],
    err2: Option<&[T]>,
) -> StraightLineFitterResult<T> {
    let fitter: Box<dyn StraightLineFitter<_>> = match err2 {
        Some(err2) => Box::new(NoisyStraightLineFitter::new(x, y, err2)),
        None => Box::new(NoiselessStraightLineFitter::new(x, y)),
    };
    fitter.fit()
}

pub struct StraightLineFitterResult<T> {
    pub slope: T,
    pub slope_sigma2: T,
    pub reduced_chi2: T,
}

struct StraightLineFitterSums<T: Float> {
    n: T,
    x0: T,
    s: T,
    x: T,
    y: T,
    xx: T,
    xy: T,
    yy: T,
}

impl<T> StraightLineFitterSums<T>
where
    T: Float,
{
    fn new(x: &[T]) -> Self {
        Self {
            n: x.len().value_as::<T>().unwrap(),
            x0: x[0],
            s: T::zero(),
            x: T::zero(),
            y: T::zero(),
            xx: T::zero(),
            xy: T::zero(),
            yy: T::zero(),
        }
    }
}

trait StraightLineFitter<T>
where
    T: Float,
{
    fn get_sums(&self) -> StraightLineFitterSums<T>;
    fn sigma_is_known(&self) -> bool;

    fn fit(&self) -> StraightLineFitterResult<T> {
        let s = self.get_sums();
        let inv_delta = T::one() / (s.s * s.xx - s.x.powi(2));
        let slope = (s.s * s.xy - s.x * s.y) * inv_delta;
        let a = (s.xx * s.y - s.x * s.xy) * inv_delta;
        let chi2 = s.yy - a * s.y - slope * s.xy;
        let reduced_chi2 = chi2 / (s.n - T::two());
        let mut slope_sigma2 = s.s * inv_delta;
        if !self.sigma_is_known() {
            slope_sigma2 *= reduced_chi2;
        }
        StraightLineFitterResult {
            slope,
            slope_sigma2,
            reduced_chi2,
        }
    }
}

struct NoiselessStraightLineFitter<'a, 'b, T: Float> {
    x: &'a [T],
    y: &'b [T],
}

impl<'a, 'b, T> NoiselessStraightLineFitter<'a, 'b, T>
where
    T: Float,
{
    fn new(x: &'a [T], y: &'b [T]) -> Self {
        assert_eq!(x.len(), y.len());
        Self { x, y }
    }
}

impl<'a, 'b, T> StraightLineFitter<T> for NoiselessStraightLineFitter<'a, 'b, T>
where
    T: Float,
{
    fn get_sums(&self) -> StraightLineFitterSums<T> {
        let mut s = StraightLineFitterSums::new(self.x);
        s.s = s.n;
        for (&x, &y) in self.x.iter().zip(self.y.iter()) {
            let dx = x - s.x0;
            s.x += dx;
            s.y += y;
            s.xx += dx.powi(2);
            s.xy += dx * y;
            s.yy += y.powi(2);
        }
        s
    }

    fn sigma_is_known(&self) -> bool {
        false
    }
}

struct NoisyStraightLineFitter<'a, 'b, 'c, T: Float> {
    x: &'a [T],
    y: &'b [T],
    err2: &'c [T],
}

impl<'a, 'b, 'c, T> NoisyStraightLineFitter<'a, 'b, 'c, T>
where
    T: Float,
{
    fn new(x: &'a [T], y: &'b [T], err2: &'c [T]) -> Self {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), err2.len());
        Self { x, y, err2 }
    }
}

impl<'a, 'b, 'c, T> StraightLineFitter<T> for NoisyStraightLineFitter<'a, 'b, 'c, T>
where
    T: Float,
{
    fn get_sums(&self) -> StraightLineFitterSums<T> {
        let mut s = StraightLineFitterSums::new(self.x);
        for ((&x, &y), w) in self
            .x
            .iter()
            .zip(self.y.iter())
            .zip(self.err2.iter().map(|&err2| T::one() / err2))
        {
            let dx = x - s.x0;
            s.s += w;
            s.x += w * dx;
            s.y += w * y;
            s.xx += w * dx.powi(2);
            s.xy += w * dx * y;
            s.yy += w * y.powi(2);
        }
        s
    }

    fn sigma_is_known(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use light_curve_common::all_close;

    #[test]
    fn straight_line_fitter() {
        let x = [0.5, 1.5, 2.5, 5.0, 7.0, 16.0];
        let y = [-1.0, 3.0, 2.0, 6.0, 10.0, 25.0];
        // scipy.optimize.curve_fit(absolute_sigma=False)
        let desired_slope = 1.63021767;
        let desired_slope_sigma2 = 0.0078127;
        let result = fit_straight_line(&x, &y, None);
        all_close(&[result.slope], &[desired_slope], 1e-6);
        all_close(&[result.slope_sigma2], &[desired_slope_sigma2], 1e-6);
    }

    #[test]
    fn noisy_straight_line_fitter() {
        let x = [0.5, 1.5, 2.5, 5.0, 7.0, 16.0];
        let y = [-1.0, 3.0, 2.0, 6.0, 10.0, 25.0];
        let err2 = [0.5, 1.0, 0.3, 0.1, 0.9, 2.5];
        // scipy.optimize.curve_fit(absolute_sigma=True)
        let desired_slope = 1.60504579;
        let desired_slope_sigma2 = 0.00868733;
        let desired_reduced_chi2 = 1.8057513419557492;
        let result = fit_straight_line(&x, &y, Some(&err2));
        all_close(&[result.slope], &[desired_slope], 1e-6);
        all_close(&[result.slope_sigma2], &[desired_slope_sigma2], 1e-6);
        all_close(&[result.reduced_chi2], &[desired_reduced_chi2], 1e-6);
    }
}
