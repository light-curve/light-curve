use crate::float_trait::Float;
use conv::ConvUtil;

// See Press et al. sec. 15.2 Fitting Data to a Straight Line, p. 661

#[allow(clippy::many_single_char_names)]
pub fn fit_straight_line<T: Float>(
    x: &[T],
    y: &[T],
    err2: Option<&[T]>,
) -> StraightLineFitterResult<T> {
    let n = x.len().value_as::<T>().unwrap();
    let weight = err2.map(|err2| err2.iter().map(|e| e.recip()).collect::<Vec<_>>());
    let (s, sx, sy) = match weight.as_ref() {
        Some(weight) => {
            let (mut s, mut sx, mut sy) = (T::zero(), T::zero(), T::zero());
            for ((&x, &y), &w) in x.iter().zip(y.iter()).zip(weight.iter()) {
                s += w;
                sx += w * x;
                sy += w * y;
            }
            (s, sx, sy)
        }
        None => {
            let (mut sx, mut sy) = (T::zero(), T::zero());
            for (&x, &y) in x.iter().zip(y.iter()) {
                sx += x;
                sy += y;
            }
            (n, sx, sy)
        }
    };
    let (stt, sty) = match weight.as_ref() {
        Some(weight) => {
            let (mut stt, mut sty) = (T::zero(), T::zero());
            for ((&x, &y), &w) in x.iter().zip(y.iter()).zip(weight.iter()) {
                let t = x - sx / s;
                stt += w * t.powi(2);
                sty += w * t * y;
            }
            (stt, sty)
        }
        None => {
            let (mut stt, mut sty) = (T::zero(), T::zero());
            for (&x, &y) in x.iter().zip(y.iter()) {
                let t = x - sx / s;
                stt += t.powi(2);
                sty += t * y;
            }
            (stt, sty)
        }
    };
    let slope = sty / stt;
    let intercept = (sy - sx * slope) / s;
    let mut slope_sigma2 = stt.recip();
    let chi2: T = match weight.as_ref() {
        Some(weight) => x
            .iter()
            .zip(y.iter())
            .zip(weight.iter())
            .map(|((&x, &y), &w)| (y - intercept - slope * x).powi(2) * w)
            .sum(),
        None => x
            .iter()
            .zip(y.iter())
            .map(|(&x, &y)| (y - intercept - slope * x).powi(2))
            .sum(),
    };
    let reduced_chi2 = chi2 / (n - T::two());
    if err2.is_none() {
        slope_sigma2 *= reduced_chi2;
    }
    StraightLineFitterResult {
        slope,
        slope_sigma2,
        reduced_chi2,
    }
}

pub struct StraightLineFitterResult<T> {
    pub slope: T,
    pub slope_sigma2: T,
    pub reduced_chi2: T,
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
        let desired_reduced_chi2 = 1.271190781049937;
        let result = fit_straight_line(&x, &y, None);
        all_close(&[result.slope], &[desired_slope], 1e-6);
        all_close(&[result.slope_sigma2], &[desired_slope_sigma2], 1e-6);
        all_close(&[result.reduced_chi2], &[desired_reduced_chi2], 1e-6);
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
