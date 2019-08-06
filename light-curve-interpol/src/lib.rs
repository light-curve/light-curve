use conv::prelude::*;
use num_traits::{float::Float, identities};

pub struct Interpolator<T, U>
where
    T: Float,
    U: Float + ValueFrom<T>,
{
    pub target_x: Vec<T>,
    pub left: U,
    pub right: U,
}

#[derive(Clone)]
struct Knot<T, U>
where
    T: Float,
    U: Float + ValueFrom<T>,
{
    x: T,
    y: U,
}

type Interval<T, U> = (Option<Knot<T, U>>, Option<Knot<T, U>>);

impl<T, U> Interpolator<T, U>
where
    T: Float,
    U: Float + ValueFrom<T>,
{
    pub fn interpolate(&self, x: &[T], y: &[U]) -> Vec<U> {
        assert_eq!(x.len(), y.len(), "x and y should have same size");
        assert!(x.len() > 0, "input arrays should have not zero-length");
        let mut interval: Interval<T, U> = (None, None);
        let mut interval_iter = x
            .iter()
            .zip(y.iter())
            .map(|(&x, &y)| -> Interval<T, U> {
                interval.0 = interval.1.clone();
                interval.1 = Some(Knot { x, y });
                interval.clone()
            })
            .peekable();
        self.target_x
            .iter()
            .map(|&target| {
                while let Some(int) = interval_iter.peek() {
                    match int {
                        (_, None) => panic!("There should not be interval with None right value"),
                        (_, Some(right)) if target > right.x => {
                            interval_iter.next();
                            continue;
                        }
                        (_, Some(right)) if target == right.x => return right.y,
                        (None, Some(right)) if target < right.x => return self.left,
                        (None, Some(_)) => {
                            panic!("This case should be covered by another statements")
                        }
                        (Some(left), Some(right)) => {
                            let alpha = ((right.x - target) / (right.x - left.x))
                                .value_as::<U>()
                                .unwrap();
                            return alpha * left.y + (identities::one::<U>() - alpha) * right.y;
                        }
                    }
                }
                self.right
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use light_curve_common::all_close;

    #[test]
    fn interpolation_empty_target_x() {
        let initial_x = [0_f32, 1.];
        let initial_y = [0_f32, 1.];
        let target_x = vec![];
        let desired = Interpolator {
            target_x,
            left: 0.,
            right: 0.,
        }
        .interpolate(&initial_x[..], &initial_y[..]);
        assert_eq!(0, desired.len());
    }

    #[test]
    #[should_panic]
    fn interpolation_zero_points() {
        let initial_x = [];
        let initial_y = [];
        let target_x = vec![1_f32, 2., 3.];
        Interpolator {
            target_x,
            left: 0.,
            right: 0.,
        }
        .interpolate(&initial_x[..], &initial_y[..]);
    }

    #[test]
    fn interpolation_one_point() {
        let left = -3_f32;
        let right = -8_f32;
        let initial_x = [0_f32];
        let initial_y = [1_f32];
        let target_x = vec![-2_f32, -1., 0., 1., 2.];
        let actual = [left, left, 1., right, right];
        let desired = Interpolator {
            target_x,
            left,
            right,
        }
        .interpolate(&initial_x[..], &initial_y[..]);
        all_close(&actual[..], &desired[..], 1e-6);
    }

    #[test]
    fn interpolation_two_points() {
        let left = -3_f32;
        let right = -8_f32;
        let initial_x = [0_f32, 1.];
        let initial_y = [1_f32, 2.];
        let target_x = vec![-1.25_f32, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75];
        let actual = [left, left, left, 1.25, 1.75, right, right];
        let desired = Interpolator {
            target_x,
            left,
            right,
        }
        .interpolate(&initial_x[..], &initial_y[..]);
        all_close(&actual[..], &desired[..], 1e-6);
    }

    #[test]
    fn interpolate_equal_time() {
        let initial_x = [0_f32, 1., 1., 2.];
        let initial_y = [1_f32, 2., 3., 4.];
        let target_x = vec![0.5, 1.5];
        let actual = [1.5, 3.5];
        let desired = Interpolator {
            target_x,
            left: -1.,
            right: -1.,
        }
        .interpolate(&initial_x[..], &initial_y[..]);
        all_close(&actual[..], &desired[..], 1e-6);
    }

    #[test]
    fn interpolation_broken_line() {
        let initial_x = [-7. / 3., -4. / 3., -1., -1. / 3., 1., 4. / 3., 7. / 3.];
        let initial_y = [2.5, 2., 0.5, 1., 0.5, 1.5, 3.];
        let target_x = vec![
            -2.,
            -5. / 3.,
            -4. / 3.,
            -1.,
            -2. / 3.,
            -1. / 3.,
            0.,
            1. / 3.,
            2. / 3.,
            1.,
            4. / 3.,
            5. / 3.,
            2.,
        ];
        let actual = [
            7. / 3.,
            13. / 6.,
            2.,
            0.5,
            0.75,
            1.,
            0.875,
            0.75,
            0.625,
            0.5,
            1.5,
            2.,
            2.5,
        ];
        let desired = Interpolator {
            target_x,
            left: -1.,
            right: -1.,
        }
        .interpolate(&initial_x[..], &initial_y[..]);
        all_close(&actual[..], &desired[..], 1e-6);
    }
}
