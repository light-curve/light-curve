use conv::prelude::*;
use num_traits::float::Float;

/// Creates uniform distributed vales
///
/// # Examples
///
/// ```
/// use light_curve_common::linspace;
///
/// let a = linspace(0.0, 1.0, 3);
///
/// assert_eq!(0.0, a[0]);
/// assert_eq!(0.5, a[1]);
/// assert_eq!(1.0, a[2]);
/// ```
pub fn linspace<T>(start: T, end: T, size: usize) -> Vec<T>
where
    T: Float + ValueFrom<usize>,
{
    let intervals: T = (size - 1).value_as::<T>().unwrap();
    let step = (end - start) / intervals;
    (0..size)
        .map(|i| start + step * i.value_as::<T>().unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::all_close;

    #[test]
    fn linspace_two_points() {
        let actual = [0_f32, 1_f32];
        let desired = linspace(0_f32, 1_f32, 2);
        all_close(&actual[..], &desired[..], 1e-6);
    }

    #[test]
    fn linspace_three_points() {
        let actual = [-1_f64, 0_f64, 1_f64];
        let desired = linspace(-1_f64, 1_f64, 3);
        all_close(&actual[..], &desired[..], 1e-12);
    }

    #[test]
    fn linspace_many_points() {
        let actual: Vec<_> = (0_u8..101).map(f32::from).collect();
        let desired = linspace(0_f32, 100_f32, 101);
        all_close(&actual[..], &desired[..], 1e-6);
    }
}
