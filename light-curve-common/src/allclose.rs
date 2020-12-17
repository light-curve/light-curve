use std::fmt::Debug;

use num_traits::float::Float;

/// Panics if two float slices are not close with respect to some absolute tolerance
///
/// # Examples
///
/// ```
/// use light_curve_common::all_close;
///
/// all_close(&[0.0, 1.0], &[0.0001, 0.9998], 1e-2);
/// ```
///
/// ```should_panic
/// use light_curve_common::all_close;
///
/// all_close(&[0.0], &[0.0, 1.0], 1e-6);
/// ```
///
/// ```should_panic
/// use light_curve_common::all_close;
///
/// all_close(&[1e-3, 1.0], &[0.0, 1.0], 1e-4);
/// ```
pub fn all_close<T>(actual: &[T], desired: &[T], tol: T)
where
    T: Float + Debug,
{
    assert_eq!(actual.len(), desired.len());
    let is_close = actual
        .iter()
        .cloned()
        .zip(desired.iter().cloned())
        .all(|(x, y)| (x - y).abs() < tol);
    assert!(
        is_close,
        "Slices are not close:\n{:?}\n{:?}\n",
        actual, desired
    );
}
