use crate::errors::{Exception, Res};
use itertools::Itertools;

pub fn is_sorted<T>(a: &[T]) -> bool
where
    T: PartialOrd,
{
    a.iter().tuple_windows().all(|(a, b)| a < b)
}

pub fn check_sorted<T>(a: &[T], sorted: Option<bool>) -> Res<()>
where
    T: PartialOrd,
{
    match sorted {
        Some(true) => Ok(()),
        Some(false) => Err(Exception::NotImplementedError(String::from(
            "sorting is not implemented, please provide time-sorted arrays",
        ))),
        None => {
            if is_sorted(a) {
                Ok(())
            } else {
                Err(Exception::ValueError(String::from(
                    "t must be in ascending order",
                )))
            }
        }
    }
}
