use crate::error::SortedVecError;
use crate::float_trait::Float;
use conv::prelude::*;
use itertools::Itertools;
use std::ops::Deref;

#[derive(Clone, Debug)]
pub struct SortedVec<T>(pub Vec<T>);

fn is_sorted<T: Float>(x: &[T]) -> bool {
    x.iter().tuple_windows().all(|(&a, &b)| a < b)
}

impl<T> SortedVec<T>
where
    T: Float,
{
    pub fn new(x: Vec<T>) -> Result<Self, SortedVecError> {
        // Replace with Slice::is_sorted when it stabilizes
        // https://github.com/rust-lang/rust/issues/53485
        if is_sorted(&x) {
            Ok(Self(x))
        } else {
            Err(SortedVecError::Unsorted)
        }
    }

    pub fn maximum(&self) -> T {
        *self.last().unwrap()
    }

    pub fn minimum(&self) -> T {
        *self.first().unwrap()
    }

    pub fn median(&self) -> T {
        assert_ne!(self.len(), 0);
        let i = (self.len() - 1) / 2;
        if self.len() % 2 == 0 {
            T::half() * (*unsafe { self.get_unchecked(i) } + *unsafe { self.get_unchecked(i + 1) })
        } else {
            *unsafe { self.get_unchecked(i) }
        }
    }

    // R-5 from https://en.wikipedia.org/wiki/Quantile
    pub fn ppf(&self, q: f32) -> T {
        assert_ne!(self.len(), 0);
        assert!(
            (q >= 0.0) && (q <= 1.0),
            "quantile should be between zero and unity"
        );
        let h = (self.len() as f32) * q - 0.5;
        let h_floor = h.floor();
        if h_floor < 0.0 {
            *unsafe { self.get_unchecked(0) }
        } else {
            let i = h_floor as usize;
            if i >= self.len() - 1 {
                *unsafe { self.get_unchecked(self.len() - 1) }
            } else {
                *unsafe { self.get_unchecked(i) }
                    + (h - h_floor).value_as::<T>().unwrap()
                        * (*unsafe { self.get_unchecked(i + 1) }
                            - *unsafe { self.get_unchecked(i) })
            }
        }
    }
}

impl<T> From<Vec<T>> for SortedVec<T>
where
    T: Float,
{
    fn from(mut v: Vec<T>) -> Self {
        v[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        Self(v)
    }
}

impl<T> From<&[T]> for SortedVec<T>
where
    T: Float,
{
    fn from(s: &[T]) -> Self {
        s.to_vec().into()
    }
}

impl<T> Deref for SortedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[allow(clippy::float_cmp)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_is_ppf_half() {
        for i in 0..10 {
            let a: SortedVec<f64> = (0..100 + i)
                .map(|_| rand::random())
                .collect::<Vec<_>>()
                .into();
            assert_eq!(a.median(), a.ppf(0.5));
        }
    }

    #[test]
    fn minimum_is_ppf_zero() {
        for i in 0..10 {
            let a: SortedVec<f64> = (0..100 + i)
                .map(|_| rand::random())
                .collect::<Vec<_>>()
                .into();
            assert_eq!(a.minimum(), a.ppf(0.0));
        }
    }

    #[test]
    fn maximum_is_ppf_unity() {
        for i in 0..10 {
            let a: SortedVec<f32> = (0..100 + i)
                .map(|_| rand::random())
                .collect::<Vec<_>>()
                .into();
            assert_eq!(a.maximum(), a.ppf(1.0));
        }
    }
}
