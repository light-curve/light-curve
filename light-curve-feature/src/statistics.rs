use conv::prelude::*;

use crate::float_trait::Float;

pub trait Statistics<T>
where
    T: Float,
{
    fn sorted(&self) -> Vec<T>;
    fn minimum(&self) -> T;
    fn maximum(&self) -> T;
    fn min_from_sorted(&self) -> T;
    fn max_from_sorted(&self) -> T;
    fn mean(&self) -> T;
    fn median(&self) -> T;
    fn median_from_sorted(&self) -> T;
    fn ppf(&self, q: f32) -> T;
    fn ppf_from_sorted(&self, q: f32) -> T;
    fn ppf_many(&self, q: &[f32]) -> Vec<T>;
    fn ppf_many_from_sorted(&self, q: &[f32]) -> Vec<T>;
}

impl<T> Statistics<T> for [T]
where
    T: Float,
{
    fn sorted(&self) -> Vec<T> {
        let mut v = self.to_vec();
        // Replace with partition_at_index when it will be available
        v[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    fn minimum(&self) -> T {
        *self
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn maximum(&self) -> T {
        *self
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn min_from_sorted(&self) -> T {
        self[0]
    }

    fn max_from_sorted(&self) -> T {
        self[self.len() - 1]
    }

    fn mean(&self) -> T {
        self.iter().cloned().sum::<T>() / self.len().value_as::<T>().unwrap()
    }

    fn median(&self) -> T {
        // Replace with partition_at_index when it will be available
        self.sorted().median_from_sorted()
    }

    fn median_from_sorted(&self) -> T {
        let i = (self.len() - 1) / 2;
        if self.len() % 2 == 0 {
            T::half() * (self[i] + self[i + 1])
        } else {
            self[i]
        }
    }

    fn ppf(&self, q: f32) -> T {
        // Replace with partition_at_index when it will be available
        self.sorted().ppf_from_sorted(q)
    }

    fn ppf_from_sorted(&self, q: f32) -> T {
        // R-5 from https://en.wikipedia.org/wiki/Quantile
        assert!(
            (q >= 0.0) && (q <= 1.0),
            "quantile should be between zero and unity"
        );
        let h = (self.len() as f32) * q - 0.5;
        let h_floor = h.floor();
        if h_floor < 0.0 {
            self[0]
        } else {
            let i = h_floor as usize;
            if i >= self.len() - 1 {
                self[self.len() - 1]
            } else {
                self[i] + (h - h_floor).value_as::<T>().unwrap() * (self[i + 1] - self[i])
            }
        }
    }

    fn ppf_many(&self, q: &[f32]) -> Vec<T> {
        self.sorted()[..].ppf_many_from_sorted(q)
    }

    fn ppf_many_from_sorted(&self, q: &[f32]) -> Vec<T> {
        q.iter().map(|&x| self.ppf_from_sorted(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use rand;

    use super::*;

    #[test]
    fn median_is_ppf_half() {
        for i in 0..10 {
            let a: Vec<f32> = (0..100 + i).map(|_| rand::random()).collect();
            assert_eq!(a[..].median(), a[..].ppf(0.5));
        }
    }

    #[test]
    fn minimum_is_ppf_zero() {
        for i in 0..10 {
            let a: Vec<f32> = (0..100 + i).map(|_| rand::random()).collect();
            assert_eq!(a[..].minimum(), a[..].ppf(0.0));
        }
    }

    #[test]
    fn maximum_is_ppf_unity() {
        for i in 0..10 {
            let a: Vec<f32> = (0..100 + i).map(|_| rand::random()).collect();
            assert_eq!(a[..].maximum(), a[..].ppf(1.0));
        }
    }
}
