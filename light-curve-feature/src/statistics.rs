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
    fn peak_indices(&self) -> Vec<usize>;
    fn peak_indices_reverse_sorted(&self) -> Vec<usize>;
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

    /// Indices of local maxima, edge points are never included
    fn peak_indices(&self) -> Vec<usize> {
        self.iter()
            .enumerate()
            .fold(
                (vec![], T::infinity(), false),
                |(mut v, prev_x, prev_is_rising), (i, &x)| {
                    let is_rising = x > prev_x;
                    if prev_is_rising && (!is_rising) {
                        v.push(i - 1)
                    }
                    (v, x, is_rising)
                },
            )
            .0
    }

    fn peak_indices_reverse_sorted(&self) -> Vec<usize> {
        let mut v = self.peak_indices();
        v[..].sort_unstable_by(|&b, &a| self[a].partial_cmp(&self[b]).unwrap());
        v
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use rand;

    use super::*;

    use light_curve_common::linspace;

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

    macro_rules! peak_indices {
        ($name: ident, $desired: expr, $x: expr $(,)?) => {
            #[test]
            fn $name() {
                assert_eq!($x.peak_indices_reverse_sorted(), $desired);
            }
        };
    }

    peak_indices!(
        peak_indices_three_points_peak,
        [1_usize],
        [0.0_f32, 1.0, 0.0]
    );
    peak_indices!(
        peak_indices_three_points_plateau,
        [] as [usize; 0],
        [0.0_f32, 0.0, 0.0]
    );
    peak_indices!(
        peak_indices_three_points_dip,
        [] as [usize; 0],
        [0.0_f32, -1.0, 0.0]
    );
    peak_indices!(peak_indices_long_plateau, [] as [usize; 0], [0.0_f32; 100]);
    peak_indices!(
        peak_indices_sawtooth,
        (1..=99) // the first and the last point cannot be peak
            .filter(|i| i % 2 == 0)
            .collect::<Vec<_>>(),
        (0..=100)
            .map(|i| {
                if i % 2 == 0 {
                    1.0_f32
                } else {
                    0.0_f32
                }
            })
            .collect::<Vec<_>>(),
    );
    peak_indices!(
        peak_indices_one_peak,
        [50],
        linspace(-5.0_f32, 5.0, 101)
            .iter()
            .map(|&x| f32::exp(-0.5 * x * x))
            .collect::<Vec<_>>(),
    );
}
