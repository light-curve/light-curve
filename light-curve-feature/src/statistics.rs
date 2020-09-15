use crate::float_trait::Float;
use conv::prelude::*;

pub trait Statistics<T>
where
    T: Float,
{
    fn minimum(&self) -> T;
    fn maximum(&self) -> T;
    fn mean(&self) -> T;
    fn peak_indices(&self) -> Vec<usize>;
    fn peak_indices_reverse_sorted(&self) -> Vec<usize>;
}

impl<T> Statistics<T> for [T]
where
    T: Float,
{
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

    fn mean(&self) -> T {
        self.iter().copied().sum::<T>() / self.len().value_as::<T>().unwrap()
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
        v[..].sort_unstable_by(|&b, &a| {
            unsafe { self.get_unchecked(a) }
                .partial_cmp(unsafe { self.get_unchecked(b) })
                .unwrap()
        });
        v
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use light_curve_common::linspace;

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
