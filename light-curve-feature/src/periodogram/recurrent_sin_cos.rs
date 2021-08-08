//! Recurrent sine-cosine implementation

use crate::float_trait::Float;

/// Iterator over sin(kx), cos(kx) pairs
///
/// It starts with sin(x), cos(x) and yields values for 2x, 3x, ... while iterating
pub struct RecurrentSinCos<T> {
    first: (T, T),
    current: (T, T),
}

impl<T: Float> RecurrentSinCos<T> {
    /// Construct [RecurrentSinCos] from angle x
    pub fn new(x: T) -> Self {
        let first = x.sin_cos();
        Self {
            first,
            current: (T::zero(), T::one()),
        }
    }
}

impl<T: Float> Iterator for RecurrentSinCos<T> {
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.current = (
            self.first.0 * self.current.1 + self.first.1 * self.current.0,
            self.first.1 * self.current.1 - self.first.0 * self.current.0,
        );
        Some(self.current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use light_curve_common::all_close;

    #[test]
    fn recurrent_sin_cos() {
        let x = 0.01;
        const N: usize = 1000;
        let (desired_sin, desired_cos): (Vec<_>, Vec<_>) =
            (1..=N).map(|i| f64::sin_cos(x * (i as f64))).unzip();
        let (actual_sin, actual_cos): (Vec<_>, Vec<_>) = RecurrentSinCos::new(x).take(N).unzip();
        all_close(&actual_sin[..], &desired_sin[..], 1e-12);
        all_close(&actual_cos[..], &desired_cos[..], 1e-12);
    }
}
