use std::cmp::PartialOrd;

/// Sorts multiple slices in order of the first one
///
/// # Examples
///
/// ```
/// use light_curve_common::sort_multiple;
///
/// let a = [1, 2, 3, 0, 4];
/// let b = [3, 2, 1, 4, 0];
///
/// let sorted = sort_multiple(&[&a, &b]);
/// let a_sorted = &sorted[0];
/// let b_sorted = &sorted[1];
///
/// assert_eq!(&vec![0, 1, 2, 3, 4], a_sorted);
/// assert_eq!(&vec![4, 3, 2, 1, 0], b_sorted);
/// ```
pub fn sort_multiple<T: PartialOrd + Copy>(slices: &[&[T]]) -> Vec<Vec<T>> {
    if slices.is_empty() {
        return vec![];
    }

    let reference = slices[0];
    let other = &slices[1..];

    let n = reference.len();
    for s in other {
        assert_eq!(s.len(), n);
    }

    let mut order: Vec<_> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| reference[a].partial_cmp(&reference[b]).unwrap());

    slices.iter().map(|s| reorder(s, &order)).collect()
}

fn reorder<T: Copy>(s: &[T], order: &[usize]) -> Vec<T> {
    order.iter().map(|&i| s[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::prelude::*;

    fn sort_via_zip_unzip<T: PartialOrd + Clone>(a: &[T], b: &[T]) -> Vec<Vec<T>> {
        let mut pairs: Vec<_> = a.iter().cloned().zip(b.iter().cloned()).collect();
        pairs.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let (sorted_a, sorted_b) = pairs.into_iter().unzip();
        vec![sorted_a, sorted_b]
    }

    #[test]
    fn two_random() {
        const N: usize = 100;

        let mut rng = rand::thread_rng();
        let a: Vec<f64> = (0..N).map(|_| rng.gen()).collect();
        let b: Vec<f64> = (0..N).map(|_| rng.gen()).collect();

        assert_eq!(sort_via_zip_unzip(&a, &b), sort_multiple(&[&a, &b]));
    }

    #[test]
    fn three() {
        let a = [5, 4, 3, 2, 1, 0];
        let b = [5, 4, 3, 2, 1, 0];
        let c = [0, 1, 2, 3, 4, 5];

        assert_eq!(
            vec![
                vec![0, 1, 2, 3, 4, 5],
                vec![0, 1, 2, 3, 4, 5],
                vec![5, 4, 3, 2, 1, 0]
            ],
            sort_multiple(&[&a, &b, &c])
        );
    }
}
