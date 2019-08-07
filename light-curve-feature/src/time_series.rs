use conv::prelude::*;

use crate::float_trait::Float;
use crate::statistics::Statistics;

pub struct DataSample<'a, T>
where
    T: Float,
{
    pub(super) sample: &'a [T],
    sorted: Vec<T>,
    min: Option<T>,
    max: Option<T>,
    mean: Option<T>,
    median: Option<T>,
    std: Option<T>,
}

macro_rules! data_sample_getter {
    ($attr: ident, $getter: ident, $method: ident) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some(self.sample.$method());
                    self.$attr.unwrap()
                },
            }
        }
    };
    ($attr: ident, $getter: ident, $method: ident, $method_sorted: ident) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some(
                        if self.sorted.is_empty() {
                            self.sample.$method()
                        } else {
                            self.sorted[..].$method_sorted()
                        }
                    );
                    self.$attr.unwrap()
                },
            }
        }
    };
    ($attr: ident, $getter: ident, $func: expr) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some($func(self));
                    self.$attr.unwrap()
                },
            }
        }
    };
}

impl<'a, T> DataSample<'a, T>
where
    T: Float,
    [T]: Statistics<T>,
{
    fn new(sample: &'a [T]) -> Self {
        assert!(
            sample.len() > 1,
            "DataSample should has at least two points"
        );
        Self {
            sample,
            sorted: vec![],
            min: None,
            max: None,
            mean: None,
            median: None,
            std: None,
        }
    }

    pub(super) fn get_sorted<'c>(&'c mut self) -> &'c [T] {
        if self.sorted.is_empty() {
            self.sorted.extend(self.sample.sorted());
        }
        &self.sorted[..]
    }

    data_sample_getter!(min, get_min, minimum, min_from_sorted);
    data_sample_getter!(max, get_max, maximum, max_from_sorted);
    data_sample_getter!(mean, get_mean, mean);
    data_sample_getter!(median, get_median, |ds: &mut DataSample<'a, T>| {
        ds.get_sorted().median_from_sorted()
    });
    data_sample_getter!(std, get_std, |ds: &mut DataSample<'a, T>| {
        let mean = ds.get_mean();
        T::sqrt(
            ds.sample.iter().map(|&x| (x - mean).powi(2)).sum::<T>()
                / (ds.sample.len() - 1).value_as::<T>().unwrap(),
        )
    });

    pub fn signal_to_noise(&mut self, value: T) -> T {
        (value - self.get_mean()) / self.get_std()
    }
}

pub struct TimeSeries<'a, 'b, T>
where
    T: Float,
{
    pub(super) t: DataSample<'a, T>,
    pub(super) m: DataSample<'b, T>,
}

impl<'a, 'b, T> TimeSeries<'a, 'b, T>
where
    T: Float,
{
    pub fn new(t: &'a [T], m: &'b [T]) -> Self {
        assert_eq!(t.len(), m.len(), "t and m should have the same size");
        Self {
            t: DataSample::new(t),
            m: DataSample::new(m),
        }
    }

    pub fn lenu(&self) -> usize {
        self.t.sample.len()
    }

    pub fn lenf(&self) -> T {
        self.lenu().value_as::<T>().unwrap()
    }

    pub fn max_by_m(&self) -> (T, T) {
        self.t
            .sample
            .iter()
            .cloned()
            .zip(self.m.sample.iter().cloned())
            .max_by(|(_t_a, m_a), (_t_b, m_b)| m_a.partial_cmp(m_b).unwrap())
            .unwrap()
    }
}
