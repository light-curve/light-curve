use crate::float_trait::Float;
use crate::periodogram::freq::FreqGrid;
use crate::time_series::TimeSeries;
use std::fmt::Debug;

pub trait PeriodogramPower<T: Float>: Debug {
    fn power(&self, freq: &FreqGrid<T>, ts: &mut TimeSeries<T>) -> Vec<T>;
}
