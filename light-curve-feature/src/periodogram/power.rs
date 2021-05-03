use crate::float_trait::Float;
use crate::periodogram::freq::FreqGrid;
use crate::time_series::TimeSeries;
use dyn_clonable::*;
use std::fmt::Debug;

/// Periodogram execution algorithm
#[clonable]
pub trait PeriodogramPower<T: Float>: Debug + Clone + Send {
    fn power(&self, freq: &FreqGrid<T>, ts: &mut TimeSeries<T>) -> Vec<T>;
}
