use crate::float_trait::Float;
use crate::periodogram::freq::FreqGrid;
use crate::time_series::TimeSeries;

use enum_dispatch::enum_dispatch;
use std::fmt::Debug;

/// Periodogram execution algorithm
#[enum_dispatch]
pub trait PeriodogramPowerTrait<T>: Debug + Clone + Send
where
    T: Float,
{
    fn power(&self, freq: &FreqGrid<T>, ts: &mut TimeSeries<T>) -> Vec<T>;
}
