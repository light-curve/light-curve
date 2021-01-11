use crate::periodogram::FloatSupportedByFft;
use conv::prelude::*;
use num_traits::{float::Float as NumFloat, float::FloatConst};
use std::cmp::PartialOrd;
use std::fmt::{Debug, Display, LowerExp};
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign};

pub trait Float:
    'static
    + NumFloat
    + FloatConst
    + PartialOrd
    + Sum
    + ValueFrom<usize>
    + ValueFrom<f32>
    + ValueInto<f64>
    + ApproxFrom<usize>
    + ApproxFrom<f64>
    + ApproxInto<usize, RoundToNearest>
    + ApproxInto<f64>
    + Clone
    + Copy
    + Send
    + Sync
    + AddAssign
    + MulAssign
    + DivAssign
    + Display
    + Debug
    + LowerExp
    + FloatSupportedByFft
{
    fn half() -> Self;
    fn two() -> Self;
    fn three() -> Self;
    fn four() -> Self;
    fn five() -> Self;
    fn ten() -> Self;
    fn hundred() -> Self;
}

impl Float for f32 {
    #[inline]
    fn half() -> Self {
        0.5
    }

    #[inline]
    fn two() -> Self {
        2.0
    }

    #[inline]
    fn three() -> Self {
        3.0
    }

    #[inline]
    fn four() -> Self {
        4.0
    }

    #[inline]
    fn five() -> Self {
        5.0
    }

    #[inline]
    fn ten() -> Self {
        10.0
    }

    #[inline]
    fn hundred() -> Self {
        100.0
    }
}

impl Float for f64 {
    #[inline]
    fn half() -> Self {
        0.5
    }

    #[inline]
    fn two() -> Self {
        2.0
    }

    #[inline]
    fn three() -> Self {
        3.0
    }

    #[inline]
    fn four() -> Self {
        4.0
    }

    #[inline]
    fn five() -> Self {
        5.0
    }

    #[inline]
    fn ten() -> Self {
        10.0
    }

    #[inline]
    fn hundred() -> Self {
        100.0
    }
}
