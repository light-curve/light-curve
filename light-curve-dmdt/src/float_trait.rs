use conv::*;

/// Float number trait, it is implemented for [f32] and [f64] only
pub trait Float:
    ndarray::NdFloat
    + num_traits::FloatConst
    + num_traits::Signed
    + num_traits::Float
    + ValueFrom<usize>
    + ApproxInto<usize, RoundToZero>
    + ApproxFrom<u64>
{
    fn half() -> Self;
    fn ten() -> Self;
}

impl Float for f32 {
    fn half() -> Self {
        0.5
    }

    fn ten() -> Self {
        10.0
    }
}
impl Float for f64 {
    fn half() -> Self {
        0.5
    }

    fn ten() -> Self {
        10.0
    }
}
