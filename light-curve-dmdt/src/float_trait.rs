use conv::*;

pub trait Float:
    ndarray::NdFloat
    + num_traits::FloatConst
    + num_traits::Signed
    + num_traits::Float
    + ValueFrom<usize>
    + ValueFrom<u64>
    + ApproxInto<usize, RoundToZero>
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
