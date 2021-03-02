use conv::*;

pub trait Float:
    ndarray::NdFloat + num_traits::FloatConst + ValueFrom<usize> + ApproxInto<usize, RoundToZero>
{
    fn half() -> Self;
}

impl Float for f32 {
    fn half() -> Self {
        0.5
    }
}
impl Float for f64 {
    fn half() -> Self {
        0.5
    }
}
