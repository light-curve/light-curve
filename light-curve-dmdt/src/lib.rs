use conv::*;
use itertools::Itertools;
use libm;
use ndarray::{Array1, Array2, NdFloat, ScalarOperand};
use num_traits;
use png;
use std::io::Write;

pub trait Normalisable:
    ApproxInto<u8, DefaultApprox>
    + num_traits::Num
    + num_traits::NumOps
    + PartialOrd
    + ScalarOperand
    + Copy
{
    fn max_u8() -> Self;
}

impl Normalisable for usize {
    #[inline]
    fn max_u8() -> Self {
        255
    }
}

impl Normalisable for f32 {
    #[inline]
    fn max_u8() -> Self {
        255.0
    }
}

impl Normalisable for f64 {
    #[inline]
    fn max_u8() -> Self {
        255.0
    }
}

pub trait Float:
    NdFloat + num_traits::FloatConst + ValueFrom<usize> + ApproxInto<usize, RoundToZero>
{
    fn erf(self) -> Self;

    fn half() -> Self;
}

impl Float for f32 {
    fn erf(self) -> Self {
        libm::erff(self)
    }

    fn half() -> Self {
        0.5
    }
}
impl Float for f64 {
    fn erf(self) -> Self {
        libm::erf(self)
    }

    fn half() -> Self {
        0.5
    }
}

fn normal_cdf<T>(x: T, mean: T, w: T) -> T
where
    T: Float,
{
    let inv_sigma = T::sqrt(w);
    T::half() * (T::one() + T::erf((x - mean) * inv_sigma * T::FRAC_1_SQRT_2()))
}

pub struct Grid<T> {
    start: T, // coordinate of the left border of the leftmost cell
    end: T,   // coordinate of the right border of the rightmost cell
    n: usize,
    length: T, // distance from the left border of the leftmost cell to the right border of the rightmost cell
    cell_size: T,
    borders: Array1<T>,
}

impl<T> Grid<T>
where
    T: Float,
{
    pub fn new(start: T, end: T, n: usize) -> Self {
        let length = end - start;
        assert!(length.is_sign_positive());
        let cell_size = (end - start) / n.value_as::<T>().unwrap();
        let borders = Array1::linspace(start, end, n + 1);
        Self {
            start,
            end,
            n,
            length,
            cell_size,
            borders,
        }
    }

    fn idx(&self, x: T) -> CellIndex {
        match x {
            _ if x < self.start => CellIndex::LowerMin,
            _ if x >= self.end => CellIndex::GreaterMax,
            _ => CellIndex::Value(((x - self.start) / self.cell_size).approx_into().unwrap()),
        }
    }
}

enum CellIndex {
    LowerMin,
    GreaterMax,
    Value(usize),
}

pub struct DmDt<T> {
    pub lgdt_grid: Grid<T>,
    pub dm_grid: Grid<T>,
}

impl<T> DmDt<T>
where
    T: Float,
{
    pub fn convert_lc_to_points(&self, t: &[T], m: &[T]) -> Array2<usize> {
        let mut a = Array2::zeros((self.lgdt_grid.n, self.dm_grid.n));
        for (i1, (&x1, &y1)) in t.iter().zip(m.iter()).enumerate() {
            for (&x2, &y2) in t[i1 + 1..].iter().zip(m[i1 + 1..].iter()) {
                let lgdt = T::log10(x2 - x1);
                let idx_lgdt = match self.lgdt_grid.idx(lgdt) {
                    CellIndex::LowerMin => continue,
                    CellIndex::GreaterMax => break,
                    CellIndex::Value(i) => i,
                };
                let dm = y2 - y1;
                let idx_dm = match self.dm_grid.idx(dm) {
                    CellIndex::Value(i) => i,
                    CellIndex::LowerMin | CellIndex::GreaterMax => continue,
                };
                a[(idx_lgdt, idx_dm)] += 1;
            }
        }
        a
    }

    pub fn convert_lc_to_gausses(&self, t: &[T], m: &[T], w: &[T]) -> Array2<T> {
        let mut a = Array2::zeros((self.lgdt_grid.n, self.dm_grid.n));
        for (i1, ((&x1, &y1), &dm_w1)) in t.iter().zip(m.iter()).zip(w.iter()).enumerate() {
            for ((&x2, &y2), &dm_w2) in t[i1 + 1..]
                .iter()
                .zip(m[i1 + 1..].iter())
                .zip(w[i1 + 1..].iter())
            {
                let lgdt = T::log10(x2 - x1);
                let idx_lgdt = match self.lgdt_grid.idx(lgdt) {
                    CellIndex::LowerMin => continue,
                    CellIndex::GreaterMax => break,
                    CellIndex::Value(i) => i,
                };
                let dm = y2 - y1;
                let dm_w = dm_w1 + dm_w2;
                a.row_mut(idx_lgdt)
                    .iter_mut()
                    .zip(
                        self.dm_grid
                            .borders
                            .iter()
                            .map(|&dm_border| normal_cdf(dm_border, dm, dm_w))
                            .tuple_windows()
                            .map(|(a, b)| b - a),
                    )
                    .for_each(|(cell, value)| *cell += value);
            }
        }
        a
    }
}

pub fn normalise<T>(a: &Array2<T>) -> Array2<u8>
where
    T: Normalisable,
{
    let max = *a.iter().max_by(|&x, &y| x.partial_cmp(y).unwrap()).unwrap();
    if max.is_zero() {
        Array2::zeros((a.nrows(), a.ncols()))
    } else {
        let normalised = a * T::max_u8() / max;
        normalised.mapv(|x| x.approx_into().unwrap())
    }
}

pub fn to_png<W>(w: W, a: &Array2<u8>) -> Result<(), png::EncodingError>
where
    W: Write,
{
    let mut encoder = png::Encoder::new(w, a.nrows() as u32, a.ncols() as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(a.as_slice().unwrap())?;
    Ok(())
}
