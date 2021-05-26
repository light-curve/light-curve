use conv::*;
use itertools::Itertools;
use ndarray::{s, Array1, Array2, ScalarOperand};
use std::fmt::Debug;
use std::io::Write;
use std::marker::PhantomData;

mod erf;
pub use erf::{ErfFloat, ErrorFunction};

mod float_trait;
pub use float_trait::Float;

pub trait Normalisable:
    ApproxInto<u8, DefaultApprox>
    + ValueFrom<usize>
    + num_traits::Num
    + num_traits::NumOps
    + PartialOrd
    + ScalarOperand
    + Copy
{
    fn clamp(self, min: Self, max: Self) -> Self;
    fn max_u8() -> Self;
}

impl Normalisable for usize {
    fn clamp(self, min: Self, max: Self) -> Self {
        match self {
            _ if self < min => min,
            x if self <= max => x,
            _ => max,
        }
    }

    #[inline]
    fn max_u8() -> Self {
        255
    }
}

impl Normalisable for f32 {
    fn clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }

    #[inline]
    fn max_u8() -> Self {
        255.0
    }
}

impl Normalisable for f64 {
    fn clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }

    #[inline]
    fn max_u8() -> Self {
        255.0
    }
}

pub trait Grid<T>: Clone + Debug
where
    T: Copy,
{
    /// Cell borders coordinates, must be len() + 1 length Array
    fn get_borders(&self) -> &Array1<T>;

    /// Number of cells
    fn cell_count(&self) -> usize {
        self.get_borders().len() - 1
    }

    /// Coordinate of the left border of the leftmost cell
    fn get_start(&self) -> T {
        self.get_borders()[0]
    }

    /// Coordinate of the right border of the rightmost cell
    fn get_end(&self) -> T {
        self.get_borders()[self.cell_count()]
    }

    /// Get index of the cell containing given value
    fn idx(&self, x: T) -> CellIndex;
}

pub fn is_sorted<T>(a: &[T]) -> bool
where
    T: PartialOrd,
{
    a.iter().tuple_windows().all(|(a, b)| a < b)
}

#[derive(Clone, Debug)]
pub struct ArrayGrid<T>
where
    Array1<T>: Clone + Debug,
{
    borders: Array1<T>,
}

impl<T> ArrayGrid<T>
where
    Array1<T>: Clone + Debug,
    T: Float,
{
    pub fn new(borders: Array1<T>) -> Self {
        assert!(!borders.is_empty());
        assert!(is_sorted(borders.as_slice().unwrap()));
        Self { borders }
    }
}

impl<T> Grid<T> for ArrayGrid<T>
where
    Array1<T>: Clone + Debug,
    T: Float,
{
    #[inline]
    fn get_borders(&self) -> &Array1<T> {
        &self.borders
    }

    fn idx(&self, x: T) -> CellIndex {
        let i = self
            .borders
            .as_slice()
            .unwrap()
            .partition_point(|&b| b <= x);
        match i {
            0 => CellIndex::LowerMin,
            _ if i == self.borders.len() => CellIndex::GreaterMax,
            _ => CellIndex::Value(i - 1),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinearGrid<T>
where
    T: Copy,
{
    start: T,
    end: T,
    n: usize,
    cell_size: T,
    borders: Array1<T>,
}

impl<T> LinearGrid<T>
where
    T: Float,
{
    pub fn new(start: T, end: T, n: usize) -> Self {
        assert!(end > start);
        let cell_size = (end - start) / n.value_as::<T>().unwrap();
        let borders = Array1::linspace(start, end, n + 1);
        Self {
            start,
            end,
            n,
            cell_size,
            borders,
        }
    }

    #[inline]
    pub fn get_cell_size(&self) -> T {
        self.cell_size
    }
}

impl<T> Grid<T> for LinearGrid<T>
where
    T: Float,
{
    #[inline]
    fn get_borders(&self) -> &Array1<T> {
        &self.borders
    }

    #[inline]
    fn cell_count(&self) -> usize {
        self.n
    }

    #[inline]
    fn get_start(&self) -> T {
        self.start
    }

    #[inline]
    fn get_end(&self) -> T {
        self.end
    }

    fn idx(&self, x: T) -> CellIndex {
        if x < self.start {
            return CellIndex::LowerMin;
        }
        if x >= self.end {
            return CellIndex::GreaterMax;
        }
        let i = ((x - self.start) / self.cell_size)
            .approx_by::<RoundToZero>()
            .unwrap();
        if i < self.n {
            CellIndex::Value(i)
        } else {
            // x is a bit smaller self.end + float rounding
            CellIndex::Value(self.n - 1)
        }
    }
}

#[derive(Clone, Debug)]
pub struct LgGrid<T>
where
    T: Copy,
{
    start: T,
    end: T,
    lg_start: T,
    lg_end: T,
    n: usize,
    cell_lg_size: T,
    borders: Array1<T>,
}

impl<T> LgGrid<T>
where
    T: Float,
{
    pub fn new(start: T, end: T, n: usize) -> Self {
        assert!(end > start);
        assert!(start.is_positive());
        let lg_start = start.log10();
        let lg_end = end.log10();
        let cell_lg_size = (lg_end - lg_start) / n.value_as::<T>().unwrap();
        let mut borders = Array1::logspace(T::ten(), lg_start, lg_end, n + 1);
        borders[0] = start;
        borders[n] = end;
        Self {
            start,
            end,
            lg_start,
            lg_end,
            n,
            cell_lg_size,
            borders,
        }
    }

    #[inline]
    pub fn get_cell_lg_size(&self) -> T {
        self.cell_lg_size
    }

    #[inline]
    pub fn get_lg_start(&self) -> T {
        self.lg_start
    }

    #[inline]
    pub fn get_lg_end(&self) -> T {
        self.lg_end
    }
}

impl<T> Grid<T> for LgGrid<T>
where
    T: Float,
{
    #[inline]
    fn get_borders(&self) -> &Array1<T> {
        &self.borders
    }

    #[inline]
    fn cell_count(&self) -> usize {
        self.n
    }

    #[inline]
    fn get_start(&self) -> T {
        self.start
    }

    #[inline]
    fn get_end(&self) -> T {
        self.end
    }

    fn idx(&self, x: T) -> CellIndex {
        if x < self.start {
            return CellIndex::LowerMin;
        }
        if x >= self.end {
            return CellIndex::GreaterMax;
        }
        let i = ((x.log10() - self.lg_start) / self.cell_lg_size)
            .approx_by::<RoundToZero>()
            .unwrap();
        if i < self.n {
            CellIndex::Value(i)
        } else {
            // x is a bit smaller self.end + float rounding
            CellIndex::Value(self.n - 1)
        }
    }
}

pub enum CellIndex {
    LowerMin,
    GreaterMax,
    Value(usize),
}

#[derive(Clone, Debug)]
pub struct DmDt<Glgdt, Gdm, T>
where
    Glgdt: Grid<T>,
    Gdm: Grid<T>,
    T: Copy,
{
    pub lgdt_grid: Glgdt,
    pub dm_grid: Gdm,
    _phantom_data: PhantomData<T>,
}

impl<Glgdt, Gdm, T> DmDt<Glgdt, Gdm, T>
where
    Glgdt: Grid<T>,
    Gdm: Grid<T>,
    T: Float,
{
    pub fn new(lgdt_grid: Glgdt, dm_grid: Gdm) -> Self {
        Self {
            lgdt_grid,
            dm_grid,
            _phantom_data: PhantomData,
        }
    }

    /// N lg_dt by N dm
    pub fn shape(&self) -> (usize, usize) {
        (self.lgdt_grid.cell_count(), self.dm_grid.cell_count())
    }

    pub fn points(&self, t: &[T], m: &[T]) -> Array2<usize> {
        let mut a = Array2::zeros(self.shape());
        for (i1, (&x1, &y1)) in t.iter().zip(m.iter()).enumerate() {
            for (&x2, &y2) in t[i1 + 1..].iter().zip(m[i1 + 1..].iter()) {
                let lgdt = T::log10(x2 - x1);
                let idx_lgdt = match self.lgdt_grid.idx(lgdt) {
                    CellIndex::LowerMin => continue,
                    CellIndex::GreaterMax => break,
                    CellIndex::Value(idx_lgdt) => idx_lgdt,
                };
                let dm = y2 - y1;
                let idx_dm = match self.dm_grid.idx(dm) {
                    CellIndex::Value(idx_dm) => idx_dm,
                    CellIndex::LowerMin | CellIndex::GreaterMax => continue,
                };
                a[(idx_lgdt, idx_dm)] += 1;
            }
        }
        a
    }

    pub fn gausses(&self, t: &[T], m: &[T], err2: &[T], erf: &ErrorFunction) -> Array2<T>
    where
        T: ErfFloat,
    {
        let mut a = Array2::zeros(self.shape());
        for (i1, ((&x1, &y1), &d1)) in t.iter().zip(m.iter()).zip(err2.iter()).enumerate() {
            for ((&x2, &y2), &d2) in t[i1 + 1..]
                .iter()
                .zip(m[i1 + 1..].iter())
                .zip(err2[i1 + 1..].iter())
            {
                let lgdt = T::log10(x2 - x1);
                let idx_lgdt = match self.lgdt_grid.idx(lgdt) {
                    CellIndex::LowerMin => continue,
                    CellIndex::GreaterMax => break,
                    CellIndex::Value(idx_lgdt) => idx_lgdt,
                };
                let dm = y2 - y1;
                let dm_err = T::sqrt(d1 + d2);

                let min_idx_dm = match self.dm_grid.idx(dm + erf.min_dx_nonzero_normal_cdf(dm_err))
                {
                    CellIndex::LowerMin => 0,
                    CellIndex::GreaterMax => continue,
                    CellIndex::Value(min_idx_dm) => min_idx_dm,
                };
                let max_idx_dm = match self
                    .dm_grid
                    .idx(dm + erf.max_dx_nonunity_normal_cdf(dm_err))
                {
                    CellIndex::LowerMin => continue,
                    CellIndex::GreaterMax => self.dm_grid.cell_count(),
                    CellIndex::Value(i) => usize::min(i + 1, self.dm_grid.cell_count()),
                };

                a.slice_mut(s![idx_lgdt, min_idx_dm..max_idx_dm])
                    .iter_mut()
                    .zip(
                        self.dm_grid
                            .get_borders()
                            .slice(s![min_idx_dm..max_idx_dm + 1])
                            .iter()
                            .map(|&dm_border| erf.normal_cdf(dm_border, dm, dm_err))
                            .tuple_windows()
                            .map(|(a, b)| b - a),
                    )
                    .for_each(|(cell, value)| *cell += value);
            }
        }
        a
    }

    pub fn lgdt_points(&self, t: &[T]) -> Array1<usize> {
        let mut a = Array1::zeros(self.lgdt_grid.cell_count());
        for (i1, &x1) in t.iter().enumerate() {
            for &x2 in t[i1 + 1..].iter() {
                let lgdt = T::log10(x2 - x1);
                let idx_lgdt = match self.lgdt_grid.idx(lgdt) {
                    CellIndex::LowerMin => continue,
                    CellIndex::GreaterMax => break,
                    CellIndex::Value(idx_lgdt) => idx_lgdt,
                };
                a[idx_lgdt] += 1;
            }
        }
        a
    }
}

pub fn to_png<W>(w: W, a: &Array2<u8>) -> Result<(), png::EncodingError>
where
    W: Write,
{
    let transposed = {
        let mut b = Array2::zeros((a.ncols(), a.nrows()));
        b.assign(&a.t());
        b
    };
    let mut encoder = png::Encoder::new(w, transposed.ncols() as u32, transposed.nrows() as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(transposed.as_slice_memory_order().unwrap())?;
    Ok(())
}
