#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("example_png", "example.png")))]
#![doc = include_str!("../README.md")]

#[cfg(test)]
#[macro_use]
extern crate static_assertions;

use conv::*;
use dyn_clonable::*;
use itertools::Itertools;
use ndarray::{s, Array1, Array2};
use std::fmt::Debug;
use std::io::Write;
use thiserror::Error;

mod erf;
pub use erf::{Eps1Over1e3Erf, ErfFloat, ErrorFunction, ExactErf};

mod float_trait;
pub use float_trait::Float;

/// Grid for dm or dt axis
#[clonable]
pub trait Grid<T>: Clone + Debug + Send + Sync
where
    T: Copy,
{
    /// Cell borders coordinates, [cell_count()](Grid::cell_count) + 1 length [Array1]
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
    ///
    /// Note that cells include their left borders but doesn't include right borders
    fn idx(&self, x: T) -> CellIndex;
}

/// Checks if slice is sorted and have no duplicates
///
/// It could be replacce with the corresponding standard library method when it will be stabilized
/// <https://github.com/rust-lang/rust/issues/53485>
pub fn is_sorted<T>(a: &[T]) -> bool
where
    T: PartialOrd,
{
    a.iter().tuple_windows().all(|(a, b)| a < b)
}

/// An error to be returned from grid constructors
#[derive(Error, Debug)]
pub enum ArrayGridError {
    #[error("given grid is empty")]
    ArrayIsEmpty,
    #[error("given grid is not ascending")]
    ArrayIsNotAscending,
}

/// Grid which cell borders are defined by an ascending array
///
/// Lookup time is O(lb n)
#[derive(Clone, Debug)]
pub struct ArrayGrid<T> {
    borders: Array1<T>,
}

impl<T> ArrayGrid<T>
where
    Array1<T>: Clone + Debug,
    T: Float,
{
    /// Wraps given array into [ArrayGrid] or return an error
    ///
    /// Note that array describes cell borders, not center or whatever else
    pub fn new(borders: Array1<T>) -> Result<Self, ArrayGridError> {
        if borders.is_empty() {
            return Err(ArrayGridError::ArrayIsEmpty);
        }
        if !is_sorted(borders.as_slice().unwrap()) {
            return Err(ArrayGridError::ArrayIsNotAscending);
        }
        Ok(Self { borders })
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

/// Linear grid defined by its start, end and number of cells
///
/// Lookup time is O(1)
#[derive(Clone, Debug)]
pub struct LinearGrid<T> {
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
    /// Create [LinearGrid] from borders and number of cells
    ///
    /// `start` is the left border of the leftmost cell, `end` is the right border of the rightmost
    /// cell, `n` is the number of cells. This means that the number of borders is `n + 1`, `start`
    /// border has zero index and `end` border has index `n`.
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

    /// Cell size
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

/// Logarithmic grid defined by its start, end and number of cells
///
/// Lookup time is O(1)
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
    /// Create [LinearGrid] from borders and number of cells
    ///
    /// `start` is the left border of the leftmost cell, `end` is the right border of the rightmost
    /// cell, `n` is the number of cells. This means that the number of borders is `n + 1`, `start`
    /// border has zero index and `end` border has index `n`.
    pub fn from_start_end(start: T, end: T, n: usize) -> Self {
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

    /// Create [LinearGrid] from decimal logarithms of borders and number of cells
    ///
    /// `lg_start` is the decimal logarithm of the left border of the leftmost cell, `lg_end` is the
    /// decimal logarithm of the right border of the rightmost cell, `n` is the number of cells.
    /// This means that the number of borders is `n + 1`, `lg_start` border has zero index and
    /// `lg_end` border has index `n`.
    pub fn from_lg_start_end(lg_start: T, lg_end: T, n: usize) -> Self {
        Self::from_start_end(T::powf(T::ten(), lg_start), T::powf(T::ten(), lg_end), n)
    }

    /// Logarithmic size of cell
    #[inline]
    pub fn get_cell_lg_size(&self) -> T {
        self.cell_lg_size
    }

    /// Logarithm of the leftmost border
    #[inline]
    pub fn get_lg_start(&self) -> T {
        self.lg_start
    }

    /// Logarithm of the rightmost border
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

/// Value to return from [Grid::idx]
pub enum CellIndex {
    /// Bellow the leftmost border
    LowerMin,
    /// Equal or greater the rightmost border
    GreaterMax,
    /// Cell index
    Value(usize),
}

/// dm–dt map plotter
#[derive(Clone, Debug)]
pub struct DmDt<T>
where
    T: Copy,
{
    pub dt_grid: Box<dyn Grid<T>>,
    pub dm_grid: Box<dyn Grid<T>>,
}

impl<T> DmDt<T>
where
    T: Float,
{
    /// Create new [DmDt]
    pub fn from_grids<Gdt, Gdm>(dt_grid: Gdt, dm_grid: Gdm) -> Self
    where
        Gdt: Grid<T> + 'static,
        Gdm: Grid<T> + 'static,
    {
        Self {
            dt_grid: Box::new(dt_grid),
            dm_grid: Box::new(dm_grid),
        }
    }

    /// Create new [DmDt] with logarithmic dt grid and linear dm grid
    ///
    /// dt grid will have borders `[10^min_lgdt, 10^max_lgdt)`, dm grid will have borders
    /// `[-max_abs_dm, max_abs_dm)`
    pub fn from_lgdt_dm_limits(
        min_lgdt: T,
        max_lgdt: T,
        lgdt_size: usize,
        max_abs_dm: T,
        dm_size: usize,
    ) -> Self {
        Self::from_grids(
            LgGrid::from_lg_start_end(min_lgdt, max_lgdt, lgdt_size),
            LinearGrid::new(-max_abs_dm, max_abs_dm, dm_size),
        )
    }

    /// N dt by N dm
    pub fn shape(&self) -> (usize, usize) {
        (self.dt_grid.cell_count(), self.dm_grid.cell_count())
    }

    /// Represents each pair of (t, m) points as a unity value in dm-dt map
    ///
    /// `t` must be an ascending slice
    pub fn points(&self, t: &[T], m: &[T]) -> Array2<u64> {
        let mut a = Array2::zeros(self.shape());
        for (i1, (&x1, &y1)) in t.iter().zip(m.iter()).enumerate() {
            for (&x2, &y2) in t[i1 + 1..].iter().zip(m[i1 + 1..].iter()) {
                let dt = x2 - x1;
                let idx_dt = match self.dt_grid.idx(dt) {
                    CellIndex::LowerMin => continue,
                    CellIndex::GreaterMax => break,
                    CellIndex::Value(idx_dt) => idx_dt,
                };
                let dm = y2 - y1;
                let idx_dm = match self.dm_grid.idx(dm) {
                    CellIndex::Value(idx_dm) => idx_dm,
                    CellIndex::LowerMin | CellIndex::GreaterMax => continue,
                };
                a[(idx_dt, idx_dm)] += 1;
            }
        }
        a
    }

    fn update_gausses_helper<Erf>(
        &self,
        a: &mut Array2<T>,
        idx_dt: usize,
        y1: T,
        y2: T,
        d1: T,
        d2: T,
    ) where
        T: ErfFloat,
        Erf: ErrorFunction<T>,
    {
        let dm = y2 - y1;
        let dm_err = T::sqrt(d1 + d2);

        let min_idx_dm = match self
            .dm_grid
            .idx(dm + Erf::min_dx_nonzero_normal_cdf(dm_err))
        {
            CellIndex::LowerMin => 0,
            CellIndex::GreaterMax => return,
            CellIndex::Value(min_idx_dm) => min_idx_dm,
        };
        let max_idx_dm = match self
            .dm_grid
            .idx(dm + Erf::max_dx_nonunity_normal_cdf(dm_err))
        {
            CellIndex::LowerMin => return,
            CellIndex::GreaterMax => self.dm_grid.cell_count(),
            CellIndex::Value(i) => usize::min(i + 1, self.dm_grid.cell_count()),
        };

        a.slice_mut(s![idx_dt, min_idx_dm..max_idx_dm])
            .iter_mut()
            .zip(
                self.dm_grid
                    .get_borders()
                    .slice(s![min_idx_dm..max_idx_dm + 1])
                    .iter()
                    .map(|&dm_border| Erf::normal_cdf(dm_border, dm, dm_err))
                    .tuple_windows()
                    .map(|(a, b)| b - a),
            )
            .for_each(|(cell, value)| *cell += value);
    }

    /// Represents each pair of (t, m, err2) points as a Gaussian distribution in dm-dt map
    ///
    /// `t` must be an ascending slice.
    ///
    /// Each observation is assumed to happen at time moment `t_i` and have Gaussian distribution of
    /// its magnitude `N(m_i, err2_i)`. Each pair of observations
    /// `(t_1, m_1, err2_1), (t_2, m_2, err2_2)` is represented by 1-D Gaussian in the dm-dt space
    /// having constant `dt` and `dm ~ N(m2-m1, err2_1 + err2_2)`. This distribution is integrated
    /// over each cell using `Erf` struct implementing [ErrorFunction].
    pub fn gausses<Erf>(&self, t: &[T], m: &[T], err2: &[T]) -> Array2<T>
    where
        T: ErfFloat,
        Erf: ErrorFunction<T>,
    {
        let mut a = Array2::zeros(self.shape());
        for (i1, ((&x1, &y1), &d1)) in t.iter().zip(m.iter()).zip(err2.iter()).enumerate() {
            for ((&x2, &y2), &d2) in t[i1 + 1..]
                .iter()
                .zip(m[i1 + 1..].iter())
                .zip(err2[i1 + 1..].iter())
            {
                let dt = x2 - x1;
                let idx_dt = match self.dt_grid.idx(dt) {
                    CellIndex::LowerMin => continue,
                    CellIndex::GreaterMax => break,
                    CellIndex::Value(idx_dt) => idx_dt,
                };
                self.update_gausses_helper::<Erf>(&mut a, idx_dt, y1, y2, d1, d2);
            }
        }
        a
    }

    /// Count dt in the each dt grid cell
    pub fn dt_points(&self, t: &[T]) -> Array1<u64> {
        let mut a = Array1::zeros(self.dt_grid.cell_count());
        for (i1, &x1) in t.iter().enumerate() {
            for &x2 in t[i1 + 1..].iter() {
                let dt = x2 - x1;
                let idx_dt = match self.dt_grid.idx(dt) {
                    CellIndex::LowerMin => continue,
                    CellIndex::GreaterMax => break,
                    CellIndex::Value(idx_dt) => idx_dt,
                };
                a[idx_dt] += 1;
            }
        }
        a
    }

    /// Conditional probability `p(m2-m1|t2-t1)`
    ///
    /// Technically it is optimized version of [DmDt::gausses()] normalized by [DmDt::dt_points] but
    /// with better performance. Mathematically it represents the distribution of conditional
    /// probability `p(m2-m1|t2-t1)`, see
    /// [Soraisam et al. 2020](https://doi.org/10.3847/1538-4357/ab7b61) for details.
    pub fn cond_prob<Erf>(&self, t: &[T], m: &[T], err2: &[T]) -> Array2<T>
    where
        T: ErfFloat,
        Erf: ErrorFunction<T>,
    {
        let mut a: Array2<T> = Array2::zeros(self.shape());
        let mut dt_points: Array1<u64> = Array1::zeros(self.dt_grid.cell_count());
        for (i1, ((&x1, &y1), &d1)) in t.iter().zip(m.iter()).zip(err2.iter()).enumerate() {
            for ((&x2, &y2), &d2) in t[i1 + 1..]
                .iter()
                .zip(m[i1 + 1..].iter())
                .zip(err2[i1 + 1..].iter())
            {
                let dt = x2 - x1;
                let idx_dt = match self.dt_grid.idx(dt) {
                    CellIndex::LowerMin => continue,
                    CellIndex::GreaterMax => break,
                    CellIndex::Value(idx_dt) => idx_dt,
                };

                dt_points[idx_dt] += 1;

                self.update_gausses_helper::<Erf>(&mut a, idx_dt, y1, y2, d1, d2);
            }
        }
        ndarray::Zip::from(a.rows_mut())
            .and(&dt_points)
            .for_each(|mut row, &count| {
                if count == 0 {
                    return;
                }
                row /= T::approx_from(count).unwrap();
            });
        a
    }
}

/// Convert [u8] dm–dt map into PNG image
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
    writer.write_image_data(transposed.as_slice().unwrap())?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::erf::{Eps1Over1e3Erf, ExactErf};
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;

    assert_impl_all!(DmDt<f32>: Clone, Debug, Send, Sync);
    assert_impl_all!(DmDt<f64>: Clone, Debug, Send, Sync);

    #[test]
    fn dt_points_vs_points() {
        let dmdt = DmDt::from_lgdt_dm_limits(0.0_f32, 2.0_f32, 32, 3.0_f32, 32);
        let t = Array1::linspace(0.0, 100.0, 101);
        // dm is within map borders
        let m = t.mapv(f32::sin);

        let points = dmdt.points(t.as_slice().unwrap(), m.as_slice().unwrap());
        let dt_points = dmdt.dt_points(t.as_slice().unwrap());

        assert_eq!(points.sum_axis(Axis(1)), dt_points,);
    }

    #[test]
    fn dt_points_vs_gausses() {
        let dmdt = DmDt::from_lgdt_dm_limits(0.0_f32, 2.0_f32, 32, 3.0_f32, 32);
        let t = Array1::linspace(0.0, 100.0, 101);
        // dm is within map borders
        let m = t.mapv(f32::sin);
        // err is ~0.03
        let err2 = Array1::from_elem(101, 0.001_f32);

        let gausses = dmdt.gausses::<ExactErf>(
            t.as_slice().unwrap(),
            m.as_slice().unwrap(),
            err2.as_slice().unwrap(),
        );
        let sum_gausses = gausses.sum_axis(Axis(1));
        let dt_points = dmdt.dt_points(t.as_slice().unwrap()).mapv(|x| x as f32);

        assert_abs_diff_eq!(
            sum_gausses.as_slice().unwrap(),
            dt_points.as_slice().unwrap(),
            epsilon = 1e-4,
        );
    }

    #[test]
    fn cond_prob() {
        let dmdt = DmDt::from_lgdt_dm_limits(0.0_f32, 2.0_f32, 32, 1.25_f32, 32);

        let t = Array1::linspace(0.0, 100.0, 101);
        let m = t.mapv(f32::sin);
        // err is ~0.03
        let err2 = Array1::from_elem(101, 0.001);

        let from_gausses_dt_points = {
            let mut map = dmdt.gausses::<Eps1Over1e3Erf>(
                t.as_slice().unwrap(),
                m.as_slice().unwrap(),
                err2.as_slice_memory_order().unwrap(),
            );
            let dt_points = dmdt.dt_points(t.as_slice().unwrap());
            let dt_non_zero_points = dt_points.mapv(|x| if x == 0 { 1.0 } else { x as f32 });
            map /= &dt_non_zero_points.into_shape((map.nrows(), 1)).unwrap();
            map
        };

        let from_cond_prob = dmdt.cond_prob::<Eps1Over1e3Erf>(
            t.as_slice().unwrap(),
            m.as_slice().unwrap(),
            err2.as_slice().unwrap(),
        );

        assert_abs_diff_eq!(
            from_gausses_dt_points.as_slice().unwrap(),
            from_cond_prob.as_slice().unwrap(),
            epsilon = std::f32::EPSILON,
        );
    }
}
