use conv::*;
use ndarray::{Array2, NdFloat};
use png;
use std::io::Write;

trait Float: NdFloat + ValueFrom<usize> + ApproxInto<usize, RoundToZero> {}

impl Float for f32 {}
impl Float for f64 {}

pub struct Grid<T> {
    start: T, // coordinate of the left border of the leftmost cell
    end: T,   // coordinate of the right border of the rightmost cell
    n: usize,
    length: T, // distance from the left border of the leftmost cell to the right border of the rightmost cell
    cell_size: T,
}

impl<T> Grid<T>
where
    T: Float,
{
    pub fn new(start: T, end: T, n: usize) -> Self {
        let length = end - start;
        let cell_size = (end - start) / n.value_as::<T>().unwrap();
        Self {
            start,
            end,
            n,
            length,
            cell_size,
        }
    }

    fn idx(&self, x: T) -> Option<usize> {
        if (self.start..self.end).contains(&x) {
            Some(((x - self.start) / self.cell_size).approx_into().unwrap())
        } else {
            None
        }
    }
}

pub struct DmDt<T> {
    pub lgdt_grid: Grid<T>,
    pub dm_grid: Grid<T>,
}

impl<T> DmDt<T>
where
    T: Float,
{
    pub fn convert_lc(&self, t: &[T], m: &[T]) -> Array2<usize> {
        let mut a = Array2::zeros((self.lgdt_grid.n, self.dm_grid.n));
        for (i1, (&x1, &y1)) in t.iter().zip(m.iter()).enumerate() {
            for (&x2, &y2) in t[i1 + 1..].iter().zip(m[i1 + 1..].iter()) {
                let lgdt = T::log10(x2 - x1);
                let dm = y2 - y1;
                if let Some(idx_lgdt) = self.lgdt_grid.idx(lgdt) {
                    if let Some(idx_dm) = self.dm_grid.idx(dm) {
                        a[(idx_lgdt, idx_dm)] += 1;
                    }
                }
            }
        }
        a
    }
}

pub fn normalise(a: &Array2<usize>) -> Array2<u8> {
    let max = *a.iter().max().unwrap();
    if max == 0 {
        a.mapv(|x| x as u8)
    } else {
        let normalised = a * (u8::MAX as usize) / max;
        normalised.mapv(|x| x as u8)
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
