use light_curve_dmdt::{normalise, to_png, DmDt, Grid};
use std::fs::File;
use std::io;
use std::io::{BufRead, BufWriter};
use std::path::Path;
use std::str::FromStr;
use thiserror::Error;

#[derive(Error, std::fmt::Debug)]
#[error("{0}")]
enum MainError {
    IOError(#[from] io::Error),
    ParseFloatError(#[from] std::num::ParseFloatError),
    PngEncodingError(#[from] png::EncodingError),
    NotEnoughColumns(&'static str),
}

fn main() -> Result<(), MainError> {
    let mut t: Vec<f32> = vec![];
    let mut m: Vec<f32> = vec![];
    let mut w: Vec<f32> = vec![];
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        t.push(f32::from_str(
            it.next()
                .ok_or(MainError::NotEnoughColumns("Empty string"))?,
        )?);
        m.push(f32::from_str(it.next().ok_or(
            MainError::NotEnoughColumns("Only one value in line, at least two required"),
        )?)?);
        w.push(f32::powi(
            f32::from_str(it.next().ok_or(MainError::NotEnoughColumns(
                "Only two values in a line, at least three required",
            ))?)?,
            -2,
        ));
    }
    let dmdt = DmDt {
        lgdt_grid: Grid::new(-2.0_f32, 1.0, 128),
        dm_grid: Grid::new(-2.0_f32, 2.0, 128),
    };

    let writer = BufWriter::new(File::create(Path::new("dmdt.png"))?);
    to_png(writer, &normalise(&dmdt.convert_lc_to_points(&t, &m)))?;

    let writer = BufWriter::new(File::create(Path::new("dmdt-gauss.png"))?);
    to_png(writer, &normalise(&dmdt.convert_lc_to_gausses(&t, &m, &w)))?;
    Ok(())
}
