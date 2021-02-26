use light_curve_dmdt::{normalise, to_png, DmDt, Grid};
use png;
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
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.starts_with("#") {
            continue;
        }
        let mut it = line.split_whitespace();
        t.push(f32::from_str(
            it.next()
                .ok_or(MainError::NotEnoughColumns("Empty string"))?,
        )?);
        m.push(f32::from_str(it.next().ok_or(
            MainError::NotEnoughColumns("Only one value on string, at least two required"),
        )?)?);
    }
    let dmdt = DmDt {
        lgdt_grid: Grid::new(-2.0_f32, 3.0, 128),
        dm_grid: Grid::new(-2.0_f32, 2.0, 128),
    };
    let path = Path::new("dmdt.png");
    let file = File::create(path)?;
    let ref mut w = BufWriter::new(file);
    to_png(w, &normalise(&dmdt.convert_lc(&t, &m)))?;
    Ok(())
}
