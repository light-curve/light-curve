use conv::*;
use criterion::{black_box, Criterion};
use light_curve_dmdt::{DmDt, ErfFloat, ErrorFunction, Grid};
use ndarray::Array1;

pub fn bench_gausses<T>(c: &mut Criterion)
where
    T: ErfFloat + ValueFrom<f32>,
{
    let dmdt = DmDt {
        lgdt_grid: Grid::new(T::zero(), 2.0_f32.value_as::<T>().unwrap(), 32),
        dm_grid: Grid::new(
            (-1.25_f32).value_as::<T>().unwrap(),
            1.25_f32.value_as::<T>().unwrap(),
            32,
        ),
    };

    let t = Array1::linspace(T::zero(), 100.0_f32.value_as::<T>().unwrap(), 101);
    let m = t.mapv(T::sin);
    // err is ~0.03
    let err2 = Array1::from_elem(101, 0.001_f32.value_as::<T>().unwrap());

    let erfs = [ErrorFunction::Exact, ErrorFunction::Eps1Over1e3];

    for erf in erfs.iter() {
        c.bench_function(
            format!(
                "DmDt<{}>::convert_lc_to_gausses, erf type is {:?}",
                std::any::type_name::<T>(),
                erf
            )
            .as_str(),
            |b| {
                b.iter(|| {
                    black_box(dmdt.convert_lc_to_gausses(
                        t.as_slice_memory_order().unwrap(),
                        m.as_slice_memory_order().unwrap(),
                        err2.as_slice_memory_order().unwrap(),
                        &erf,
                    ));
                })
            },
        );
    }
}
