#[macro_use]
extern crate criterion;

mod erf;
use erf::{bench_erf, bench_erfinv};

criterion_group!(
    benches_erf,
    bench_erf<f32>,
    bench_erf<f64>,
    bench_erfinv<f32>,
    bench_erfinv<f64>
);
criterion_main!(benches_erf);
