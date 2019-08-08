#[macro_use]
extern crate criterion;

mod statistics;
use statistics::bench_peak_indices;

criterion_group!(benches_statistics, bench_peak_indices);
criterion_main!(benches_statistics);
