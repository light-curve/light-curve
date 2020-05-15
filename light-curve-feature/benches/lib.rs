#[macro_use]
extern crate criterion;

mod fft_crates;
use fft_crates::bench_fft;

mod recurrent_sin_cos;
use recurrent_sin_cos::bench_recurrent_sin_cos;

mod statistics;
use statistics::bench_peak_indices;

mod feature_evaluator {
    use criterion::{black_box, Criterion};
    use light_curve_common::linspace;
    use light_curve_feature::time_series::TimeSeries;
    use light_curve_feature::*;

    pub fn bench_periodogram(c: &mut Criterion) {
        let period = 0.22;
        let x = linspace(0.0_f32, 1.0, 100);
        let y: Vec<_> = x
            .iter()
            .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / period * x + 0.5) + 4.0)
            .collect();
        let ts = TimeSeries::new(&x[..], &y[..], None);
        c.bench_function("periodogram_evenly_sinus_one_peak", |b| {
            b.iter(|| Periodogram::new(1).eval(black_box(&mut ts.clone())))
        });
        c.bench_function("periodogram_evenly_sinus_three_peaks", |b| {
            b.iter(|| Periodogram::new(3).eval(black_box(&mut ts.clone())))
        });
    }
}

criterion_group!(
    benches_feature_evaluator,
    feature_evaluator::bench_periodogram
);
criterion_group!(benches_fft, bench_fft<f32>, bench_fft<f64>);
criterion_group!(benches_recurrent_sin_cos, bench_recurrent_sin_cos);
criterion_group!(benches_statistics, bench_peak_indices);
criterion_main!(
    benches_feature_evaluator,
    benches_fft,
    benches_recurrent_sin_cos,
    benches_statistics
);
