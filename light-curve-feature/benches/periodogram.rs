use criterion::{black_box, Criterion};
use light_curve_common::linspace;
use light_curve_feature::periodogram::*;
use light_curve_feature::time_series::TimeSeries;

type BoxedPeriodogram = Box<dyn PeriodogramPower<f32>>;

pub fn bench_periodogram(c: &mut Criterion) {
    const N: [usize; 3] = [10, 100, 1000];
    let power_resolution: [(fn() -> BoxedPeriodogram, f32); 2] = [
        (|| Box::new(PeriodogramPowerDirect), 5.0),
        (|| Box::new(PeriodogramPowerFft), 10.0),
    ];
    const PERIOD: f32 = 0.22;
    let nyquist: Box<dyn NyquistFreq<f32>> = Box::new(AverageNyquistFreq);

    Periodogram::<f32>::init_thread_local_fft_plans(&N);

    for &n in N.iter() {
        let x = linspace(0.0_f32, 1.0, n);
        let y: Vec<_> = x
            .iter()
            .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / PERIOD * x + 0.5) + 4.0)
            .collect();
        for (periodogram_power, resolution) in power_resolution.iter() {
            c.bench_function(
                format!("Periodogram: {} length, {:?}", n, periodogram_power()).as_str(),
                |b| {
                    b.iter(|| {
                        let mut ts = TimeSeries::new(&x[..], &y[..], None);
                        let periodogram = Periodogram::from_t(
                            periodogram_power(),
                            &x,
                            *resolution,
                            1.0,
                            &nyquist,
                        );
                        periodogram.power(black_box(&mut ts));
                    })
                },
            );
        }
    }
}
