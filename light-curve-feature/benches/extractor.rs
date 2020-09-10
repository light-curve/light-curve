use criterion::{black_box, Criterion};
use light_curve_feature::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::any::type_name;

pub fn bench_extractor<T>(c: &mut Criterion)
where
    T: Float + 'static,
    StandardNormal: Distribution<T>,
{
    const N: [usize; 3] = [10, 100, 1000];

    let fe_without_periodogram = feat_extr!(
        Amplitude::default(),
        AndersonDarlingNormal::default(),
        BeyondNStd::default(),
        Cusum::default(),
        Eta::default(),
        EtaE::default(),
        InterPercentileRange::default(),
        Kurtosis::default(),
        LinearTrend::default(),
        LinearFit::default(),
        MaximumSlope::default(),
        ReducedChi2::default(),
        Skew::default(),
        StandardDeviation::default(),
        StetsonK::default(),
        WeightedMean::default(),
    );
    let mut periodogram = Periodogram::new(5);
    periodogram.set_max_freq_factor(2.0);
    let fe_periodogram = feat_extr!(
        Amplitude::default(),
        AndersonDarlingNormal::default(),
        BeyondNStd::default(),
        Cusum::default(),
        Eta::default(),
        EtaE::default(),
        InterPercentileRange::default(),
        Kurtosis::default(),
        LinearTrend::default(),
        LinearFit::default(),
        MaximumSlope::default(),
        periodogram,
        ReducedChi2::default(),
        Skew::default(),
        StandardDeviation::default(),
        StetsonK::default(),
        WeightedMean::default(),
    );

    for &n in N.iter() {
        let x = randspace(n);
        let y = randvec(n);
        let err = randvec(n);
        for (&fe, &name) in [&fe_without_periodogram, &fe_periodogram]
            .iter()
            .zip(["w/o Periodogram", "w/ Periodogram"].iter())
        {
            c.bench_function(
                format!("FeatureExtractor {}: [{}; {}]", name, n, type_name::<T>()).as_str(),
                |b| {
                    b.iter(|| {
                        run(black_box(fe), black_box(&x), black_box(&y), black_box(&err)).unwrap();
                    });
                },
            );
        }
    }
}

fn run<T: Float>(
    fe: &FeatureExtractor<T>,
    x: &[T],
    y: &[T],
    err: &[T],
) -> Result<Vec<T>, EvaluatorError> {
    let w: Vec<_> = err.iter().map(|&e| e.powi(-2)).collect();
    let mut ts = TimeSeries::new(&x, &y, Some(&w));
    fe.eval(&mut ts)
}

fn randvec<T>(n: usize) -> Vec<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    (0..n)
        .map(|_| {
            let x: T = thread_rng().sample(StandardNormal);
            x
        })
        .collect()
}

fn randspace<T>(n: usize) -> Vec<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    let mut x = randvec::<T>(n);
    x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    x
}
