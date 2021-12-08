use conv::ConvUtil;
use criterion::{black_box, Criterion};
use light_curve_feature::*;
use light_curve_feature_test_util::iter_sn1a_flux_ts;
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::any::type_name;

pub fn bench_extractor<T>(c: &mut Criterion)
where
    T: Float + 'static,
    StandardNormal: Distribution<T>,
{
    const N: [usize; 2] = [100, 1000];

    let features: Vec<Feature<_>> = vec![
        Amplitude::default().into(),
        AndersonDarlingNormal::default().into(),
        BeyondNStd::default().into(),
        Cusum::default().into(),
        Eta::default().into(),
        EtaE::default().into(),
        ExcessVariance::default().into(),
        InterPercentileRange::default().into(),
        Kurtosis::default().into(),
        LinearFit::default().into(),
        LinearTrend::default().into(),
        MagnitudePercentageRatio::default().into(),
        MaximumSlope::default().into(),
        Mean::default().into(),
        MeanVariance::default().into(),
        Median::default().into(),
        MedianAbsoluteDeviation::default().into(),
        MedianBufferRangePercentage::default().into(),
        PercentAmplitude::default().into(),
        PercentDifferenceMagnitudePercentile::default().into(),
        ReducedChi2::default().into(),
        Skew::default().into(),
        StandardDeviation::default().into(),
        StetsonK::default().into(),
        WeightedMean::default().into(),
    ];

    let observation_count_vec: Vec<_> = (0..20)
        .map(|_| antifeatures::ObservationCount::default().into())
        .collect();

    let beyond_n_std_vec: Vec<_> = (1usize..21)
        .map(|i| BeyondNStd::new(i.value_as::<T>().unwrap() / T::ten()).into())
        .collect();

    let mut bins = Bins::default();
    bins.add_feature(StetsonK::default().into());

    let mut periodogram = Periodogram::default();
    periodogram.set_max_freq_factor(10.0);

    let names_fes: Vec<_> = features
        .iter()
        .map(|f| (f.get_names()[0], FeatureExtractor::new(vec![f.clone()])))
        .chain(std::iter::once((
            "all non-meta features",
            FeatureExtractor::new(features.clone()),
        )))
        .chain(std::iter::once((
            "multiple ObservationCount",
            FeatureExtractor::new(observation_count_vec.clone()),
        )))
        .chain(std::iter::once((
            "multiple BeyondNStd",
            FeatureExtractor::new(beyond_n_std_vec),
        )))
        .chain(std::iter::once((
            "BazinFit",
            FeatureExtractor::new(vec![BazinFit::default().into()]),
        )))
        .chain(std::iter::once((
            "Bins",
            FeatureExtractor::new(vec![bins.into()]),
        )))
        .chain(std::iter::once((
            "Periodogram",
            FeatureExtractor::new(vec![periodogram.into()]),
        )))
        .chain(std::iter::once((
            "VillarFit",
            FeatureExtractor::new(vec![VillarFit::default().into()]),
        )))
        .collect();

    for &n in N.iter() {
        let mut ts = randts(n);
        for (name, fe) in names_fes.iter() {
            c.bench_function(
                format!("FeatureExtractor {}: [{}; {}]", name, n, type_name::<T>()).as_str(),
                |b| {
                    b.iter(|| {
                        let _v = fe.eval(black_box(&mut ts)).unwrap();
                    });
                },
            );
        }
    }

    {
        let n = 10;
        let mut ts = randts(n);
        let fe = FeatureExtractor::new(observation_count_vec);
        c.bench_function(
            format!("Multiple ObservationCount {}", type_name::<T>()).as_str(),
            |b| {
                b.iter(|| {
                    let _v = fe.eval(black_box(&mut ts)).unwrap();
                });
            },
        );
    }

    {
        let mut real_data: Vec<_> = iter_sn1a_flux_ts::<T>().map(|(_ztf_id, ts)| ts).collect();
        let curve_fits: Vec<CurveFitAlgorithm> = vec![
            LmsderCurveFit::new(5).into(),
            LmsderCurveFit::new(10).into(),
            LmsderCurveFit::new(15).into(),
            McmcCurveFit::new(128, None).into(),
            McmcCurveFit::new(1024, None).into(),
            McmcCurveFit::new(128, Some(LmsderCurveFit::new(5).into())).into(),
            McmcCurveFit::new(1024, Some(LmsderCurveFit::new(10).into())).into(),
        ];
        for curve_fit in curve_fits.into_iter() {
            let features: Vec<Feature<_>> = vec![
                BazinFit::new(curve_fit.clone(), LnPrior::none()).into(),
                VillarFit::new(curve_fit, LnPrior::none()).into(),
            ];
            for f in features {
                c.bench_function(
                    format!("SN Ia {:?} {}", f, type_name::<T>()).as_str(),
                    |b| {
                        b.iter(|| {
                            real_data.iter_mut().for_each(|ts| {
                                let _v = f.eval(black_box(ts)).unwrap();
                            });
                        });
                    },
                );
            }
        }
    }
}

fn randvec<T>(n: usize) -> Array1<T>
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

fn randspace<T>(n: usize) -> Array1<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    let mut x = randvec::<T>(n);
    x.as_slice_mut()
        .unwrap()
        .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    x
}

pub fn randts<T>(n: usize) -> TimeSeries<'static, T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    let t = randspace(n);
    let m = randvec(n);
    let w = randvec(n).mapv(|x: T| x.powi(2));
    TimeSeries::new(t, m, w)
}
