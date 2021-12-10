use clap::Parser;
use light_curve_feature::{
    features::VillarLnPrior, prelude::*, BazinFit, Feature, FeatureEvaluator, LmsderCurveFit,
    LnPrior, McmcCurveFit, TimeSeries, VillarFit,
};
use light_curve_feature_test_util::iter_sn1a_flux_ts;
use ndarray::{Array1, ArrayView1};
use plotters::prelude::*;
use plotters_bitmap::BitMapBackend;
use rayon::prelude::*;

fn main() {
    let Opts { n, dir } = Opts::parse();
    let n = match n {
        Some(n) => n,
        None => usize::MAX,
    };

    std::fs::create_dir_all(&dir).expect("Cannot create output directory");

    let features: Vec<(&str, Feature<_>)> = vec![
        (
            "BazinFit LMSDER",
            BazinFit::new(
                LmsderCurveFit::default().into(),
                LnPrior::none(),
                BazinFit::default_inits_bounds(),
            )
            .into(),
        ),
        (
            "BazinFit MCMC+LMSDER",
            BazinFit::new(
                McmcCurveFit::new(1024, Some(LmsderCurveFit::default().into())).into(),
                LnPrior::none(),
                BazinFit::default_inits_bounds(),
            )
            .into(),
        ),
        (
            "VillarFit LMSDER",
            VillarFit::new(
                LmsderCurveFit::default().into(),
                LnPrior::none(),
                VillarFit::default_inits_bounds(),
            )
            .into(),
        ),
        (
            "VillarFit MCMC+LMSDER",
            VillarFit::new(
                McmcCurveFit::new(1024, Some(LmsderCurveFit::default().into())).into(),
                LnPrior::none(),
                VillarFit::default_inits_bounds(),
            )
            .into(),
        ),
        (
            "VillarFit MCMC+prior",
            VillarFit::new(
                McmcCurveFit::new(1024, None).into(),
                // We set minimum amplitude to be corresponded to 21 mag
                VillarLnPrior::hosseinzadeh2020(1.0, f64::powf(10.0, -0.4 * 21.0)),
                VillarInitsBounds::option_arrays(
                    // init
                    [
                        None,        // amplitude
                        None,        // offset
                        None,        // t0
                        Some(30.0),  // tau_rise
                        Some(100.0), // tau_fall
                        Some(0.0),   // nu
                        Some(20.0),  // gamma
                    ],
                    // lower
                    [
                        None,
                        None,
                        None,
                        Some(0.01),
                        Some(1.0),
                        Some(0.0),
                        Some(0.0),
                    ],
                    // upper
                    [
                        None,
                        None,
                        None,
                        Some(50.0),
                        Some(300.0),
                        Some(1.0),
                        Some(180.0),
                    ],
                ),
            )
            .into(),
        ),
    ];
    iter_sn1a_flux_ts()
        .take(n)
        .filter(|(ztf_id, _)| ztf_id.starts_with("ZTF18aaxsioa"))
        .par_bridge()
        .for_each(|(ztf_id, mut ts)| {
            let filename = format!("{}.png", ztf_id);
            let path = {
                let mut path = std::path::PathBuf::from(&dir);
                path.push(filename);
                path
            };
            fit_and_plot(&mut ts, &features, ztf_id, path);
        });
}

#[derive(Parser)]
struct Opts {
    #[clap(short)]
    n: Option<usize>,
    #[clap(short, long, default_value = "figures")]
    dir: String,
}

fn fitted_model(
    t: ArrayView1<f64>,
    ts: &mut TimeSeries<f64>,
    feature: &Feature<f64>,
) -> (Array1<f64>, f64) {
    let values = feature.eval(ts).expect("Feature cannot be extracted");
    println!("{:?}", values);
    let reduced_chi2 = values[values.len() - 1];
    let model: Box<dyn Fn(f64, &[f64]) -> f64> = match feature {
        Feature::BazinFit(..) => Box::new(BazinFit::f),
        Feature::VillarFit(..) => Box::new(VillarFit::f),
        _ => panic!("Unknown *Fit variant"),
    };
    let flux = t.mapv(|t| model(t, &values));
    (flux, reduced_chi2)
}

fn fit_and_plot<P>(
    ts: &mut TimeSeries<f64>,
    features: &[(&str, Feature<f64>)],
    ztf_id: &str,
    path: P,
) where
    P: AsRef<std::path::Path>,
{
    let root = BitMapBackend::new(path.as_ref(), (640, 480)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(ztf_id, ("sans-serif", 20))
        .margin(2)
        .set_label_area_size(LabelAreaPosition::Left, 20)
        .set_label_area_size(LabelAreaPosition::Bottom, 20)
        .build_cartesian_2d(
            ts.t.get_min()..ts.t.get_max(),
            ts.m.get_min()..1.1 * ts.m.get_max(),
        )
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(
            ts.t.as_slice()
                .iter()
                .zip(ts.m.as_slice().iter())
                .zip(ts.w.as_slice().iter())
                .map(|((&t, &m), &w)| {
                    let sigma = w.recip().sqrt();
                    ErrorBar::new_vertical(t, m - sigma, m, m + sigma, BLACK.filled(), 10)
                }),
        )
        .unwrap()
        .label("Light curve")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLACK));

    let t = Array1::linspace(ts.t.get_min(), ts.t.get_max(), 101);
    for (i, (name, feature)) in features.iter().enumerate() {
        let (model, reduced_chi2) = fitted_model(t.view(), ts, feature);
        chart
            .draw_series(LineSeries::new(
                t.as_slice()
                    .unwrap()
                    .iter()
                    .copied()
                    .zip(model.as_slice().unwrap().iter().copied()),
                &Palette99::pick(i),
            ))
            .unwrap()
            .label(&format!("{}, reduced chi2 = {:.2}", name, reduced_chi2))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &Palette99::pick(i)));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.filled())
        .draw()
        .unwrap();

    root.present().unwrap();
}
