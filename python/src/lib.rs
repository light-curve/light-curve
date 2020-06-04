use light_curve_feature::{PeriodogramPowerDirect, PeriodogramPowerFft, TimeSeries};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::ValueError;
use pyo3::prelude::{pyclass, pymethods, pymodule, Py, PyModule, PyObject, PyResult, Python};

type F = f64;

fn ts_from_arrays<'a, 'b, 'c>(
    t: &'a PyArray1<F>,
    m: &'b PyArray1<F>,
    err2: Option<&'c PyArray1<F>>,
) -> PyResult<TimeSeries<'a, 'b, 'c, F>> {
    Ok(TimeSeries::new(
        t.as_slice()?,
        m.as_slice()?,
        match err2 {
            Some(array) => Some(array.as_slice()?),
            None => None,
        },
    ))
}

#[pyclass]
struct PyFeatureEvaluator {
    feature_evaluator: Box<dyn light_curve_feature::FeatureEvaluator<F>>,
}

#[pymethods]
impl PyFeatureEvaluator {
    #[call]
    fn __call__(
        &self,
        py: Python,
        t: &PyArray1<F>,
        m: &PyArray1<F>,
        err2: Option<&PyArray1<F>>,
    ) -> PyResult<Py<PyArray1<F>>> {
        let mut ts = ts_from_arrays(t, m, err2)?;
        Ok(self
            .feature_evaluator
            .eval(&mut ts)
            .into_pyarray(py)
            .to_owned())
    }
}

macro_rules! evaluator {
    ($name: ident, $eval: ty $(,)?) => {
        #[pyclass(extends = PyFeatureEvaluator)]
        struct $name {}

        #[pymethods]
        impl $name {
            #[new]
            fn __new__() -> (Self, PyFeatureEvaluator) {
                (
                    Self {},
                    PyFeatureEvaluator {
                        feature_evaluator: Box::new(<$eval>::new()),
                    },
                )
            }
        }
    };
}

evaluator!(Amplitude, light_curve_feature::Amplitude);

#[pyclass(extends = PyFeatureEvaluator)]
struct BeyondNStd {}

#[pymethods]
impl BeyondNStd {
    #[new]
    #[args(nstd = "None")]
    fn __new__(nstd: Option<F>) -> (Self, PyFeatureEvaluator) {
        let eval = match nstd {
            Some(nstd) => light_curve_feature::BeyondNStd::new(nstd),
            None => light_curve_feature::BeyondNStd::default(),
        };
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(eval),
            },
        )
    }
}

evaluator!(Cusum, light_curve_feature::Cusum);

evaluator!(Eta, light_curve_feature::Eta);

evaluator!(EtaE, light_curve_feature::EtaE);

evaluator!(Kurtosis, light_curve_feature::Kurtosis);

evaluator!(LinearFit, light_curve_feature::LinearFit);

evaluator!(LinearTrend, light_curve_feature::LinearTrend);

#[pyclass(extends = PyFeatureEvaluator)]
struct MagnitudePercentageRatio {}

#[pymethods]
impl MagnitudePercentageRatio {
    #[new]
    #[args(quantile_numerator = "None", quantile_denominator = "None")]
    fn __new__(
        quantile_numerator: Option<f32>,
        quantile_denominator: Option<f32>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        let eval = match (quantile_numerator, quantile_denominator) {
            (Some(n), Some(d)) => light_curve_feature::MagnitudePercentageRatio::new(n, d),
            (None, None) => light_curve_feature::MagnitudePercentageRatio::default(),
            _ => {
                return Err(ValueError::py_err(
                    "Both quantile_numerator and quantile_denominator must be floats or Nones",
                ))
            }
        };
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(eval),
            },
        ))
    }
}

evaluator!(MaximumSlope, light_curve_feature::MaximumSlope);

evaluator!(
    MedianAbsoluteDeviation,
    light_curve_feature::MedianAbsoluteDeviation,
);

#[pyclass(extends = PyFeatureEvaluator)]
struct MedianBufferRangePercentage {}

#[pymethods]
impl MedianBufferRangePercentage {
    #[new]
    #[args(quantile = "None")]
    fn __new__(quantile: Option<F>) -> (Self, PyFeatureEvaluator) {
        let eval = match quantile {
            Some(quantile) => light_curve_feature::MedianBufferRangePercentage::new(quantile),
            None => light_curve_feature::MedianBufferRangePercentage::default(),
        };
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(eval),
            },
        )
    }
}

evaluator!(PercentAmplitude, light_curve_feature::PercentAmplitude);

#[pyclass(extends = PyFeatureEvaluator)]
struct PercentDifferenceMagnitudePercentile {}

#[pymethods]
impl PercentDifferenceMagnitudePercentile {
    #[new]
    #[args(quantile = "None")]
    fn __new__(quantile: Option<f32>) -> (Self, PyFeatureEvaluator) {
        let eval = match quantile {
            Some(quantile) => {
                light_curve_feature::PercentDifferenceMagnitudePercentile::new(quantile)
            }
            None => light_curve_feature::PercentDifferenceMagnitudePercentile::default(),
        };
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(eval),
            },
        )
    }
}

#[pyclass(extends = PyFeatureEvaluator)]
struct Periodogram {}

#[pymethods]
impl Periodogram {
    #[new]
    #[args(
        peaks = "None",
        resolution = "None",
        max_freq_factor = "None",
        nyquist = "None",
        fast = "None"
    )]
    fn __new__(
        py: Python,
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<PyObject>,
        fast: Option<bool>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        let mut eval = match peaks {
            Some(peaks) => light_curve_feature::Periodogram::new(peaks),
            None => light_curve_feature::Periodogram::default(),
        };
        if let Some(resolution) = resolution {
            eval.set_freq_resolution(resolution);
        }
        if let Some(max_freq_factor) = max_freq_factor {
            eval.set_max_freq_factor(max_freq_factor);
        }
        if let Some(nyquist) = nyquist {
            let nyquist_freq: Box<dyn light_curve_feature::NyquistFreq<F>> =
                if let Ok(s) = nyquist.extract::<&str>(py) {
                    match s {
                        "average" => Box::new(light_curve_feature::AverageNyquistFreq {}),
                        "mean" => Box::new(light_curve_feature::MedianNyquistFreq {}),
                        _ => {
                            return Err(ValueError::py_err(
                                "nyquist must be one of: None, 'average', 'mean' or quantile value",
                            ))
                        }
                    }
                } else if let Ok(quantile) = nyquist.extract::<f32>(py) {
                    Box::new(light_curve_feature::QuantileNyquistFreq { quantile })
                } else {
                    return Err(ValueError::py_err(
                        "nyquist must be one of: None, 'average', 'mean' or quantile value",
                    ));
                };
            eval.set_nyquist(nyquist_freq);
        }
        if let Some(fast) = fast {
            if fast {
                eval.set_periodogram_algorithm(move || Box::new(PeriodogramPowerFft));
            } else {
                eval.set_periodogram_algorithm(move || Box::new(PeriodogramPowerDirect));
            }
        }
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(eval),
            },
        ))
    }
}

evaluator!(ReducedChi2, light_curve_feature::ReducedChi2);

evaluator!(Skew, light_curve_feature::Skew);

evaluator!(StandardDeviation, light_curve_feature::StandardDeviation);

evaluator!(StetsonK, light_curve_feature::StetsonK);

#[pymodule]
fn light_curve(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Amplitude>()?;
    m.add_class::<BeyondNStd>()?;
    m.add_class::<Cusum>()?;
    m.add_class::<Eta>()?;
    m.add_class::<EtaE>()?;
    m.add_class::<Kurtosis>()?;
    m.add_class::<LinearFit>()?;
    m.add_class::<LinearTrend>()?;
    m.add_class::<MagnitudePercentageRatio>()?;
    m.add_class::<MaximumSlope>()?;
    m.add_class::<MedianAbsoluteDeviation>()?;
    m.add_class::<MedianBufferRangePercentage>()?;
    m.add_class::<PercentAmplitude>()?;
    m.add_class::<PercentDifferenceMagnitudePercentile>()?;
    m.add_class::<Periodogram>()?;
    m.add_class::<ReducedChi2>()?;
    m.add_class::<Skew>()?;
    m.add_class::<StandardDeviation>()?;
    m.add_class::<StetsonK>()?;

    Ok(())
}