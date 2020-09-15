use itertools::Itertools;
use light_curve_feature::{
    FeatureEvaluator, PeriodogramPowerDirect, PeriodogramPowerFft, TimeSeries,
};
use ndarray::Array1 as NDArray;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{NotImplementedError, ValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pymodule;
use std::ops::Deref;

type F = f64;
type Arr = PyArray1<F>;
type RoArr<'a> = PyReadonlyArray1<'a, F>;

enum ArrWrapper<'a> {
    Readonly(RoArr<'a>),
    Owned(NDArray<F>),
}

impl<'a> ArrWrapper<'a> {
    fn new(a: &'a Arr, required: bool) -> PyResult<Self> {
        if a.is_contiguous() {
            Ok(Self::Readonly(a.readonly()))
        } else if required {
            if a.strides().iter().any(|&i| i < 0) {
                return Err(ValueError::py_err(
                    "input array has a negative stride, which is currently unsupported due to \
                    https://github.com/PyO3/rust-numpy/issues/151, \
                    please copy the array before passing",
                ));
            }
            Ok(Self::Owned(a.to_owned_array()))
        } else {
            Ok(Self::Owned(unsafe {
                ndarray::Array1::<F>::uninitialized(a.len())
            }))
        }
    }
}

impl<'a> Deref for ArrWrapper<'a> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        match self {
            Self::Readonly(a) => a.as_slice().unwrap(),
            Self::Owned(a) => a.as_slice().unwrap(),
        }
    }
}

fn is_sorted(a: &[F]) -> bool {
    a.iter().tuple_windows().all(|(&a, &b)| a < b)
}

#[pyclass]
struct PyFeatureEvaluator {
    feature_evaluator: Box<dyn light_curve_feature::FeatureEvaluator<F>>,
}

#[pymethods]
impl PyFeatureEvaluator {
    #[call]
    #[args(t, m, sigma = "None", sorted = "None", fill_value = "None")]
    fn __call__(
        &self,
        py: Python,
        t: &Arr,
        m: &Arr,
        sigma: Option<&Arr>,
        sorted: Option<bool>,
        fill_value: Option<F>,
    ) -> PyResult<Py<Arr>> {
        let t = ArrWrapper::new(t, self.feature_evaluator.is_t_required())?;
        match sorted {
            Some(true) => {}
            Some(false) => {
                return Err(NotImplementedError::py_err(
                    "sorting is not implemented, please provide time-sorted arrays",
                ))
            }
            None => {
                if self.feature_evaluator.is_sorting_required() & !is_sorted(&t) {
                    return Err(ValueError::py_err("t must be in ascending order"));
                }
            }
        }

        let m = ArrWrapper::new(m, self.feature_evaluator.is_m_required())?;

        let w = sigma.and_then(|sigma| {
            if self.feature_evaluator.is_w_required() {
                let mut w = sigma.to_owned_array();
                w.mapv_inplace(|x| x.powi(-2));
                Some(ArrWrapper::Owned(w))
            } else {
                None
            }
        });

        let mut ts = TimeSeries::new(&t, &m, w.as_deref());

        let result = match fill_value {
            Some(x) => self.feature_evaluator.eval_or_fill(&mut ts, x),
            None => self
                .feature_evaluator
                .eval(&mut ts)
                .map_err(|e| ValueError::py_err(e.to_string()))?,
        };
        Ok(result.into_pyarray(py).to_owned())
    }
}

#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(*args)"]
struct Extractor {
    feature_extractor: light_curve_feature::FeatureExtractor<F>,
}

#[pymethods]
impl Extractor {
    #[new]
    #[args(args = "*")]
    fn __new__(args: &PyTuple) -> PyResult<(Self, PyFeatureEvaluator)> {
        let evals = args
            .iter()
            .map(|arg| {
                arg.downcast::<PyCell<PyFeatureEvaluator>>()
                    .map(|fe| fe.borrow().feature_evaluator.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let eval = light_curve_feature::FeatureExtractor::new(evals);
        Ok((
            Self {
                feature_extractor: eval.clone(),
            },
            PyFeatureEvaluator {
                feature_evaluator: Box::new(eval),
            },
        ))
    }

    #[getter]
    fn names(&self) -> Vec<&str> {
        self.feature_extractor.get_names()
    }
}

macro_rules! evaluator {
    ($name: ident, $eval: ty $(,)?) => {
        #[pyclass(extends = PyFeatureEvaluator)]
        #[text_signature = "()"]
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

evaluator!(
    AndersonDarlingNormal,
    light_curve_feature::AndersonDarlingNormal
);

#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(nstd=None)"]
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

#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(extractor, window=None, offset=None)"]
struct Bins {}

#[pymethods]
impl Bins {
    #[new]
    #[args(features, window = "None", offset = "None")]
    fn __new__(
        py: Python,
        features: PyObject,
        window: Option<F>,
        offset: Option<F>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        let mut eval = light_curve_feature::Bins::default();
        for x in features.extract::<&PyAny>(py)?.iter()? {
            let feature = x?
                .downcast::<PyCell<PyFeatureEvaluator>>()?
                .borrow()
                .feature_evaluator
                .clone();
            eval.add_feature(feature);
        }
        if let Some(window) = window {
            eval.set_window(window);
        }
        if let Some(offset) = offset {
            eval.set_offset(offset);
        }
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(eval),
            },
        ))
    }
}

evaluator!(Cusum, light_curve_feature::Cusum);

evaluator!(Eta, light_curve_feature::Eta);

evaluator!(EtaE, light_curve_feature::EtaE);

#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile=None)"]
struct InterPercentileRange {}

#[pymethods]
impl InterPercentileRange {
    #[new]
    #[args(quantile = "None")]
    fn __new__(quantile: Option<f32>) -> (Self, PyFeatureEvaluator) {
        let eval = match quantile {
            Some(quantile) => light_curve_feature::InterPercentileRange::new(quantile),
            None => light_curve_feature::InterPercentileRange::default(),
        };
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(eval),
            },
        )
    }
}

evaluator!(Kurtosis, light_curve_feature::Kurtosis);

evaluator!(LinearFit, light_curve_feature::LinearFit);

evaluator!(LinearTrend, light_curve_feature::LinearTrend);

#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile_numerator=None, quantile_denominator=None)"]
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

evaluator!(Mean, light_curve_feature::Mean);

evaluator!(Median, light_curve_feature::Median);

evaluator!(
    MedianAbsoluteDeviation,
    light_curve_feature::MedianAbsoluteDeviation,
);

#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile=None)"]
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
#[text_signature = "(quantile=None)"]
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

/// Find periodogram peaks
///
/// Parameters
/// ----------
/// peaks: int or None, optional
///     Number of peaks to find
///
/// resolution: float or None, optional
///     Resolution of frequency grid
///
/// max_freq_factor: float or None, optional
///     Mulitplier for Nyquist frequency
///
/// nyquist: str or float or None, optional
///     Type of Nyquist frequency. Could be one of:
///      - 'average': "Average" Nyquist frequency
///      - 'median': Nyquist frequency is defined by median time interval
///         between observations
///      - float: Nyquist frequency is defined by given quantile of time
///         intervals between observations
///
/// fast: bool or None, optional
///     Use "Fast" (approximate and FFT-based) or direct periodogram algorithm
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(peaks=None, resolution=None, max_freq_factor=None, nyquist=None, fast=None, features=None)"]
struct Periodogram {
    eval: light_curve_feature::Periodogram<F>,
}

impl Periodogram {
    fn create_eval(
        py: Python,
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<PyObject>,
        fast: Option<bool>,
        features: Option<PyObject>,
    ) -> PyResult<light_curve_feature::Periodogram<F>> {
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
                        "median" => Box::new(light_curve_feature::MedianNyquistFreq {}),
                        _ => return Err(ValueError::py_err(
                            "nyquist must be one of: None, 'average', 'median' or quantile value",
                        )),
                    }
                } else if let Ok(quantile) = nyquist.extract::<f32>(py) {
                    Box::new(light_curve_feature::QuantileNyquistFreq { quantile })
                } else {
                    return Err(ValueError::py_err(
                        "nyquist must be one of: None, 'average', 'median' or quantile value",
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
        if let Some(features) = features {
            for x in features.extract::<&PyAny>(py)?.iter()? {
                let feature = x?
                    .downcast::<PyCell<PyFeatureEvaluator>>()?
                    .borrow()
                    .feature_evaluator
                    .clone();
                eval.add_feature(feature);
            }
        }
        Ok(eval)
    }
}

#[pymethods]
impl Periodogram {
    #[new]
    #[args(
        peaks = "None",
        resolution = "None",
        max_freq_factor = "None",
        nyquist = "None",
        fast = "None",
        features = "None"
    )]
    fn __new__(
        py: Python,
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<PyObject>,
        fast: Option<bool>,
        features: Option<PyObject>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        Ok((
            Self {
                eval: Self::create_eval(
                    py,
                    peaks,
                    resolution,
                    max_freq_factor,
                    nyquist.as_ref().map(|x| x.clone_ref(py)),
                    fast,
                    features.as_ref().map(|x| x.clone_ref(py)),
                )?,
            },
            PyFeatureEvaluator {
                feature_evaluator: Box::new(Self::create_eval(
                    py,
                    peaks,
                    resolution,
                    max_freq_factor,
                    nyquist,
                    fast,
                    features,
                )?),
            },
        ))
    }

    /// Angular frequencies and periodogram values
    #[text_signature = "(t, m, err2=None)"]
    fn freq_power(&self, py: Python, t: &Arr, m: &Arr) -> PyResult<(Py<Arr>, Py<Arr>)> {
        let t = ArrWrapper::new(t, true)?;
        let m = ArrWrapper::new(m, true)?;
        let mut ts = TimeSeries::new(&t, &m, None);
        let (freq, power) = self.eval.freq_power(&mut ts);
        Ok((
            freq.into_pyarray(py).to_owned(),
            power.into_pyarray(py).to_owned(),
        ))
    }
}

evaluator!(ReducedChi2, light_curve_feature::ReducedChi2);

evaluator!(Skew, light_curve_feature::Skew);

evaluator!(StandardDeviation, light_curve_feature::StandardDeviation);

evaluator!(StetsonK, light_curve_feature::StetsonK);

evaluator!(WeightedMean, light_curve_feature::WeightedMean);

evaluator!(Duration, light_curve_feature::antifeatures::Duration);

evaluator!(
    MaximumTimeInterval,
    light_curve_feature::antifeatures::MaximumTimeInterval
);

evaluator!(
    MinimumTimeInterval,
    light_curve_feature::antifeatures::MinimumTimeInterval
);

evaluator!(
    ObservationCount,
    light_curve_feature::antifeatures::ObservationCount
);

evaluator!(TimeMean, light_curve_feature::antifeatures::TimeMean);

evaluator!(
    TimeStandardDeviation,
    light_curve_feature::antifeatures::TimeStandardDeviation
);

#[pymodule]
fn antifeatures(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Duration>()?;
    m.add_class::<MaximumTimeInterval>()?;
    m.add_class::<MinimumTimeInterval>()?;
    m.add_class::<ObservationCount>()?;
    m.add_class::<TimeMean>()?;
    m.add_class::<TimeStandardDeviation>()?;

    Ok(())
}

#[pymodule]
fn light_curve(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<Extractor>()?;

    m.add_class::<Amplitude>()?;
    m.add_class::<AndersonDarlingNormal>()?;
    m.add_class::<BeyondNStd>()?;
    m.add_class::<Bins>()?;
    m.add_class::<Cusum>()?;
    m.add_class::<Eta>()?;
    m.add_class::<EtaE>()?;
    m.add_class::<InterPercentileRange>()?;
    m.add_class::<Kurtosis>()?;
    m.add_class::<LinearFit>()?;
    m.add_class::<LinearTrend>()?;
    m.add_class::<MagnitudePercentageRatio>()?;
    m.add_class::<MaximumSlope>()?;
    m.add_class::<Mean>()?;
    m.add_class::<Median>()?;
    m.add_class::<MedianAbsoluteDeviation>()?;
    m.add_class::<MedianBufferRangePercentage>()?;
    m.add_class::<PercentAmplitude>()?;
    m.add_class::<PercentDifferenceMagnitudePercentile>()?;
    m.add_class::<Periodogram>()?;
    m.add_class::<ReducedChi2>()?;
    m.add_class::<Skew>()?;
    m.add_class::<StandardDeviation>()?;
    m.add_class::<StetsonK>()?;
    m.add_class::<WeightedMean>()?;

    m.add_wrapped(wrap_pymodule!(antifeatures))?;

    Ok(())
}
