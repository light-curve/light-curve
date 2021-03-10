use conv::ConvUtil;
use enumflags2::{bitflags, BitFlags};
use itertools::Itertools;
use light_curve_dmdt as lcdmdt;
use light_curve_feature as lcf;
use ndarray::Array1 as NDArray;
use numpy::{Element, IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pymodule;
use std::ops::Deref;

type F = f64;
type Arr<T> = PyArray1<T>;
type PyArrF = Py<Arr<F>>;

enum ArrWrapper<'a, T> {
    Readonly(PyReadonlyArray1<'a, T>),
    Owned(NDArray<T>),
}

impl<'a, T> ArrWrapper<'a, T>
where
    T: Element + ndarray::NdFloat,
{
    /// Construct ndarray::Array1 wrapper of numpy array
    ///
    /// Right now it always returns Ok
    fn new(a: &'a PyArray1<T>, required: bool) -> Self {
        match (a.is_contiguous(), required) {
            (true, _) => Self::Readonly(a.readonly()),
            (false, true) => Self::Owned(a.to_owned_array()),
            (false, false) => Self::Owned(ndarray::Array1::<T>::zeros(a.len())),
        }
    }
}

impl<'a, T> Deref for ArrWrapper<'a, T>
where
    T: Element,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Readonly(a) => a.as_slice().unwrap(),
            Self::Owned(a) => a.as_slice().unwrap(),
        }
    }
}

fn is_sorted<T>(a: &[T]) -> bool
where
    T: PartialOrd,
{
    a.iter().tuple_windows().all(|(a, b)| a < b)
}

#[derive(FromPyObject)]
enum GenericArray1<'a> {
    #[pyo3(transparent, annotation = "np.ndarray[float32]")]
    Float32(&'a Arr<f32>),
    #[pyo3(transparent, annotation = "np.ndarray[float64]")]
    Float64(&'a Arr<f64>),
}

#[bitflags]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
enum NormFlag {
    LgDt,
    Max,
}

#[pyclass]
struct DmDt {
    dmdt_f64: lcdmdt::DmDt<f64>,
    dmdt_f32: lcdmdt::DmDt<f32>,
    norm: BitFlags<NormFlag>,
    error_func: lcdmdt::ErrorFunction,
}

impl DmDt {
    fn convert_t<T>(t: &Arr<T>, sorted: Option<bool>) -> PyResult<ArrWrapper<T>>
    where
        T: Element + ndarray::NdFloat,
    {
        let t = ArrWrapper::new(t, true);

        match sorted {
            Some(true) => {}
            Some(false) => {
                return Err(PyNotImplementedError::new_err(
                    "sorting is not implemented, please provide time-sorted arrays",
                ))
            }
            None => {
                if !is_sorted(&t) {
                    return Err(PyValueError::new_err("t must be in ascending order"));
                }
            }
        }

        Ok(t)
    }

    fn generic_points<T>(
        &self,
        py: Python,
        dmdt: &lcdmdt::DmDt<T>,
        t: &Arr<T>,
        m: &Arr<T>,
        sorted: Option<bool>,
    ) -> PyResult<PyObject>
    where
        T: Element + ndarray::NdFloat + lcdmdt::Float,
    {
        let t = Self::convert_t(t, sorted)?;
        let m = ArrWrapper::new(m, true);

        let mut result = dmdt.points(&t, &m).mapv(|x| x.value_as::<T>().unwrap());
        self.normalize(&mut result, dmdt, &t);
        Ok(result.into_pyarray(py).to_owned().into_py(py))
    }

    #[allow(clippy::too_many_arguments)]
    fn generic_gausses<T>(
        &self,
        py: Python,
        dmdt: &lcdmdt::DmDt<T>,
        t: &Arr<T>,
        m: &Arr<T>,
        sigma: &Arr<T>,
        sorted: Option<bool>,
    ) -> PyResult<PyObject>
    where
        T: Element + ndarray::NdFloat + lcdmdt::ErfFloat,
    {
        let t = Self::convert_t(t, sorted)?;
        let m = ArrWrapper::new(m, true);
        let err2 = {
            let mut err2 = sigma.to_owned_array();
            err2.mapv_inplace(|x| x.powi(2));
            ArrWrapper::Owned(err2)
        };

        let mut result = dmdt.gausses(&t, &m, &err2, &self.error_func);
        self.normalize(&mut result, dmdt, &t);

        Ok(result.into_pyarray(py).to_owned().into_py(py))
    }

    fn normalize<T>(&self, a: &mut ndarray::Array2<T>, dmdt: &lcdmdt::DmDt<T>, t: &[T])
    where
        T: Element + ndarray::NdFloat + lcdmdt::Float,
    {
        if self.norm.contains(NormFlag::LgDt) {
            let lgdt = dmdt.lgdt_points(&t);
            let lgdt_no_zeros = lgdt.mapv(|x| {
                if x == 0 {
                    T::one()
                } else {
                    x.value_as::<T>().unwrap()
                }
            });
            *a /= &lgdt_no_zeros.into_shape((a.nrows(), 1)).unwrap();
        }
        if self.norm.contains(NormFlag::Max) {
            let max = *a.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            if !max.is_zero() {
                a.mapv_inplace(|x| x / max);
            }
        }
    }
}

#[pymethods]
impl DmDt {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[args(
        min_lgdt,
        max_lgdt,
        max_abs_dm,
        lgdt_size,
        dm_size,
        norm = "vec![\"lgdt\"]",
        approx_erf = "false"
    )]
    fn __new__(
        min_lgdt: f64,
        max_lgdt: f64,
        max_abs_dm: f64,
        lgdt_size: usize,
        dm_size: usize,
        norm: Vec<&str>,
        approx_erf: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            dmdt_f64: lcdmdt::DmDt {
                lgdt_grid: lcdmdt::Grid::new(min_lgdt, max_lgdt, lgdt_size),
                dm_grid: lcdmdt::Grid::new(-max_abs_dm, max_abs_dm, dm_size),
            },
            dmdt_f32: lcdmdt::DmDt {
                lgdt_grid: lcdmdt::Grid::new(min_lgdt as f32, max_lgdt as f32, lgdt_size),
                dm_grid: lcdmdt::Grid::new(-max_abs_dm as f32, max_abs_dm as f32, dm_size),
            },
            norm: norm
                .iter()
                .map(|&s| match s {
                    "lgdt" => Ok(NormFlag::LgDt),
                    "max" => Ok(NormFlag::Max),
                    _ => Err(PyValueError::new_err(format!(
                        "normalisation name {:?} is unknown, known names are: \"lgdt\", \"norm\"",
                        s
                    ))),
                })
                .collect::<PyResult<BitFlags<NormFlag>>>()?,
            error_func: match approx_erf {
                true => lcdmdt::ErrorFunction::Eps1Over1e3,
                false => lcdmdt::ErrorFunction::Exact,
            },
        })
    }

    #[args(t, m, sorted = "None")]
    fn points(
        &self,
        py: Python,
        t: GenericArray1,
        m: GenericArray1,
        sorted: Option<bool>,
    ) -> PyResult<PyObject> {
        match (t, m) {
            (GenericArray1::Float32(t), GenericArray1::Float32(m)) => {
                self.generic_points(py, &self.dmdt_f32, t, m, sorted)
            }
            (GenericArray1::Float64(t), GenericArray1::Float64(m)) => {
                self.generic_points(py, &self.dmdt_f64, t, m, sorted)
            }
            _ => Err(PyValueError::new_err("t and m must have the same dtype")),
        }
    }

    #[args(t, m, sigma, sorted = "None")]
    fn gausses(
        &self,
        py: Python,
        t: GenericArray1,
        m: GenericArray1,
        sigma: GenericArray1,
        sorted: Option<bool>,
    ) -> PyResult<PyObject> {
        match (t, m, sigma) {
            (
                GenericArray1::Float32(t),
                GenericArray1::Float32(m),
                GenericArray1::Float32(sigma),
            ) => self.generic_gausses(py, &self.dmdt_f32, t, m, sigma, sorted),
            (
                GenericArray1::Float64(t),
                GenericArray1::Float64(m),
                GenericArray1::Float64(sigma),
            ) => self.generic_gausses(py, &self.dmdt_f64, t, m, sigma, sorted),
            _ => Err(PyValueError::new_err(
                "t, m and sigma must have the same dtype",
            )),
        }
    }
}

#[pyclass(subclass, name = "_FeatureEvaluator")]
struct PyFeatureEvaluator {
    feature_evaluator: Box<dyn lcf::FeatureEvaluator<F>>,
}

#[pymethods]
impl PyFeatureEvaluator {
    #[call]
    #[args(t, m, sigma = "None", sorted = "None", fill_value = "None")]
    fn __call__(
        &self,
        py: Python,
        t: &Arr<F>,
        m: &Arr<F>,
        sigma: Option<&Arr<F>>,
        sorted: Option<bool>,
        fill_value: Option<F>,
    ) -> PyResult<Py<Arr<F>>> {
        if t.len() != m.len() {
            return Err(PyValueError::new_err("t and m must have the same size"));
        }
        if let Some(sigma) = sigma {
            if t.len() != sigma.len() {
                return Err(PyValueError::new_err("t and sigma must have the same size"));
            }
        }
        if t.len() < self.feature_evaluator.min_ts_length() {
            return Err(PyValueError::new_err(format!(
                "input arrays must have size not smaller than {}, but having {}",
                self.feature_evaluator.min_ts_length(),
                t.len()
            )));
        }

        let is_t_required = match (
            self.feature_evaluator.is_t_required(),
            self.feature_evaluator.is_sorting_required(),
            sorted,
        ) {
            // feature requires t
            (true, _, _) => true,
            // t is required because sorting is required and data can be unsorted
            (false, true, Some(false)) | (false, true, None) => true,
            // sorting is required but user guarantees that data is already sorted
            (false, true, Some(true)) => false,
            // neither t or sorting is required
            (false, false, _) => false,
        };
        let t = ArrWrapper::new(t, is_t_required);
        match sorted {
            Some(true) => {}
            Some(false) => {
                return Err(PyNotImplementedError::new_err(
                    "sorting is not implemented, please provide time-sorted arrays",
                ))
            }
            None => {
                if self.feature_evaluator.is_sorting_required() & !is_sorted(&t) {
                    return Err(PyValueError::new_err("t must be in ascending order"));
                }
            }
        }

        let m = ArrWrapper::new(m, self.feature_evaluator.is_m_required());

        let w = sigma.and_then(|sigma| {
            if self.feature_evaluator.is_w_required() {
                let mut w = sigma.to_owned_array();
                w.mapv_inplace(|x| x.powi(-2));
                Some(ArrWrapper::Owned(w))
            } else {
                None
            }
        });

        let mut ts = lcf::TimeSeries::new(&t, &m, w.as_deref());

        let result = match fill_value {
            Some(x) => self.feature_evaluator.eval_or_fill(&mut ts, x),
            None => self
                .feature_evaluator
                .eval(&mut ts)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        };
        Ok(result.into_pyarray(py).to_owned())
    }

    /// Feature names
    #[getter]
    fn names(&self) -> Vec<&str> {
        self.feature_evaluator.get_names()
    }

    /// Feature descriptions
    #[getter]
    fn descriptions(&self) -> Vec<&str> {
        self.feature_evaluator.get_descriptions()
    }
}

/// Features extractor
///
/// Extract multiple features simultaneously, which should be more
/// performant than calling them separately
///
/// Parameters
/// ----------
/// *features : iterable
///     Feature objects
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(*args)"]
struct Extractor {}

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
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::FeatureExtractor::new(evals)),
            },
        ))
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

evaluator!(Amplitude, lcf::Amplitude);

evaluator!(AndersonDarlingNormal, lcf::AndersonDarlingNormal);

/// Fraction of observations beyond N*std from mean
///
/// Parameters
/// ----------
/// nstd : positive float
///     N
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(nstd, /)"]
struct BeyondNStd {}

#[pymethods]
impl BeyondNStd {
    #[new]
    fn __new__(nstd: F) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::BeyondNStd::new(nstd)),
            },
        )
    }
}

/// Binned time-series
///
/// Parameters
/// ----------
/// features : iterable
///     Features to extract from binned time-series
/// window : positive float
///     Width of binning interval in units of time
/// offset : float
///     Zero time moment
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(features, window, offset)"]
struct Bins {}

#[pymethods]
impl Bins {
    #[new]
    #[args(features, window, offset)]
    fn __new__(
        py: Python,
        features: PyObject,
        window: F,
        offset: F,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        let mut eval = lcf::Bins::default();
        for x in features.extract::<&PyAny>(py)?.iter()? {
            let feature = x?
                .downcast::<PyCell<PyFeatureEvaluator>>()?
                .borrow()
                .feature_evaluator
                .clone();
            eval.add_feature(feature);
        }
        eval.set_window(window);
        eval.set_offset(offset);
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(eval),
            },
        ))
    }
}

evaluator!(Cusum, lcf::Cusum);

evaluator!(Eta, lcf::Eta);

evaluator!(EtaE, lcf::EtaE);

evaluator!(ExcessVariance, lcf::ExcessVariance);

/// Inner percentile range
///
/// Parameters
/// ----------
/// quantile : positive float
///     Range is (100% * quantile, 100% * (1 - quantile))
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile)"]
struct InterPercentileRange {}

#[pymethods]
impl InterPercentileRange {
    #[new]
    #[args(quantile)]
    fn __new__(quantile: f32) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::InterPercentileRange::new(quantile)),
            },
        )
    }
}

evaluator!(Kurtosis, lcf::Kurtosis);

evaluator!(LinearFit, lcf::LinearFit);

evaluator!(LinearTrend, lcf::LinearTrend);

/// Ratio of two inter-percentile ranges
///
/// Parameters
/// ----------
/// quantile_numerator: positive float
///     Numerator is inter-percentile range (100% * q, 100% (1 - q))
/// quantile_denominator: positive float
///     Denominator is inter-percentile range (100% * q, 100% (1 - q))
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile_numerator, quantile_denominator)"]
struct MagnitudePercentageRatio {}

#[pymethods]
impl MagnitudePercentageRatio {
    #[new]
    #[args(quantile_numerator, quantile_denominator)]
    fn __new__(
        quantile_numerator: f32,
        quantile_denominator: f32,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        if !(0.0..0.5).contains(&quantile_numerator) {
            return Err(PyValueError::new_err(
                "quantile_numerator must be between 0.0 and 0.5",
            ));
        }
        if !(0.0..0.5).contains(&quantile_denominator) {
            return Err(PyValueError::new_err(
                "quantile_denumerator must be between 0.0 and 0.5",
            ));
        }
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::MagnitudePercentageRatio::new(
                    quantile_numerator,
                    quantile_denominator,
                )),
            },
        ))
    }
}

evaluator!(MaximumSlope, lcf::MaximumSlope);

evaluator!(Mean, lcf::Mean);

evaluator!(MeanVariance, lcf::MeanVariance);

evaluator!(Median, lcf::Median);

evaluator!(MedianAbsoluteDeviation, lcf::MedianAbsoluteDeviation,);

/// Median Buffer Range Percentage
///
/// Parameters
/// ----------
/// quantile : positive float
///     Relative range size
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile)"]
struct MedianBufferRangePercentage {}

#[pymethods]
impl MedianBufferRangePercentage {
    #[new]
    #[args(quantile)]
    fn __new__(quantile: F) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::MedianBufferRangePercentage::new(quantile)),
            },
        )
    }
}

evaluator!(PercentAmplitude, lcf::PercentAmplitude);

/// Percent Difference Magnitude Percentile
///
/// Parameters
/// ----------
/// quantile : positive float
///     Relative range size
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile)"]
struct PercentDifferenceMagnitudePercentile {}

#[pymethods]
impl PercentDifferenceMagnitudePercentile {
    #[new]
    #[args(quantile)]
    fn __new__(quantile: f32) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::PercentDifferenceMagnitudePercentile::new(
                    quantile,
                )),
            },
        )
    }
}

/// Periodogram-based features
///
/// Parameters
/// ----------
/// peaks : int or None, optional
///     Number of peaks to find
///
/// resolution : float or None, optional
///     Resolution of frequency grid
///
/// max_freq_factor : float or None, optional
///     Mulitplier for Nyquist frequency
///
/// nyquist : str or float or None, optional
///     Type of Nyquist frequency. Could be one of:
///      - 'average': "Average" Nyquist frequency
///      - 'median': Nyquist frequency is defined by median time interval
///         between observations
///      - float: Nyquist frequency is defined by given quantile of time
///         intervals between observations
///
/// fast : bool or None, optional
///     Use "Fast" (approximate and FFT-based) or direct periodogram algorithm
///
/// features : iterable or None, optional
///     Features to extract from periodogram considering it as a time-series
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(peaks=None, resolution=None, max_freq_factor=None, nyquist=None, fast=None, features=None)"]
struct Periodogram {
    eval: lcf::Periodogram<F>,
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
    ) -> PyResult<lcf::Periodogram<F>> {
        let mut eval = match peaks {
            Some(peaks) => lcf::Periodogram::new(peaks),
            None => lcf::Periodogram::default(),
        };
        if let Some(resolution) = resolution {
            eval.set_freq_resolution(resolution);
        }
        if let Some(max_freq_factor) = max_freq_factor {
            eval.set_max_freq_factor(max_freq_factor);
        }
        if let Some(nyquist) = nyquist {
            let nyquist_freq: Box<dyn lcf::NyquistFreq<F>> =
                if let Ok(s) = nyquist.extract::<&str>(py) {
                    match s {
                        "average" => Box::new(lcf::AverageNyquistFreq {}),
                        "median" => Box::new(lcf::MedianNyquistFreq {}),
                        _ => return Err(PyValueError::new_err(
                            "nyquist must be one of: None, 'average', 'median' or quantile value",
                        )),
                    }
                } else if let Ok(quantile) = nyquist.extract::<f32>(py) {
                    Box::new(lcf::QuantileNyquistFreq { quantile })
                } else {
                    return Err(PyValueError::new_err(
                        "nyquist must be one of: None, 'average', 'median' or quantile value",
                    ));
                };
            eval.set_nyquist(nyquist_freq);
        }
        if let Some(fast) = fast {
            if fast {
                eval.set_periodogram_algorithm(move || Box::new(lcf::PeriodogramPowerFft));
            } else {
                eval.set_periodogram_algorithm(move || Box::new(lcf::PeriodogramPowerDirect));
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
    #[text_signature = "(t, m)"]
    fn freq_power(&self, py: Python, t: &Arr<F>, m: &Arr<F>) -> (PyArrF, PyArrF) {
        let t = ArrWrapper::new(t, true);
        let m = ArrWrapper::new(m, true);
        let mut ts = lcf::TimeSeries::new(&t, &m, None);
        let (freq, power) = self.eval.freq_power(&mut ts);
        (
            freq.into_pyarray(py).to_owned(),
            power.into_pyarray(py).to_owned(),
        )
    }
}

evaluator!(ReducedChi2, lcf::ReducedChi2);

evaluator!(Skew, lcf::Skew);

evaluator!(StandardDeviation, lcf::StandardDeviation);

evaluator!(StetsonK, lcf::StetsonK);

evaluator!(WeightedMean, lcf::WeightedMean);

evaluator!(Duration, lcf::antifeatures::Duration);

evaluator!(MaximumTimeInterval, lcf::antifeatures::MaximumTimeInterval);

evaluator!(MinimumTimeInterval, lcf::antifeatures::MinimumTimeInterval);

evaluator!(ObservationCount, lcf::antifeatures::ObservationCount);

evaluator!(TimeMean, lcf::antifeatures::TimeMean);

evaluator!(
    TimeStandardDeviation,
    lcf::antifeatures::TimeStandardDeviation
);

/// Features highly dependent on time-series cadence
///
/// See feature interface documentation in the top-level module
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

/// High-performance time-series feature extractor
///
/// The module provides a collection of features to be extracted from time-series.
/// This module if based on a Rust crate `light-curve-feature`, features
/// documentation can be found on https://docs.rs/light-curve-feature
///
/// All features are represented by classes with callable instances, which have
/// the same attributes and call signature:
///
/// Attributes
/// ----------
/// names : list of feature names
///
/// Methods
/// -------
/// __call__(t, m, sigma=None, sorted=None, fill_value=None)
///     Extract features and return them as numpy array.
///
///     Parameters
///     ----------
///     t : numpy.ndarray of np.float64 dtype
///         Time moments
///     m : numpy.ndarray of np.float64 dtype
///         Power of observed signal (magnitude or flux)
///     sigma : numpy.ndarray of np.float64 dtype or None, optional
///         Observation error, if None it is assumed to be unity
///     sorted : bool or None, optional
///         Specifies if input array are sorted by time moments.
///         True is for certainly sorted, False is for unsorted.
///         If None is specified than sorting is checked and an exception is
///         raised for unsorted `t`
///     fill_value : float or None, optional
///         Value to fill invalid feature values, for example if count of
///         observations is not enough to find a proper value.
///         None causes exception for invalid features
///
#[pymodule]
fn light_curve(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<DmDt>()?;

    m.add_class::<PyFeatureEvaluator>()?;

    m.add_class::<Extractor>()?;

    m.add_class::<Amplitude>()?;
    m.add_class::<AndersonDarlingNormal>()?;
    m.add_class::<BeyondNStd>()?;
    m.add_class::<Bins>()?;
    m.add_class::<Cusum>()?;
    m.add_class::<Eta>()?;
    m.add_class::<EtaE>()?;
    m.add_class::<ExcessVariance>()?;
    m.add_class::<InterPercentileRange>()?;
    m.add_class::<Kurtosis>()?;
    m.add_class::<LinearFit>()?;
    m.add_class::<LinearTrend>()?;
    m.add_class::<MagnitudePercentageRatio>()?;
    m.add_class::<MaximumSlope>()?;
    m.add_class::<Mean>()?;
    m.add_class::<MeanVariance>()?;
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
