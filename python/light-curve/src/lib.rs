use pyo3::prelude::*;
use pyo3::wrap_pymodule;

mod arr_wrapper;

mod dmdt;
use dmdt::DmDt;

mod errors;

mod features;
use features::*;

mod sorted;

/// High-performance time-series feature extractor
///
/// The module provides a collection of features to be extracted from unevenly
/// separated time-series. This module if based on Rust crates
/// `light-curve-feature` & `light-curve-dmdt`.
///
/// dm-lg(dt) maps generator is represented by `DmDt` class, it has different
/// interface from all other features, see its documentation for details.
///
/// Features documentation can be found on https://docs.rs/light-curve-feature
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

    #[cfg(feature = "nonlinear-fit")]
    m.add_class::<BazinFit>()?;

    m.add_wrapped(wrap_pymodule!(antifeatures))?;

    Ok(())
}
