use pyo3::prelude::*;
use pyo3::wrap_pymodule;

mod cont_array;

mod dmdt;
use dmdt::DmDt;

mod errors;

mod features;
use features::*;

mod sorted;

/// High-performance time-series feature extractor
///
/// The module provides a collection of features to be extracted from unevenly separated
/// time-series. This module if based on Rust crates `light-curve-feature` & `light-curve-dmdt`.
///
/// dm-lg(dt) maps generator is represented by `DmDt` class, while all other classes are
/// feature extractors
#[pymodule]
fn light_curve(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add("_built_with_gsl", {
        #[cfg(feature = "gsl")]
        {
            true
        }
        #[cfg(not(feature = "gsl"))]
        {
            false
        }
    })?;
    m.add("_fft_backend", {
        #[cfg(feature = "fftw-static")]
        {
            "statically linked FFTW"
        }
        #[cfg(feature = "fftw-dynamic")]
        {
            "dynamically linked FFTW"
        }
        #[cfg(feature = "mkl")]
        {
            "Intel MKL"
        }
    })?;

    m.add_class::<DmDt>()?;

    m.add_class::<PyFeatureEvaluator>()?;

    m.add_class::<Extractor>()?;

    m.add_class::<Amplitude>()?;
    m.add_class::<AndersonDarlingNormal>()?;
    m.add_class::<BazinFit>()?;
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
    m.add_class::<VillarFit>()?;
    m.add_class::<WeightedMean>()?;

    #[cfg(feature = "nonlinear-fit")]
    m.add_class::<BazinFit>()?;

    m.add_wrapped(wrap_pymodule!(antifeatures))?;

    Ok(())
}
