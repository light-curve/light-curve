use crate::evaluator::*;
use crate::extractor::FeatureExtractor;
use crate::features::antifeatures::*;
use crate::features::*;
use crate::float_trait::Float;
use crate::time_series::TimeSeries;

use enum_dispatch::enum_dispatch;
use std::fmt::Debug;

/// All features are available as variants of this enum
///
/// Consider to import [crate::FeatureEvaluator] as well
#[enum_dispatch(FeatureEvaluator<T>)]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(bound = "T: Float")]
#[non_exhaustive]
pub enum Feature<T>
where
    T: Float,
{
    // extractor
    FeatureExtractor(FeatureExtractor<T, Self>),
    // features
    Amplitude,
    AndersonDarlingNormal,
    BazinFit,
    BeyondNStd(BeyondNStd<T>),
    Bins(Bins<T, Self>),
    Cusum,
    Eta,
    EtaE,
    ExcessVariance,
    InterPercentileRange,
    Kurtosis,
    LinearFit,
    LinearTrend,
    MagnitudePercentageRatio,
    MaximumSlope,
    Mean,
    MeanVariance,
    Median,
    MedianAbsoluteDeviation,
    MedianBufferRangePercentage(MedianBufferRangePercentage<T>),
    PercentAmplitude,
    PercentDifferenceMagnitudePercentile,
    Periodogram(Periodogram<T, Self>),
    _PeriodogramPeaks,
    ReducedChi2,
    Skew,
    StandardDeviation,
    StetsonK,
    VillarFit,
    WeightedMean,
    // antifeatures
    Duration,
    MaximumTimeInterval,
    MinimumTimeInterval,
    ObservationCount,
    TimeMean,
    TimeStandardDeviation,
}
