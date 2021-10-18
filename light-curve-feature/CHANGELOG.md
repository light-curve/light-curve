# Changelog

All notable changes to `light-curve-feature` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

—

### Changed

—

### Deprecated

—

### Removed

—

### Fixed

- Update clap to 3.0.0-beta5, it is used for the example executable only

### Security

—

## [0.3.3]

### Fixed

- Equation in `VillarFit` was different from the paper
- `VillarFit`'s and `BazinFit`'s amplitude, tau_rise, tau_fall and plateau duration (for `VillarFit`) are insured to be positive

## [0.3.2]

### Changed

- Rust-specific docs for `BazinFit` and `VillarFit` are moved from struct docs to `::new()` docs

### Fixed

- Equation rendering in `VillarFit` HTML docs


## [0.3.1]

### Changed

- `periodogram` module and `_PeriodogramPeaks` are hidden from the docs
- Update katex for <http://docs.rs> to 0.13.13

### Fixed

- Docs for `Extractor`, `FeatureEvaluator`, `AndersonDarlingNormal`, `Bins`, `Cusum`, `EtaE`, `InterPercentileRange`, `LinearFit`, `LinearTrend`, `Median`, `PercentAmplitude`, `Periodogram`, `ReducedChi2`, `VillarFit` and `BazinFit`

## [0.3.0]

### Added

- This `CHANGELOG.md` file
- `Feature` enum containing all available features, it implements `FeatureEvaluator`
- (De)serialization with [`serde`](http://serde.rs) is implemented for all features
- JSON schema generation with [`schemars`](http://graham.cool/schemars/) is implemented for all features  
- `TimeSeries` and `DataSample` use `ndarray::CowArray` to hold data, their constructors accept `ArrayBase` objects
- Static method `::doc()` for every feature, it returns language-agnostic feature evaluator description
- `examples` directory with an example which fits and plots some SN Ia light curves
- "private" sub-crate `light-curve-feature-test-util` with common tools for tests, benchmarks and examples

### Changed

- `FeatureExtractor`, `Bins` and `Periodogram` accepts `Feature` enum objects instead of `Box<FeatureEvaluator>`
- Periodogram-related `NyquistFreq` and `PeriodogramPower` are changed from traits to enums
- `TimeSeries::new` accepts non-optional weights, use `TimeSeries::new_without_weight` to initialize time series with unity weight array 
- `BazinFit` is parameterized by a curve-fit algorithm, MCMC and GSL's LMSDER are available, but the last one requires non-default `gsl` Cargo feature. MCMC becomes the default algorithm, some wide boundary conditions are included  
- Rename `BazinFit::get_names()[1]` from "bazin_fit_offset" to "bazin_fit_baseline"
- Add `VillarFit` feature for the Villar function [arXiv:1905.07422](http://arxiv.org/abs/1905.07422), see `BazinFit` above for technical details
- `LinearTrend` requires at least three observations and returns three values: slope, its error and standard deviation of noise (new) 
- Publicly exported stuff

## [0.2.x]

—

## [0.1.x]

—
