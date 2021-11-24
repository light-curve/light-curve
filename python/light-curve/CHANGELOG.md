# Changelog

All notable changes to `light-curve-dmdt` will be documented in this file.

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

—

### Security

—

## [0.4.0]

### Added

- Pure-Python implemented `OtsuSplit` feature extractor (experimental)
- Python snippets in `README.md` are tested, this requires `pytest-markdown`
- A lot of new unit-tests for the pure-Python implementation
- New benchmarks to compare pure-Python and Rust implementations performance

### Changed

- The Python package is renamed to `light-curve`, `light-curve-python` still exists as an alias
- Pure Python implementation of the most of the features are added, now Rust-implemented features live in `light_curve_ext` sub-package, while the Python implementation is in `light_curve_py`. Python-implemented feature extractors have an experimental status
- Now `dataclasses` (for Python 3.6 only) and `scipy` are required, they are used by the pure-Python implementation


### Fixed

- The Rust implemented classes `.__module__` was `buitins`, now it is `light_curve.light_curve_ext`

## [0.3.5]

### Changed

- Rust edition 2021
- Minimum supported Rust version is 1.56
- Maturin version 0.11

## [0.3.4]

### Fixed

- An exception shouldn't be raised for the case of small time-series length and non-`None` `fill_value`

## [0.3.3]

### Added

- Support `dtype=np.float32` for feature extractors
- `_FeatureEvaluator.many(lcs)` method for parallel execution

## [0.3.2]

### Changed

- Update `light-curve-feature` to `0.3.3`
- `__call__` docs for features

### Fixed

- `light-curve-feature` 0.3.3 fixes wrong `VillarFit` equation

## [0.3.1]

### Added

- `mcmc_niter` and `lmsder_niter` parameters of `*Fit` features

### Changed

- Amd64 PyPI packages are manylinux2014
- PyPI releases compiled with `gsl` feature enabled

### Deprecated

- `gsl` will become a default feature in future releases

### Fixed

- `*Fit` `algorithm` parameter was marked as optional in the docstrings

## [0.3.0]

### Added

- This `CHANGELOG.md` file
- `BazinFit` is enabled
- `VillarFit` feature

### Changed

- Remove `DmDt.point_from_columnar` and `DmDt.gausses_from_columnar`
- Update `ndarray` to 0.15.3 and `light-curve-dmdt` to 0.4.0
- Update `rust-numpy` and `pyo3` to 0.14
- Update `light-curve-feature` to 0.3.1
- Rename `nonlinear-fit` Cargo feature to `gsl`
- Docstrings improvements

## [0.2.x]

—

## [0.1.x]

—
