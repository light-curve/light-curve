# `light-curve`

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/light-curve/light-curve/master.svg)](https://results.pre-commit.ci/latest/github/light-curve/light-curve/master)

## Irregular time series analysis toolbox for Rust and Python

The project is aimed to build high-performance tools for light curve analysis suitable to process alert and archival data of current [ZTF](https://ztf.caltech.edu) and future [Vera Rubin Observatory LSST](https://lsst.org) photometric surveys.

The most of the code base is written on Rust and split into several crates.
This repository contains `light-curve` crate only, which will be an umbrella for all other Rust crates.
Please refer to a specific repository in this organisation to find a code you need:

- `light-curve` this repo containing a `light-curve` crate which is WIP
- `light-curve-common` ![Crates.io](https://img.shields.io/crates/v/light-curve-common) [![docs.rs badge](https://docs.rs/light-curve-common/badge.svg)](https://docs.rs/light-curve-common) (Currently in this repo) Common tools for other crates
- [`light-curve-dmdt`](https://github.com/light-curve/light-curve-dmdt) ![Crates.io](https://img.shields.io/crates/v/light-curve-dmdt) [![docs.rs badge](https://docs.rs/light-curve-dmdt/badge.svg)](https://docs.rs/light-curve-dmdt) [dm-dt](https://arxiv.org/abs/1709.06257) dm-dt mapper crate and executable
- [`light-curve-feature`](https://github.com/light-curve/light-curve-feature) ![Crates.io](https://img.shields.io/crates/v/light-curve-feature) [![docs.rs badge](https://docs.rs/light-curve-feature/badge.svg)](https://docs.rs/light-curve-feature) A collection of features to be extracted from light curves
- `light-curve-interpol`**WIP** (currently in this repo) Light curve interpolation tools. Currently it includes linear interpolation only
- [`light-curve-python`](https://github.com/light-curve/light-curve-python) [![PyPI version](https://badge.fury.io/py/light-curve.svg)](https://pypi.org/project/light-curve/) Python `light-curve` and `light-curve-python` packages, a Python interface to `light-curve-feature` and `light-curve-dmdt` Cargo crates. It also contains a few features implemented in Python
- [`base-docker-images`](https://github.com/light-curve/base-docker-images) Custom Docker images. Currently used for internal CI needs only
- [`test-data`](https://github.com/light-curve/test-data) Light-curve data required for testing, benchmarking and development of the new features

### Versioning

All package versions are unrelated, which means that `light-curve-python v0.a.b` can depend on `light-curve-dmdt v0.d.e` and `light-curve-feature v0.f.g`. We use [semver](https://semver.org), so a release `0.3.0` can be backward incompatible with `0.2.5` version of the same crate. Please refer to `CHAMGELOG.md` files for the changes between versions.

### Citation

If you found this project useful for your research please cite [Malanchev et al., 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract)

```bibtex
@ARTICLE{2021MNRAS.502.5147M,
       author = {{Malanchev}, K.~L. and {Pruzhinskaya}, M.~V. and {Korolev}, V.~S. and {Aleo}, P.~D. and {Kornilov}, M.~V. and {Ishida}, E.~E.~O. and {Krushinsky}, V.~V. and {Mondon}, F. and {Sreejith}, S. and {Volnova}, A.~A. and {Belinski}, A.~A. and {Dodin}, A.~V. and {Tatarnikov}, A.~M. and {Zheltoukhov}, S.~G. and {(The SNAD Team)}},
        title = "{Anomaly detection in the Zwicky Transient Facility DR3}",
      journal = {\mnras},
     keywords = {methods: data analysis, astronomical data bases: miscellaneous, stars: variables: general, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2021,
        month = apr,
       volume = {502},
       number = {4},
        pages = {5147-5175},
          doi = {10.1093/mnras/stab316},
archivePrefix = {arXiv},
       eprint = {2012.01419},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
