# `light-curve` processing toolbox for Python

This package provides a collection of light curve feature extractions classes.

## Installation

```sh
python3 -mpip install light-curve
```

The package is tested on Linux (x86-64, aarch64, ppc64) and macOS (x86-64). x86-64 pre-build wheels are available, other systems are required to have the Rust toolchain to build the package, please install it using your OS package manager of [`rustup` script](https://rustup.rs).

## Feature evaluators

Most of the classes implement various feature evaluators useful for astrophysical sources classification and
characterisation using their light curves.

```python
import light_curve as lc
import numpy as np

# Time values can be non-evenly separated but must be an ascending array
n = 101
t = np.linspace(0.0, 1.0, n)
perfect_m = 1e3 * t + 1e2
err = np.sqrt(perfect_m)
m = perfect_m + np.random.normal(0, err)

# Half-amplitude of magnitude
amplitude = lc.Amplitude()
# Fraction of points beyond standard deviations from mean
beyond_std = lc.BeyondNStd(nstd=1)
# Slope, its error and reduced chi^2 of linear fit
linear_fit = lc.LinearFit()
# Feature extractor, it will evaluate all features in more efficient way
extractor = lc.Extractor(amplitude, beyond_std, linear_fit)

# Array with all 5 extracted features
result = extractor(t, m, err)

print('\n'.join("{} = {:.2f}".format(name, value) for name, value in zip(extractor.names, result)))

# Run in parallel for multiple light curves:
results = amplitude.many([(t[:i], m[:i], err[:i]) for i in range(int(0.5 * n), n)], n_jobs=-1)
print("Amplitude of amplitude is {:.2f}".format(np.ptp(results)))
```

Print feature classes list
```python
import light_curve as lc

print([x for x in dir(lc) if hasattr(getattr(lc, x), "names")])
```

Read feature docs
```python
import light_curve as lc

help(lc.BazinFit)
```

### Experimental extractors

From the technical point of view the package consists of two parts: a wrapper for [`light-curve-feature` Rust crate](https://crates.io/crates/light-curve-feature) (`light_curve_ext` sub-package) and pure Python sub-package `light_curve_py`.
We use the Python implementation of feature extractors to test Rust implementation and to implement new experimental extractors.
Please note, that the Python implementation is much slower for the most of the extractors and doesn't provide the same functionality as the Rust implementation.
However, the Python implementation provides some new feature extractors you can find useful.

You can manually use extractors from both implementations:

```python
import numpy as np
from numpy.testing import assert_allclose
from light_curve.light_curve_ext import LinearTrend as RustLinearTrend
from light_curve.light_curve_py import LinearTrend as PythonLinearTrend

rust_fe = RustLinearTrend()
py_fe = PythonLinearTrend()

n = 100
t = np.sort(np.random.normal(size=n))
m = 3.14 * t - 2.16 + np.random.normal(size=n)

assert_allclose(rust_fe(t, m), py_fe(t, m),
                err_msg="Python and Rust implementations must provide the same result")
```

This should print a warning about experimental status of the Python class

## dm-dt map

Class `DmDt` provides dm–dt mapper (based on [Mahabal et al. 2011](https://ui.adsabs.harvard.edu/abs/2011BASI...39..387M/abstract), [Soraisam et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...892..112S/abstract)). It is a Python wrapper for [`light-curve-dmdt` Rust crate](https://crates.io/crates/light-curve-dmdt).

```python
import numpy as np
from light_curve import DmDt
from numpy.testing import assert_array_equal

dmdt = DmDt.from_borders(min_lgdt=0, max_lgdt=np.log10(3), max_abs_dm=3, lgdt_size=2, dm_size=4, norm=[])

t = np.array([0, 1, 2], dtype=np.float32)
m = np.array([0, 1, 2], dtype=np.float32)

desired = np.array(
    [
        [0, 0, 2, 0],
        [0, 0, 0, 1],
    ]
)
actual = dmdt.points(t, m)

assert_array_equal(actual, desired)
```
