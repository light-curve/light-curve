# `light-curve` processing toolbox for Python

This package provides a collection of light curve feature extractions classes.

## Installation

```sh
python3 -mpip install light-curve-python
```

Note that in the future the package will be renamed to `light-curve`

## Feature evaluators 

Most of the classes implement various feature evaluators useful for astrophysical sources classification and
characterisation using their light curves.

```python
import light_curve as lc
import numpy as np

# Time values can be non-evenly separated but must be an ascending array
t = np.linspace(0.0, 1.0, 101)
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

print('\n'.join(f'{name} = {value:.2f}' for name, value in zip(extractor.names, result)))
```

## dm-dt map

Class `DmDt` provides dm–dt mapper (based on [Mahabal et al. 2011](https://ui.adsabs.harvard.edu/abs/2011BASI...39..387M/abstract), [Soraisam et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...892..112S/abstract)).

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
