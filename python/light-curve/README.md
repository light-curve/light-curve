# Light-curve feature extraction Python package

Example

```python
import light_curve as lc
import numpy as np  # only numpy arrays with dtype==float64  are supported


t = np.linspace(0.0, 3.0, 100)
m = np.sin(t)
sigma = np.full_like(t, 0.1)

amplitude = lc.Amplitude()
beyond_2_std = lc.BeyondNStd(2)
extr = lc.Extractor(amplitude, beyond_2_std)

print(f'Half-amplitude is {amplitude(t, m, sigma).item()}')
print(f'Fraction of observations beyond 2 std from mean is {beyond_2_std(t, m, sigma).item()}')
print(f'All features: {extr(t, m, sigma)}')

# show module-level docs
help(lc)
```
