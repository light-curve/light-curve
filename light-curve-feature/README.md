`light-curve-feature` is a part of [`light-curve`](https://docs.rs/light-curve) family that
implements extraction of numerous light curve features used in astrophysics.

[![docs.rs badge](https://docs.rs/light-curve-feature/badge.svg)](https://docs.rs/light-curve-feature)

All features are available in [Feature](crate::Feature) enum, and the recommended way to extract multiple features at
once is [FeatureExtractor](crate::FeatureExtractor) struct built from a `Vec<Feature>`. Data is represented by
[TimeSeries](crate::TimeSeries) struct built from time, magnitude (or flux) and weight arrays, all having the same length. Note
that multiple features interpret weight array as inversed squared observation errors.

```rust
use light_curve_feature::prelude::*;

// Let's find amplitude and reduced Chi-squared of the light curve
let fe = FeatureExtractor::from_features(vec![
    Amplitude::new().into(),
    ReducedChi2::new().into()
]);
// Define light curve
let time = [0.0, 1.0, 2.0, 3.0, 4.0];
let magn = [-1.0, 2.0, 1.0, 3.0, 4.5];
let weights = [5.0, 10.0, 2.0, 10.0, 5.0]; // inverse squared magnitude errors
let mut ts = TimeSeries::new(&time, &magn, &weights);
// Get results and print
let result = fe.eval(&mut ts)?;
let names = fe.get_names();
println!("{:?}", names.iter().zip(result.iter()).collect::<Vec<_>>());
# Ok::<(), EvaluatorError>(())
```

There are a couple of meta-features, which transform a light curve before feature extraction. For example
[Bins](crate::Bins) feature accumulates data inside time-windows and extracts features from this new light curve.

```rust
use light_curve_feature::prelude::*;
use ndarray::Array1;

// Define features, "raw" MaximumSlope and binned with zero offset and 1-day window
let max_slope: Feature<_> = MaximumSlope::default().into();
let bins: Feature<_> = {
    let mut bins = Bins::new(1.0, 0.0);
    bins.add_feature(max_slope.clone());
    bins.into()
};
let fe = FeatureExtractor::from_features(vec![max_slope, bins]);
// Define light curve
let time = [0.1, 0.2, 1.1, 2.1, 2.1];
let magn = [10.0, 10.1, 10.5, 11.0, 10.9];
// We don't need weight for MaximumSlope, this would assign unity weight
let mut ts = TimeSeries::new_without_weight(&time, &magn);
// Get results and print
let result = fe.eval(&mut ts)?;
println!("{:?}", result);
# Ok::<(), EvaluatorError>(())
```
