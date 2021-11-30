use enum_dispatch::enum_dispatch;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[enum_dispatch]
pub trait LnPrior1DTrait: Clone + Debug + Serialize + DeserializeOwned {
    fn ln_prior_1d(&self, x: f64) -> f64;
}

/// Natural logarithm of prior for a single parameter of the curve-fit problem
#[enum_dispatch(LnPrior1DTrait)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum LnPrior1D {
    None(NoneLnPrior1D),
    LogNormal(LogNormalLnPrior1D),
    LogUniform(LogUniformLnPrior1D),
    Normal(NormalLnPrior1D),
    Uniform(UniformLnPrior1D),
    Mix(MixLnPrior1D),
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct NoneLnPrior1D {}

impl LnPrior1DTrait for NoneLnPrior1D {
    fn ln_prior_1d(&self, _x: f64) -> f64 {
        0.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(
    into = "LogNormalLnPrior1DParameters",
    from = "LogNormalLnPrior1DParameters"
)]
pub struct LogNormalLnPrior1D {
    mu: f64,
    inv_std2: f64,
    ln_prob_coeff: f64,
}

impl LogNormalLnPrior1D {
    pub fn new(mu: f64, std: f64) -> Self {
        Self {
            mu,
            inv_std2: std.powi(-2),
            ln_prob_coeff: -f64::ln(std) - 0.5 * f64::ln(std::f64::consts::TAU),
        }
    }
}

impl LnPrior1DTrait for LogNormalLnPrior1D {
    fn ln_prior_1d(&self, x: f64) -> f64 {
        let ln_x = f64::ln(x);
        self.ln_prob_coeff - 0.5 * (self.mu - ln_x).powi(2) * self.inv_std2 - ln_x
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "NormalLnPrior1D")]
struct LogNormalLnPrior1DParameters {
    mu: f64,
    std: f64,
}

impl From<LogNormalLnPrior1D> for LogNormalLnPrior1DParameters {
    fn from(f: LogNormalLnPrior1D) -> Self {
        Self {
            mu: f.mu,
            std: f.inv_std2.recip().sqrt(),
        }
    }
}

impl From<LogNormalLnPrior1DParameters> for LogNormalLnPrior1D {
    fn from(f: LogNormalLnPrior1DParameters) -> Self {
        Self::new(f.mu, f.std)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(
    into = "LogUniformLnPrior1DParameters",
    from = "LogUniformLnPrior1DParameters"
)]
pub struct LogUniformLnPrior1D {
    ln_range: std::ops::RangeInclusive<f64>,
    ln_prob_coeff: f64,
}

impl LogUniformLnPrior1D {
    pub fn new(left: f64, right: f64) -> Self {
        assert!(left < right);
        let ln_left = f64::ln(left);
        let ln_right = f64::ln(right);
        Self {
            ln_range: ln_left..=ln_right,
            ln_prob_coeff: -f64::ln(ln_right - ln_left),
        }
    }
}

impl LnPrior1DTrait for LogUniformLnPrior1D {
    fn ln_prior_1d(&self, x: f64) -> f64 {
        let ln_x = f64::ln(x);
        if self.ln_range.contains(&ln_x) {
            self.ln_prob_coeff - ln_x
        } else {
            f64::NEG_INFINITY
        }
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "LogUniformLnPrior")]
struct LogUniformLnPrior1DParameters {
    ln_range: std::ops::RangeInclusive<f64>,
}

impl From<LogUniformLnPrior1D> for LogUniformLnPrior1DParameters {
    fn from(f: LogUniformLnPrior1D) -> Self {
        Self {
            ln_range: f.ln_range,
        }
    }
}

impl From<LogUniformLnPrior1DParameters> for LogUniformLnPrior1D {
    fn from(f: LogUniformLnPrior1DParameters) -> Self {
        Self {
            ln_prob_coeff: -f64::ln(f.ln_range.end() - f.ln_range.start()),
            ln_range: f.ln_range,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(into = "NormalLnPrior1DParameters", from = "NormalLnPrior1DParameters")]
pub struct NormalLnPrior1D {
    mu: f64,
    inv_std2: f64,
    ln_prob_coeff: f64,
}

impl NormalLnPrior1D {
    pub fn new(mu: f64, std: f64) -> Self {
        Self {
            mu,
            inv_std2: std.powi(-2),
            ln_prob_coeff: -f64::ln(std) - 0.5 * f64::ln(std::f64::consts::TAU),
        }
    }
}

impl LnPrior1DTrait for NormalLnPrior1D {
    fn ln_prior_1d(&self, x: f64) -> f64 {
        self.ln_prob_coeff - 0.5 * (self.mu - x).powi(2) * self.inv_std2
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "NormalLnPrior1D")]
struct NormalLnPrior1DParameters {
    mu: f64,
    std: f64,
}

impl From<NormalLnPrior1D> for NormalLnPrior1DParameters {
    fn from(f: NormalLnPrior1D) -> Self {
        Self {
            mu: f.mu,
            std: f.inv_std2.recip().sqrt(),
        }
    }
}

impl From<NormalLnPrior1DParameters> for NormalLnPrior1D {
    fn from(f: NormalLnPrior1DParameters) -> Self {
        Self::new(f.mu, f.std)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(
    into = "UniformLnPrior1DParameters",
    from = "UniformLnPrior1DParameters"
)]
pub struct UniformLnPrior1D {
    range: std::ops::RangeInclusive<f64>,
    ln_prob: f64,
}

impl UniformLnPrior1D {
    pub fn new(left: f64, right: f64) -> Self {
        assert!(left < right);
        Self {
            range: left..=right,
            ln_prob: -f64::ln(right - left),
        }
    }
}

impl LnPrior1DTrait for UniformLnPrior1D {
    fn ln_prior_1d(&self, x: f64) -> f64 {
        if self.range.contains(&x) {
            self.ln_prob
        } else {
            f64::NEG_INFINITY
        }
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "UniformLnPrior")]
struct UniformLnPrior1DParameters {
    range: std::ops::RangeInclusive<f64>,
}

impl From<UniformLnPrior1D> for UniformLnPrior1DParameters {
    fn from(f: UniformLnPrior1D) -> Self {
        Self { range: f.range }
    }
}

impl From<UniformLnPrior1DParameters> for UniformLnPrior1D {
    fn from(f: UniformLnPrior1DParameters) -> Self {
        Self::new(*f.range.start(), *f.range.end())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MixLnPrior1D {
    mix: Vec<(f64, LnPrior1D)>,
}

impl MixLnPrior1D {
    /// Create MixLnPrior1D from pairs of a weight (positive number) and an instance of `LnPrior1D`
    pub fn new(mut weight_prior_pairs: Vec<(f64, LnPrior1D)>) -> Self {
        let total_weight: f64 = weight_prior_pairs.iter().map(|(weight, _)| *weight).sum();
        weight_prior_pairs
            .iter_mut()
            .for_each(|(weight, _)| *weight /= total_weight);
        Self {
            mix: weight_prior_pairs,
        }
    }
}

impl LnPrior1DTrait for MixLnPrior1D {
    fn ln_prior_1d(&self, x: f64) -> f64 {
        f64::ln(
            self.mix
                .iter()
                .map(|(weight, prior)| *weight * f64::exp(prior.ln_prior_1d(x)))
                .sum(),
        )
    }
}
