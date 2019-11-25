use crate::float_trait::Float;
use crate::time_series::{DataSample, TimeSeries};
use conv::ConvUtil;

pub struct Periodogram<T> {
    freq: Vec<T>,
    power: Vec<T>,
}

impl<T> Periodogram<T>
where
    T: Float,
{
    pub fn new(freq: Vec<T>, power: Vec<T>) -> Self {
        assert_eq!(freq.len(), power.len());
        Self { freq, power }
    }

    fn tau(t: &[T], omega: T) -> T {
        let two_omega: T = T::two() * omega;

        let mut sum_sin = T::zero();
        let mut sum_cos = T::zero();
        for &x in t {
            let (sin, cos) = T::sin_cos(two_omega * x);
            sum_sin += sin;
            sum_cos += cos;
        }
        T::half() / omega * T::atan2(sum_sin, sum_cos)
    }

    pub fn p_n(ts: &mut TimeSeries<T>, omega: T) -> T {
        let tau = Self::tau(ts.t.sample, omega);
        let m_mean = ts.m.get_mean();

        let mut sum_m_sin = T::zero();
        let mut sum_m_cos = T::zero();
        let mut sum_sin2 = T::zero();
        let it = ts.t.sample.iter().zip(ts.m.sample.iter());
        for (&x, &y) in it {
            let (sin, cos) = T::sin_cos(omega * (x - tau));
            sum_m_sin += (y - m_mean) * sin;
            sum_m_cos += (y - m_mean) * cos;
            sum_sin2 += sin.powi(2);
        }
        let sum_cos2 = ts.lenf() - sum_sin2;

        if (sum_m_sin.is_zero() & sum_sin2.is_zero())
            | (sum_m_cos.is_zero() & sum_cos2.is_zero())
            | ts.m.get_std().is_zero()
        {
            T::zero()
        } else {
            T::half() * (sum_m_sin.powi(2) / sum_sin2 + sum_m_cos.powi(2) / sum_cos2)
                / ts.m.get_std().powi(2)
        }
    }

    pub fn from_time_series(ts: &mut TimeSeries<T>, freq: &PeriodogramFreq<T>) -> Self {
        let freq = freq.get(&mut ts.t);
        let power: Vec<_> = freq.iter().map(|&omega| Self::p_n(ts, omega)).collect();
        Self::new(freq, power)
    }

    pub fn ts(&self) -> TimeSeries<T> {
        TimeSeries::new(&self.freq[..], &self.power[..], None)
    }

    pub fn get_freq(&self) -> &[T] {
        &self.freq[..]
    }

    pub fn get_power(&self) -> &[T] {
        &self.power[..]
    }
}

pub struct PeriodogramFreqFactors<T> {
    resolution: T,
    nyquist: T,
}

impl<T: Float> PeriodogramFreqFactors<T> {
    pub fn new(resolution: T, nyquist: T) -> Self {
        assert!(resolution > T::zero());
        assert!(nyquist > T::zero());
        Self {
            resolution,
            nyquist,
        }
    }
}

impl<T: Float> Default for PeriodogramFreqFactors<T> {
    fn default() -> Self {
        Self {
            resolution: T::ten(),
            nyquist: T::one(),
        }
    }
}

pub enum PeriodogramFreq<T> {
    Vector(Vec<T>),
    Factors(PeriodogramFreqFactors<T>),
}

impl<T: Float> PeriodogramFreq<T> {
    fn get(&self, t: &mut DataSample<T>) -> Vec<T> {
        match self {
            PeriodogramFreq::Vector(v) => v.clone(),
            PeriodogramFreq::Factors(f) => {
                let observation_time = t.get_max() - t.get_min();
                let min_freq = T::PI() / (f.resolution * observation_time);
                let max_freq = f.nyquist * T::PI() * t.sample.len().value_as::<T>().unwrap()
                    / observation_time;
                (1..) // we don't need zero frequency
                    .map(|i| min_freq * i.value_as::<T>().unwrap())
                    .take_while(|omega| *omega < max_freq + min_freq)
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use light_curve_common::{all_close, linspace};

    #[test]
    fn test_sin() {
        const OMEGA_SIN: f64 = 0.07;
        const N: usize = 100;
        let t = linspace(0.0, 99.0, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA_SIN * x)).collect();
        let mut ts = TimeSeries::new(&t[..], &m[..], None);
        all_close(
            &[Periodogram::p_n(&mut ts, OMEGA_SIN) * 2.0 / (N as f64 - 1.0)],
            &[1.0],
            1.0 / (N as f64),
        );

        // import numpy as np
        // from scipy.signal import lombscargle
        //
        // t = np.arange(100)
        // m = np.sin(0.07 * t)
        // y = (m - m.mean()) / m.std()
        // lombscargle(t, y, [0.01, 0.03, 0.1, 0.3, 1.0])

        let omegas = vec![0.01, 0.03, 0.1, 0.3, 1.0];
        let desired = [
            1.69901802e+01,
            2.19604974e+01,
            1.78799427e+01,
            1.96816849e-01,
            1.11222515e-02,
        ];
        let actual = Periodogram::from_time_series(&mut ts, &PeriodogramFreq::Vector(omegas)).power;
        all_close(&actual[..], &desired[..], 1e-6);
    }
}
