use crate::float_trait::Float;
use crate::recurrent_sin_cos::RecurrentSinCos;
use crate::time_series::{DataSample, TimeSeries};
use conv::ConvUtil;

#[derive(Clone)]
struct PeriodogramSums<T> {
    m_sin: T,
    m_cos: T,
    sin2: T,
}

impl<T: Float> Default for PeriodogramSums<T> {
    fn default() -> Self {
        Self {
            m_sin: T::zero(),
            m_cos: T::zero(),
            sin2: T::zero(),
        }
    }
}

pub struct Periodogram<T> {
    freq: Vec<T>,
    power: Vec<T>,
}

impl<T> Periodogram<T>
where
    T: Float,
{
    fn new(freq: Vec<T>, power: Vec<T>) -> Self {
        assert_eq!(freq.len(), power.len());
        Self { freq, power }
    }

    fn tau(t: &[T], freq: &[T]) -> Vec<T> {
        let mut sin_cos: Vec<_> = t
            .iter()
            .map(|&x| RecurrentSinCos::new(T::two() * freq[0] * x))
            .collect();
        freq.iter()
            .map(|&omega| {
                let mut sum_sin = T::zero();
                let mut sum_cos = T::zero();
                for s_c in sin_cos.iter_mut() {
                    let (sin, cos) = s_c.next().unwrap();
                    sum_sin += sin;
                    sum_cos += cos;
                }
                T::half() / omega * T::atan2(sum_sin, sum_cos)
            })
            .collect()
    }

    fn p_n(ts: &mut TimeSeries<T>, freq: &[T]) -> Vec<T> {
        let tau_ = Self::tau(ts.t.sample, freq);

        let m_mean = ts.m.get_mean();

        freq.iter()
            .zip(tau_.iter())
            .map(|(&omega, &tau)| {
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
            })
            .collect()
    }

    pub fn from_time_series(ts: &mut TimeSeries<T>, freq_factors: &PeriodogramFreqFactors) -> Self {
        let freq = freq_factors.get(&mut ts.t);
        let power: Vec<_> = Self::p_n(ts, &freq);
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

pub struct PeriodogramFreqFactors {
    resolution: usize,
    nyquist: usize,
}

impl PeriodogramFreqFactors {
    pub fn new(resolution: usize, nyquist: usize) -> Self {
        assert!(resolution > 0);
        assert!(nyquist > 0);
        Self {
            resolution,
            nyquist,
        }
    }

    fn get<T>(&self, t: &mut DataSample<T>) -> Vec<T>
    where
        T: Float,
    {
        let obs_time = t.get_max() - t.get_min();
        let delta_freq = T::two() * T::PI() / (obs_time * self.resolution.value_as::<T>().unwrap());
        let freq_size = self.resolution * self.nyquist * t.sample.len() / 2;
        (1..freq_size + 1)
            .map(|i| delta_freq * i.value_as::<T>().unwrap())
            .collect()
    }
}

impl Default for PeriodogramFreqFactors {
    fn default() -> Self {
        Self {
            resolution: 10,
            nyquist: 2,
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
            &[Periodogram::p_n(&mut ts, &[OMEGA_SIN])[0] * 2.0 / (N as f64 - 1.0)],
            &[1.0],
            1.0 / (N as f64),
        );

        // import numpy as np
        // from scipy.signal import lombscargle
        //
        // t = np.arange(100)
        // m = np.sin(0.07 * t)
        // y = (m - m.mean()) / m.std(ddof=1)
        // freq = np.linspace(0.01, 0.05, 5)
        // print(lombscargle(t, y, freq, precenter=True, normalize=False))

        let freq = linspace(0.01, 0.05, 5);
        let desired = [
            16.99018018,
            18.57722516,
            21.96049738,
            28.15056806,
            36.66519435,
        ];
        let actual = Periodogram::p_n(&mut ts, &freq);
        all_close(&actual[..], &desired[..], 1e-6);
    }
}
