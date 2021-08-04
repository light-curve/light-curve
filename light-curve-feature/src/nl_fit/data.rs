use crate::float_trait::Float;
use crate::time_series::{DataSample, TimeSeries};

use conv::ConvUtil;
use ndarray::Array1;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Data<T> {
    pub t: Array1<T>,
    pub m: Array1<T>,
    pub inv_err: Array1<T>,
}

#[derive(Clone, Debug)]
pub struct NormalizedData<T> {
    pub data: Rc<Data<T>>,
    t_mean: T,
    t_std: T,
    m_mean: T,
    m_std: T,
    inv_err_scale: T,
}

impl<T> NormalizedData<T>
where
    T: Float,
{
    fn normalized<U>(ds: &mut DataSample<U>) -> (T, T, Array1<T>)
    where
        U: Float + conv::ApproxInto<T>,
    {
        let std = ds.get_std().approx_as::<T>().unwrap();
        if std.is_zero() {
            (
                ds.sample[0].approx_as::<T>().unwrap(),
                T::zero(),
                Array1::zeros(ds.sample.len()),
            )
        } else {
            let mean = ds.get_mean().approx_as::<T>().unwrap();
            let v = ds
                .sample
                .mapv(|x| (x.approx_as::<T>().unwrap() - mean) / std);
            (mean, std, v)
        }
    }

    pub fn from_ts<U>(ts: &mut TimeSeries<U>) -> Self
    where
        U: Float + conv::ApproxInto<T>,
    {
        let (t_mean, t_std, t) = Self::normalized(&mut ts.t);
        let (m_mean, m_std, m) = Self::normalized(&mut ts.m);
        let (inv_err_scale, inv_err) = if m_std.is_zero() {
            (
                T::one(),
                ts.w.sample.mapv(|x| x.approx_as::<T>().unwrap().sqrt()),
            )
        } else {
            let scale = m_std.recip();
            let inv_scale =
                ts.w.sample
                    .mapv(|x| x.approx_as::<T>().unwrap().sqrt() * m_std);
            (scale, inv_scale)
        };

        Self {
            data: Rc::new(Data { t, m, inv_err }),
            t_mean,
            t_std,
            m_mean,
            m_std,
            inv_err_scale,
        }
    }

    pub fn t_to_orig(&self, t_norm: T) -> T {
        t_norm * self.t_std + self.t_mean
    }

    pub fn t_to_orig_scale(&self, t_norm: T) -> T {
        t_norm * self.t_std
    }

    pub fn m_to_orig(&self, m_norm: T) -> T {
        m_norm * self.m_std + self.m_mean
    }

    pub fn m_to_orig_scale(&self, m_norm: T) -> T {
        m_norm * self.m_std
    }

    pub fn slope_to_orig(&self, slope_norm: T) -> T {
        if self.t_std.is_zero() || self.m_std.is_zero() {
            slope_norm
        } else {
            slope_norm * self.m_std / self.t_std
        }
    }

    #[allow(dead_code)]
    pub fn inv_err_to_orig(&self, inv_err_norm: T) -> T {
        inv_err_norm * self.inv_err_scale
    }

    pub fn t_to_norm(&self, t_orig: T) -> T {
        if self.t_std.is_zero() {
            T::zero()
        } else {
            (t_orig - self.t_mean) / self.t_std
        }
    }

    pub fn t_to_norm_scale(&self, t_orig: T) -> T {
        if self.t_std.is_zero() {
            t_orig
        } else {
            t_orig / self.t_std
        }
    }

    pub fn m_to_norm(&self, m_orig: T) -> T {
        if self.m_std.is_zero() {
            T::zero()
        } else {
            (m_orig - self.m_mean) / self.m_std
        }
    }

    pub fn m_to_norm_scale(&self, m_orig: T) -> T {
        if self.m_std.is_zero() {
            m_orig
        } else {
            m_orig / self.m_std
        }
    }

    #[allow(dead_code)]
    pub fn inv_err_to_norm(&self, inv_err_orig: T) -> T {
        inv_err_orig / self.inv_err_scale
    }

    pub fn slope_to_norm(&self, slope_orig: T) -> T {
        if self.t_std.is_zero() || self.m_std.is_zero() {
            slope_orig
        } else {
            slope_orig * self.t_std / self.m_std
        }
    }
}
