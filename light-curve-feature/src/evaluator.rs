use crate::error::EvaluatorError;
use crate::float_trait::Float;
use crate::time_series::TimeSeries;

use dyn_clonable::*;

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatorInfo {
    pub size: usize,
    pub min_ts_length: usize,
    pub t_required: bool,
    pub m_required: bool,
    pub w_required: bool,
    pub sorting_required: bool,
}

#[macro_export]
macro_rules! lazy_info {
    (
        $name: ident,
        size: $size: expr,
        min_ts_length: $len: expr,
        t_required: $t: expr,
        m_required: $m: expr,
        w_required: $w: expr,
        sorting_required: $sort: expr,
    ) => {
        lazy_static! {
            static ref $name: EvaluatorInfo = EvaluatorInfo {
                size: $size,
                min_ts_length: $len,
                t_required: $t,
                m_required: $m,
                w_required: $w,
                sorting_required: $sort,
            };
        }
    };
}

/// The trait each feature should implement
#[clonable]
pub trait FeatureEvaluator<T: Float>: Send + Sync + Clone {
    /// Should return the vector of feature values
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError>;

    fn eval_or_fill(&self, ts: &mut TimeSeries<T>, fill_value: T) -> Vec<T> {
        match self.eval(ts) {
            Ok(v) => v,
            Err(_) => vec![fill_value; self.size_hint()],
        }
    }

    /// Get feature evaluator meta-information
    fn get_info(&self) -> &EvaluatorInfo;

    /// Should return the vector of feature names. The length and feature order should
    /// correspond to `eval()` output
    fn get_names(&self) -> Vec<&str>;

    /// Should return the size of vectors returned by `eval()` and `get_names()`
    fn size_hint(&self) -> usize {
        self.get_info().size
    }

    /// Should return minimum time series length to successfully find feature value
    fn min_ts_length(&self) -> usize {
        self.get_info().min_ts_length
    }

    fn is_t_required(&self) -> bool {
        self.get_info().t_required
    }

    fn is_m_required(&self) -> bool {
        self.get_info().m_required
    }

    fn is_w_required(&self) -> bool {
        self.get_info().w_required
    }

    fn is_sorting_required(&self) -> bool {
        self.get_info().sorting_required
    }
}

pub type VecFE<T> = Vec<Box<dyn FeatureEvaluator<T>>>;

pub fn check_ts_length<FE, T>(
    feature_evaluator: &FE,
    ts: &TimeSeries<T>,
) -> Result<usize, EvaluatorError>
where
    T: Float,
    FE: FeatureEvaluator<T>,
{
    let length = ts.lenu();
    if length < feature_evaluator.size_hint() {
        Err(EvaluatorError::ShortTimeSeries {
            actual: length,
            minimum: feature_evaluator.size_hint(),
        })
    } else {
        Ok(length)
    }
}

pub fn get_nonzero_m_std<T: Float>(ts: &mut TimeSeries<T>) -> Result<T, EvaluatorError> {
    let std = ts.m.get_std();
    if std.is_zero() {
        Err(EvaluatorError::FlatTimeSeries)
    } else {
        Ok(std)
    }
}

pub fn get_nonzero_reduced_chi2<T: Float>(ts: &mut TimeSeries<T>) -> Result<T, EvaluatorError> {
    let reduced_chi2 = ts.get_m_reduced_chi2();
    if reduced_chi2.is_zero() {
        Err(EvaluatorError::FlatTimeSeries)
    } else {
        Ok(reduced_chi2)
    }
}

pub struct TMWVectors<T> {
    pub t: Vec<T>,
    pub m: Vec<T>,
    pub w: Option<Vec<T>>,
}

/// Helper for FeatureEvaluator implementations using time-series transformation.
/// You must implement:
/// - method `transform_ts(ts: &mut TimeSeries<T>) -> TMWVectors<T>`
/// - attribute `info: EvaluatorInfo`
/// - attribute `feature_names: Vec<String>`
#[macro_export]
macro_rules! transformer_eval {
    () => {
        fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
            let tmw = self.transform_ts(ts)?;
            let mut new_ts = TimeSeries::new(&tmw.t, &tmw.m, tmw.w.as_ref().map(|w| &w[..]));
            self.feature_extractor.eval(&mut new_ts)
        }

        fn eval_or_fill(&self, ts: &mut TimeSeries<T>, fill_value: T) -> Vec<T> {
            let tmw = match self.transform_ts(ts) {
                Ok(x) => x,
                Err(_) => return vec![fill_value; self.size_hint()],
            };
            let mut new_ts = TimeSeries::new(&tmw.t, &tmw.m, tmw.w.as_ref().map(|w| &w[..]));
            self.feature_extractor.eval_or_fill(&mut new_ts, fill_value)
        }

        fn get_info(&self) -> &EvaluatorInfo {
            &self.info
        }

        fn get_names(&self) -> Vec<&str> {
            self.feature_names
                .iter()
                .map(|name| name.as_str())
                .collect()
        }
    };
}
