#[cfg(test)]
#[macro_export]
macro_rules! feature_test {
    ($name: ident, $fe: tt, $desired: expr, $y: expr $(,)?) => {
        feature_test!($name, $fe, $desired, $y, $y);
    };
    ($name: ident, $fe: tt, $desired: expr, $x: expr, $y: expr $(,)?) => {
        feature_test!($name, $fe, $desired, $x, $y, None);
    };
    ($name: ident, $fe: tt, $desired: expr, $x: expr, $y: expr, $w: expr $(,)?) => {
        feature_test!($name, $fe, $desired, $x, $y, $w, 1e-6);
    };
    ($name: ident, $fe: tt, $desired: expr, $x: expr, $y: expr, $w: expr, $tol: expr $(,)?) => {
        #[test]
        fn $name() {
            let fe = FeatureExtractor::new(vec!$fe);
            let desired = $desired;
            let x = $x;
            let y = $y;
            let mut ts = TimeSeries::new(&x[..], &y[..], $w);
            let actual = fe.eval(&mut ts).unwrap();
            all_close(&desired[..], &actual[..], $tol);

            let names = fe.get_names();
            assert_eq!(fe.size_hint(), actual.len(), "size_hint() returns wrong size");
            assert_eq!(actual.len(), names.len(),
                "Length of values and names should be the same");
        }
    };
}

/// Helper for static EvaluatorInfo creation
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

/// Constructs a `FeatureExtractor` object from a list of objects that implement `FeatureEvaluator`
/// ```
/// use light_curve_feature::*;
///
/// let fe = feat_extr!(BeyondNStd::new(1.0), Cusum::default());
/// ```
#[macro_export]
macro_rules! feat_extr{
    ( $( $x: expr ),* $(,)? ) => {
        FeatureExtractor::new(
            vec![$(
                Box::new($x),
            )*]
        )
    }
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
