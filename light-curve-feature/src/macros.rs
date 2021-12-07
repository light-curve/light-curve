/// Helper for static EvaluatorInfo creation
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
    (
        $name: ident,
        $feature: ty,
        size: $size: expr,
        min_ts_length: $len: expr,
        t_required: $t: expr,
        m_required: $m: expr,
        w_required: $w: expr,
        sorting_required: $sort: expr,
    ) => {
        lazy_info!(
            $name,
            size: $size,
            min_ts_length: $len,
            t_required: $t,
            m_required: $m,
            w_required: $w,
            sorting_required: $sort,
        );

        impl EvaluatorInfoTrait for $feature {
            fn get_info(&self) -> &EvaluatorInfo {
                &$name
            }
        }
    };
    (
        $name: ident,
        $feature: ty,
        T,
        size: $size: expr,
        min_ts_length: $len: expr,
        t_required: $t: expr,
        m_required: $m: expr,
        w_required: $w: expr,
        sorting_required: $sort: expr,
    ) => {
        lazy_info!(
            $name,
            size: $size,
            min_ts_length: $len,
            t_required: $t,
            m_required: $m,
            w_required: $w,
            sorting_required: $sort,
        );

        impl<T: Float> EvaluatorInfoTrait for $feature {
            fn get_info(&self) -> &EvaluatorInfo {
                &$name
            }
        }
    };
}

/// Helper for FeatureEvaluator implementations using time-series transformation.
/// You must implement:
/// - `transform_ts(&self, ts: &mut TimeSeries<T>) -> Result<impl OwnedArrays<T>, EvaluatorError>`
macro_rules! transformer_eval {
    () => {
        fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
            let arrays = self.transform_ts(ts)?;
            let mut new_ts = arrays.ts();
            self.feature_extractor.eval(&mut new_ts)
        }

        fn eval_or_fill(&self, ts: &mut TimeSeries<T>, fill_value: T) -> Vec<T> {
            let arrays = match self.transform_ts(ts) {
                Ok(x) => x,
                Err(_) => return vec![fill_value; self.size_hint()],
            };
            let mut new_ts = arrays.ts();
            self.feature_extractor.eval_or_fill(&mut new_ts, fill_value)
        }
    };
}

/// Helper implementing JsonSchema crate
macro_rules! json_schema {
    ($parameters: ty, $is_referenceable: expr) => {
        fn is_referenceable() -> bool {
            $is_referenceable
        }

        fn schema_name() -> String {
            <$parameters>::schema_name()
        }

        fn json_schema(gen: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
            <$parameters>::json_schema(gen)
        }
    };
}
