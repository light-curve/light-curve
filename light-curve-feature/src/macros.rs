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

/// Helper implemnting *Fit feature evaluators
/// You must:
/// - implement all traits of [nl_fit::evaluator]
/// - satisfy all [FeatureEvaluator] trait constraints
/// - declare `const NPARAMS: usize` in your code
macro_rules! fit_eval {
    () => {
        fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
            self.check_ts_length(ts)?;

            let norm_data = NormalizedData::<f64>::from_ts(ts);

            let (x0, lower, upper) = {
                let FitInitsBounds {
                    init: mut x0,
                    mut lower,
                    mut upper,
                } = Self::init_and_bounds_from_ts(ts);
                x0 = Self::convert_to_internal(&norm_data, &x0);
                lower = Self::convert_to_internal(&norm_data, &lower);
                upper = Self::convert_to_internal(&norm_data, &upper);
                (x0, lower, upper)
            };

            let result = {
                let norm_data_for_prior = norm_data.clone();
                let CurveFitResult {
                    x, reduced_chi2, ..
                } = self.get_algorithm().curve_fit(
                    norm_data.data.clone(),
                    &x0,
                    (&lower, &upper),
                    Self::model,
                    Self::derivatives,
                    self.get_ln_prior()
                        .as_func_with_transformation(move |params| {
                            Self::convert_to_external(&norm_data_for_prior, params)
                        }),
                );
                let result =
                    Self::convert_to_external(&norm_data, (&x as &[_]).try_into().unwrap());
                result
                    .into_iter()
                    .chain(std::iter::once(reduced_chi2))
                    .map(|x| x.approx_as::<T>().unwrap())
                    .collect()
            };

            Ok(result)
        }
    };
}
