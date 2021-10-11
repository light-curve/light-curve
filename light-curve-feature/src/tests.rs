pub use crate::evaluator::FeatureEvaluator;
pub use crate::extractor::FeatureExtractor;
pub use crate::feature::Feature;
pub use crate::float_trait::Float;
pub use crate::time_series::TimeSeries;

pub use light_curve_common::{all_close, linspace};
pub use ndarray::{Array1, ArrayView1};
pub use rand::prelude::*;
pub use rand_distr::StandardNormal;

#[macro_export]
macro_rules! feature_test {
    ($name: ident, $fe: tt, $desired: expr, $y: expr $(,)?) => {
        feature_test!($name, $fe, $desired, $y, $y);
    };
    ($name: ident, $fe: tt, $desired: expr, $x: expr, $y: expr $(,)?) => {
        feature_test!($name, $fe, $desired, $x, $y, vec![1.0; $x.len()]);
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
            let w = $w;
            let mut ts = TimeSeries::new(&x, &y, &w);
            let actual = fe.eval(&mut ts).unwrap();
            all_close(&desired[..], &actual[..], $tol);

            let names = fe.get_names();
            let descs = fe.get_descriptions();
            assert_eq!(fe.size_hint(), actual.len(), "size_hint() returns wrong size");
            assert_eq!(actual.len(), names.len(),
                "Length of values and names should be the same");
            assert_eq!(actual.len(), descs.len(),
                "Length of values and descriptions should be the same");
        }
    };
}

#[macro_export]
macro_rules! eval_info_test {
    ($name: ident, $eval: expr $(,)?) => {
        #[test]
        fn $name() {
            eval_info_tests($eval.into());
        }
    };
}

pub fn eval_info_tests(eval: Feature<f64>) {
    const N: usize = 128;

    let mut rng = StdRng::seed_from_u64(0);

    let t = randvec::<f64>(&mut rng, N);
    let t_sorted = sorted(&t);
    assert_ne!(t, t_sorted);
    let m = randvec::<f64>(&mut rng, N);
    let w = positive_randvec::<f64>(&mut rng, N);

    let size_hint = FeatureEvaluator::<f64>::size_hint(&eval);
    assert_eq!(
        FeatureEvaluator::<f64>::get_names(&eval).len(),
        size_hint,
        "names vector has a wrong size"
    );
    assert_eq!(
        FeatureEvaluator::<f64>::get_descriptions(&eval).len(),
        size_hint,
        "description vector has a wrong size"
    );
    let check_size =
        |v: &Vec<f64>| assert_eq!(size_hint, v.len(), "size_hint() returns wrong value");

    let baseline = eval.eval(&mut TimeSeries::new(&t_sorted, &m, &w)).unwrap();
    check_size(&baseline);

    for n in 0..10 {
        eval_info_ts_length_test(&eval, &t_sorted, &m, &w, n)
            .as_ref()
            .map(check_size);
    }

    check_size(&eval_info_t_required_test(
        &eval, &baseline, &t_sorted, &m, &w, &mut rng,
    ));

    check_size(&eval_info_m_required_test(
        &eval, &baseline, &t_sorted, &m, &w, &mut rng,
    ));

    check_size(&eval_info_w_required_test(
        &eval, &baseline, &t_sorted, &m, &w, &mut rng,
    ));

    eval_info_sorting_required_test(&eval, &baseline, &t, &m, &w)
        .as_ref()
        .map(check_size);
}

fn eval_info_ts_length_test(
    eval: &Feature<f64>,
    t_sorted: &[f64],
    m: &[f64],
    w: &[f64],
    n: usize,
) -> Option<Vec<f64>> {
    let min_ts_length = FeatureEvaluator::<f64>::min_ts_length(eval);
    let mut ts = TimeSeries::new(&t_sorted[..n], &m[..n], &w[..n]);
    let result = eval.eval(&mut ts);
    assert_eq!(
        n >= min_ts_length,
        result.is_ok(),
        "min_ts_length() returns wrong value, \
                    time-series length: {}, \
                    min_ts_length(): {}, \
                    eval(ts).is_ok(): {}",
        n,
        min_ts_length,
        result.is_ok(),
    );
    result.ok()
}

fn eval_info_t_required_test(
    eval: &Feature<f64>,
    baseline: &[f64],
    t_sorted: &[f64],
    m: &[f64],
    w: &[f64],
    rng: &mut StdRng,
) -> Vec<f64> {
    let t2_sorted = sorted(&randvec::<f64>(rng, t_sorted.len()));
    assert_ne!(t_sorted, t2_sorted);

    let mut ts = TimeSeries::new(&t2_sorted, m, w);

    let v = eval.eval(&mut ts).unwrap();
    let neq_baseline = !simeq(&v, baseline, 1e-12);
    assert_eq!(
        neq_baseline,
        FeatureEvaluator::<f64>::is_t_required(eval),
        "is_t_required() returns wrong value, \
                    v != baseline: {} ({:?} <=> {:?}), \
                    is_t_required(): {}",
        neq_baseline,
        v,
        baseline,
        FeatureEvaluator::<f64>::is_t_required(eval),
    );
    v
}

fn eval_info_m_required_test(
    eval: &Feature<f64>,
    baseline: &[f64],
    t_sorted: &[f64],
    m: &[f64],
    w: &[f64],
    rng: &mut StdRng,
) -> Vec<f64> {
    let m2 = randvec::<f64>(rng, m.len());
    assert_ne!(m, m2);

    let mut ts = TimeSeries::new(t_sorted, m2, w);

    let v = eval.eval(&mut ts).unwrap();
    let neq_baseline = !simeq(&v, baseline, 1e-12);
    assert_eq!(
        neq_baseline,
        FeatureEvaluator::<f64>::is_m_required(eval),
        "is_m_required() returns wrong value, \
                    v != baseline: {} ({:?} <=> {:?}), \
                    is_m_required(): {}",
        neq_baseline,
        v,
        baseline,
        FeatureEvaluator::<f64>::is_m_required(eval),
    );
    v
}

fn eval_info_w_required_test(
    eval: &Feature<f64>,
    baseline: &[f64],
    t_sorted: &[f64],
    m: &[f64],
    w: &[f64],
    rng: &mut StdRng,
) -> Vec<f64> {
    let w2 = positive_randvec::<f64>(rng, w.len());
    assert_ne!(w, w2);

    let mut ts = TimeSeries::new(t_sorted, m, &w2);
    let v = eval.eval(&mut ts).unwrap();
    let neq_baseline = !simeq(&v, baseline, 1e-12);
    assert_eq!(
        neq_baseline,
        FeatureEvaluator::<f64>::is_w_required(eval),
        "is_w_required() returns wrong value, \
                    v != baseline: {}, \
                    is_w_required(): {}",
        neq_baseline,
        FeatureEvaluator::<f64>::is_w_required(eval),
    );
    v
}

fn eval_info_sorting_required_test(
    eval: &Feature<f64>,
    baseline: &[f64],
    t: &[f64],
    m: &[f64],
    w: &[f64],
) -> Option<Vec<f64>> {
    let m_ordered = sorted_by(m, t);
    assert_ne!(m_ordered, m);
    let w_ordered = sorted_by(w, t);
    assert_ne!(w_ordered, w);

    let is_sorting_required = FeatureEvaluator::<f64>::is_sorting_required(eval);

    // FeatureEvaluator is allowed to panic for unsorted input if it requires sorted input
    let v = match (
        std::panic::catch_unwind(|| eval.eval(&mut TimeSeries::new(t, &m_ordered, &w_ordered))),
        is_sorting_required,
    ) {
        (Ok(result), _) => result.unwrap(),
        (Err(_), true) => return None,
        (Err(err), false) => panic!("{:?}", err),
    };

    let neq_baseline = !simeq(&v, baseline, 1e-12);
    assert_eq!(
        neq_baseline, is_sorting_required,
        "is_sorting_required() returns wrong value, \
                    unsorted result: {:?}, \
                    sorted result: {:?}, \
                    is_sorting_required: {}",
        v, baseline, is_sorting_required,
    );
    Some(v)
}

#[macro_export]
macro_rules! serialization_name_test {
    ($feature: ty) => {
        #[test]
        fn serialization_name() {
            let feature = <$feature>::default();
            let actual_name = serde_type_name::type_name(&feature).unwrap();

            let str_type = stringify!($feature);
            let desired_name = match str_type.split_once('<') {
                Some((name, _)) => name,
                None => str_type,
            };

            assert_eq!(actual_name, desired_name);
        }
    };
}

#[macro_export]
macro_rules! serde_json_test {
    ($name: ident, $feature_type: ty, $feature_expr: expr $(,)?) => {
        #[test]
        fn $name() {
            const N: usize = 128;
            let mut rng = StdRng::seed_from_u64(0);

            let t = sorted(&randvec::<f64>(&mut rng, N));
            let m = randvec::<f64>(&mut rng, N);
            let w = positive_randvec::<f64>(&mut rng, N);

            let eval = $feature_expr;
            let eval_serde: $feature_type =
                serde_json::from_str(&serde_json::to_string(&eval).unwrap()).unwrap();
            assert_eq!(
                eval.eval(&mut TimeSeries::new(&t, &m, &w)),
                eval_serde.eval(&mut TimeSeries::new(&t, &m, &w))
            );

            let feature: Feature<_> = eval.into();
            let feature_serde: Feature<_> =
                serde_json::from_str(&serde_json::to_string(&feature).unwrap()).unwrap();
            assert_eq!(
                feature.eval(&mut TimeSeries::new(&t, &m, &w)),
                feature_serde.eval(&mut TimeSeries::new(&t, &m, &w))
            );
        }
    };
}

#[macro_export]
macro_rules! check_doc_static_method {
    ($name: ident, $feature: ty) => {
        #[test]
        fn $name() {
            let doc = <$feature>::doc();
            assert!(doc.contains("Depends on: "));
            assert!(doc.contains("Minimum number of observations: "));
            assert!(doc.contains("Number of features: "));
        }
    };
}

#[macro_export]
macro_rules! check_feature {
    ($feature: ty) => {
        eval_info_test!(info_default, <$feature>::default());
        serialization_name_test!($feature);
        serde_json_test!(ser_json_de, $feature, <$feature>::default());
        check_doc_static_method!(doc_static_method, $feature);
    };
}

pub fn simeq<T: Float>(a: &[T], b: &[T], eps: T) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .all(|(&x, &y)| (x - y).abs() < eps + T::max(x.abs(), y.abs()) * eps)
}

pub fn randvec<T>(rng: &mut StdRng, n: usize) -> Vec<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    (0..n)
        .map(|_| {
            let x: T = rng.sample(StandardNormal);
            x
        })
        .collect()
}

pub fn positive_randvec<T>(rng: &mut StdRng, n: usize) -> Vec<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    let mut v = randvec(rng, n);
    v.iter_mut().for_each(|x| *x = x.abs());
    v
}

pub fn sorted<T>(v: &[T]) -> Vec<T>
where
    T: Float,
{
    let mut v = v.to_vec();
    v[..].sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

pub fn sorted_by<T: Float>(to_sort: &[T], key: &[T]) -> Vec<T> {
    assert_eq!(to_sort.len(), key.len());
    let mut idx: Vec<_> = (0..to_sort.len()).collect();
    idx[..].sort_by(|&a, &b| key[a].partial_cmp(&key[b]).unwrap());
    idx.iter().map(|&i| to_sort[i]).collect()
}
