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
