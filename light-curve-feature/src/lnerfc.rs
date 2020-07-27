/// Natural logarythm of the error function
/// Manually translated from GNU Scientific Library's log_erfc

#[allow(clippy::excessive_precision)]
const TEN_ROOT6_DBL_EPSILON: f64 = 2.4607833005759251e-02;
#[allow(clippy::excessive_precision)]
const SQRTPI: f64 = 1.77245385090551602729816748334;

pub fn ln_erfc(x: f64) -> f64 {
    if x.powi(2) < TEN_ROOT6_DBL_EPSILON {
        log_erfc_smallabs(x)
    } else if x > 8.0 {
        log_erfc8(x)
    } else {
        log_erfc_direct(x)
    }
}

fn sum_series(a: &[f64], x: f64) -> f64 {
    a.iter().fold(0.0, |acc, c| acc * x + c)
}

#[allow(clippy::excessive_precision)]
fn log_erfc_smallabs(x: f64) -> f64 {
    let y = x / SQRTPI;
    /* series for -1/2 Log[Erfc[Sqrt[Pi] y]] */
    const REVERSE_C: [f64; 15] = [
        0.00048204,
        -0.00142906,
        0.0013200243174,
        0.0009461589032,
        -0.0045563339802,
        0.00556964649138,
        0.00125993961762116,
        -0.01621575378835404, /* (96.0 - 40.0*M_PI + 3.0*M_PI*M_PI)/30.0  */
        0.02629651521057465,  /* 2.0*(120.0 - 60.0*M_PI + 7.0*M_PI*M_PI)/45.0 */
        -0.001829764677455021,
        2.0 * (1.0 - std::f64::consts::PI / 3.0),
        (4.0 - std::f64::consts::PI) / 3.0,
        1.0,
        1.0,
        0.0,
    ];
    -2.0 * sum_series(&REVERSE_C, y)
}

fn log_erfc8(x: f64) -> f64 {
    f64::ln(erfc8_sum(x)) - x.powi(2)
}

#[allow(clippy::excessive_precision)]
fn erfc8_sum(x: f64) -> f64 {
    const REVERSE_P: [f64; 6] = [
        0.5641895835477550741253201704,
        1.275366644729965952479585264,
        5.019049726784267463450058,
        6.1602098531096305440906,
        7.409740605964741794425,
        2.97886562639399288862,
    ];
    const REVERSE_Q: [f64; 7] = [
        1.0,
        2.260528520767326969591866945,
        9.396034016235054150430579648,
        12.0489519278551290360340491,
        17.08144074746600431571095,
        9.608965327192787870698,
        3.3690752069827527677,
    ];
    sum_series(&REVERSE_P, x) / sum_series(&REVERSE_Q, x)
}

fn log_erfc_direct(x: f64) -> f64 {
    f64::ln(libm::erfc(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_erfc_smallabs_argument() {
        let x = TEN_ROOT6_DBL_EPSILON;
        let from_series = log_erfc_smallabs(x);
        let from_direct = log_erfc_direct(x);
        assert!(
            f64::abs(1.0 - from_series / from_direct) < 1e-14,
            "from_series = {}, from_direct = {}",
            from_series,
            from_direct
        );
    }

    #[test]
    fn log_erfc_eight() {
        let x = 8.0;
        let from_series = log_erfc8(x);
        let from_direct = log_erfc_direct(x);
        assert!(
            f64::abs(1.0 - from_series / from_direct) < 1e-14,
            "from_series = {}, from_direct = {}",
            from_series,
            from_direct
        );
    }
}
