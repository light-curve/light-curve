use criterion::{black_box, Criterion};
use light_curve_dmdt::{ErfEps1Over1e3Float, Float, LibMFloat};

pub trait BenchFloat: Sized {
    const X_FOR_ERF: [Self; 13];
    const X_FOR_ERFINV: [Self; 13];
}

impl BenchFloat for f32 {
    const X_FOR_ERF: [Self; 13] = [
        -10.0, -3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 10.0,
    ];
    const X_FOR_ERFINV: [Self; 13] = [
        -0.999, -0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 0.999,
    ];
}

impl BenchFloat for f64 {
    const X_FOR_ERF: [Self; 13] = [
        -10.0, -3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 10.0,
    ];
    const X_FOR_ERFINV: [Self; 13] = [
        -0.999, -0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 0.999,
    ];
}

pub trait MathruFloat:
    mathru::algebra::abstr::Real + mathru::special::error::Error + mathru::special::gamma::Gamma
{
}

impl MathruFloat for f32 {}
impl MathruFloat for f64 {}

pub fn bench_erf<T>(c: &mut Criterion)
where
    T: BenchFloat + ErfEps1Over1e3Float + LibMFloat + MathruFloat + special::Error + Float,
{
    c.bench_function(
        format!("erf_eps_1over1e3 for {}", std::any::type_name::<T>()).as_str(),
        |b| {
            b.iter(|| {
                T::X_FOR_ERF.iter().for_each(|&x| {
                    black_box(x.erf_eps_1over1e3());
                })
            })
        },
    );
    c.bench_function(
        format!("libm::erf(f) for {}", std::any::type_name::<T>()).as_str(),
        |b| {
            b.iter(|| {
                T::X_FOR_ERF.iter().for_each(|&x| {
                    black_box(x.libm_erf());
                })
            })
        },
    );
    c.bench_function(
        format!(
            "mathru::special::error::erf for {}",
            std::any::type_name::<T>()
        )
        .as_str(),
        |b| {
            b.iter(|| {
                T::X_FOR_ERF.iter().for_each(|&x| {
                    black_box(mathru::special::error::erf(x));
                })
            })
        },
    );
    c.bench_function(
        format!("special::Error::error for {}", std::any::type_name::<T>()).as_str(),
        |b| {
            b.iter(|| {
                T::X_FOR_ERF.iter().for_each(|&x| {
                    black_box(x.error());
                })
            })
        },
    );
}

pub fn bench_erfinv<T>(c: &mut Criterion)
where
    T: BenchFloat + MathruFloat + special::Error,
{
    c.bench_function(
        format!(
            "mathru::special::error::erfinv for {}",
            std::any::type_name::<T>()
        )
        .as_str(),
        |b| {
            b.iter(|| {
                T::X_FOR_ERFINV.iter().for_each(|&x| {
                    black_box(mathru::special::error::erfinv(x));
                })
            })
        },
    );
    c.bench_function(
        format!(
            "special::Error::inv_error for {}",
            std::any::type_name::<T>()
        )
        .as_str(),
        |b| {
            b.iter(|| {
                T::X_FOR_ERFINV.iter().for_each(|&x| {
                    black_box(x.inv_error());
                })
            })
        },
    );
}
