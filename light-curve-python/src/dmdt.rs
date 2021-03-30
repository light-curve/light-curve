use crate::arr_wrapper::ArrWrapper;
use crate::errors::{Exception, Res};
use crate::sorted::check_sorted;
use conv::ConvUtil;
use enumflags2::{bitflags, BitFlags};
use light_curve_dmdt as lcdmdt;
use ndarray::IntoNdProducer;
use numpy::{Element, IntoPyArray, PyArray1};
use pyo3::class::iter::PyIterProtocol;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::cell::RefCell;
use std::ops::{DerefMut, Range};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use unzip3::Unzip3;

#[derive(FromPyObject)]
enum GenericFloatArray1<'a> {
    #[pyo3(transparent, annotation = "np.ndarray[float32]")]
    Float32(&'a PyArray1<f32>),
    #[pyo3(transparent, annotation = "np.ndarray[float64]")]
    Float64(&'a PyArray1<f64>),
}

#[derive(FromPyObject, Copy, Clone, std::fmt::Debug)]
enum DropNObsType {
    #[pyo3(transparent, annotation = "int")]
    Int(usize),
    #[pyo3(transparent, annotation = "float")]
    Float(f64),
}

#[bitflags]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
enum NormFlag {
    LgDt,
    Max,
}

#[derive(Clone)]
struct GenericDmDt<T> {
    dmdt: lcdmdt::DmDt<T>,
    norm: BitFlags<NormFlag>,
    error_func: lcdmdt::ErrorFunction,
    n_jobs: usize,
}

impl<T> GenericDmDt<T>
where
    T: Element + ndarray::NdFloat + lcdmdt::ErfFloat,
{
    fn sigma_to_err2(sigma: &PyArray1<T>) -> ndarray::Array1<T> {
        let mut a = sigma.to_owned_array();
        a.mapv_inplace(|x| x.powi(2));
        a
    }

    fn normalize(&self, a: &mut ndarray::Array2<T>, t: &[T]) {
        if self.norm.contains(NormFlag::LgDt) {
            let lgdt = self.dmdt.lgdt_points(&t);
            let lgdt_no_zeros = lgdt.mapv(|x| {
                if x == 0 {
                    T::one()
                } else {
                    x.value_as::<T>().unwrap()
                }
            });
            *a /= &lgdt_no_zeros.into_shape((a.nrows(), 1)).unwrap();
        }
        if self.norm.contains(NormFlag::Max) {
            let max = *a.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            if !max.is_zero() {
                a.mapv_inplace(|x| x / max);
            }
        }
    }

    fn count_lgdt(&self, t: &[T], sorted: Option<bool>) -> Res<ndarray::Array1<T>> {
        check_sorted(t, sorted)?;
        Ok(self
            .dmdt
            .lgdt_points(t)
            .mapv(|x| x.value_as::<T>().unwrap()))
    }

    fn points(&self, t: &[T], m: &[T], sorted: Option<bool>) -> Res<ndarray::Array2<T>> {
        check_sorted(t, sorted)?;

        let mut result = self.dmdt.points(t, m).mapv(|x| x.value_as::<T>().unwrap());
        self.normalize(&mut result, &t);
        Ok(result)
    }

    fn points_many(&self, lcs: Vec<(&[T], &[T])>, sorted: Option<bool>) -> Res<ndarray::Array3<T>> {
        let dmdt_shape = self.dmdt.shape();
        let mut result = ndarray::Array3::zeros((lcs.len(), dmdt_shape.0, dmdt_shape.1));

        rayon::ThreadPoolBuilder::new()
            .num_threads(self.n_jobs)
            .build()
            .unwrap()
            .install(|| {
                ndarray::Zip::from(result.outer_iter_mut())
                    .and(lcs.into_producer())
                    .into_par_iter()
                    .try_for_each::<_, Res<_>>(|(mut map, (t, m))| {
                        map.assign(&self.points(t, m, sorted)?);
                        Ok(())
                    })
            })?;
        Ok(result)
    }

    fn points_from_columnar(
        &self,
        edges: &PyArray1<i64>,
        t: &PyArray1<T>,
        m: &PyArray1<T>,
        sorted: Option<bool>,
    ) -> Res<ndarray::Array3<T>> {
        let edges = &ArrWrapper::new(edges, true)[..];
        let t = &ArrWrapper::new(t, true)[..];
        let m = &ArrWrapper::new(m, true)[..];

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.n_jobs)
            .build()
            .unwrap();

        let maps = pool.install(|| {
            edges
                .par_windows(2)
                .map(|idx| {
                    let (first, last) = (idx[0] as usize, idx[1] as usize);
                    let t_lc = &t[first..last];
                    check_sorted(t_lc, sorted)?;
                    let m_lc = &m[first..last];

                    let mut a = self
                        .dmdt
                        .points(t_lc, m_lc)
                        .mapv(|x| x.value_as::<T>().unwrap());
                    self.normalize(&mut a, t_lc);

                    let (nrows, ncols) = (a.nrows(), a.ncols());
                    Ok(a.into_shape((1, nrows, ncols)).unwrap())
                })
                .collect::<Res<Vec<_>>>()
        })?;
        let map_views: Vec<_> = maps.iter().map(|a| a.view()).collect();
        Ok(ndarray::concatenate(ndarray::Axis(0), &map_views).unwrap())
    }

    fn gausses(
        &self,
        t: &[T],
        m: &[T],
        err2: &[T],
        sorted: Option<bool>,
    ) -> Res<ndarray::Array2<T>> {
        check_sorted(t, sorted)?;

        let mut result = self.dmdt.gausses(&t, &m, err2, &self.error_func);
        self.normalize(&mut result, &t);
        Ok(result)
    }

    fn gausses_many(
        &self,
        lcs: Vec<(&[T], &[T], &[T])>,
        sorted: Option<bool>,
    ) -> Res<ndarray::Array3<T>> {
        let dmdt_shape = self.dmdt.shape();
        let mut result = ndarray::Array3::zeros((lcs.len(), dmdt_shape.0, dmdt_shape.1));

        rayon::ThreadPoolBuilder::new()
            .num_threads(self.n_jobs)
            .build()
            .unwrap()
            .install(|| {
                ndarray::Zip::from(result.outer_iter_mut())
                    .and(lcs.into_producer())
                    .into_par_iter()
                    .try_for_each::<_, Res<_>>(|(mut map, (t, m, err2))| {
                        map.assign(&self.gausses(t, m, err2, sorted)?);
                        Ok(())
                    })
            })?;
        Ok(result)
    }

    fn gausses_from_columnar(
        &self,
        edges: &PyArray1<i64>,
        t: &PyArray1<T>,
        m: &PyArray1<T>,
        sigma: &PyArray1<T>,
        sorted: Option<bool>,
    ) -> Res<ndarray::Array3<T>> {
        let edges = &ArrWrapper::new(edges, true)[..];
        let t = &ArrWrapper::new(t, true)[..];
        let m = &ArrWrapper::new(m, true)[..];
        let sigma = &ArrWrapper::new(sigma, true)[..];

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.n_jobs)
            .build()
            .unwrap();

        let maps = pool.install(|| {
            edges
                .par_windows(2)
                .map(|idx| {
                    let (first, last) = (idx[0] as usize, idx[1] as usize);
                    let t_lc = &t[first..last];
                    check_sorted(t_lc, sorted)?;
                    let m_lc = &m[first..last];
                    let err2_lc: Vec<_> = sigma[first..last].iter().map(|x| x.powi(2)).collect();

                    let mut a = self.dmdt.gausses(t_lc, m_lc, &err2_lc, &self.error_func);
                    self.normalize(&mut a, t_lc);

                    let (nrows, ncols) = (a.nrows(), a.ncols());
                    Ok(a.into_shape((1, nrows, ncols)).unwrap())
                })
                .collect::<Res<Vec<_>>>()
        })?;
        let map_views: Vec<_> = maps.iter().map(|a| a.view()).collect();
        Ok(ndarray::concatenate(ndarray::Axis(0), &map_views).unwrap())
    }
}

struct GenericDmDtBatches<T, LC> {
    dmdt: GenericDmDt<T>,
    lcs: Vec<LC>,
    batch_size: usize,
    yield_index: bool,
    shuffle: bool,
    drop_nobs: Option<DropNObsType>,
    rng: Mutex<Xoshiro256PlusPlus>,
}

impl<T, LC> GenericDmDtBatches<T, LC> {
    fn new(
        dmdt: GenericDmDt<T>,
        lcs: Vec<LC>,
        batch_size: usize,
        yield_index: bool,
        shuffle: bool,
        drop_nobs: DropNObsType,
        random_seed: Option<u64>,
    ) -> Res<Self> {
        let rng = match random_seed {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_rng(&mut rand::thread_rng()).unwrap(),
        };
        let drop_nobs = match drop_nobs {
            DropNObsType::Int(0) => None,
            DropNObsType::Int(_) => Some(drop_nobs),
            DropNObsType::Float(x) => match x {
                _ if x == 0.0 => None,
                _ if (0.0..1.0).contains(&x) => Some(drop_nobs),
                _ => {
                    return Err(Exception::ValueError(String::from(
                        "if drop_nobs is float, it must be in [0.0, 1.0)",
                    )))
                }
            },
        };
        Ok(Self {
            dmdt,
            lcs,
            batch_size,
            yield_index,
            shuffle,
            drop_nobs,
            rng: Mutex::new(rng),
        })
    }

    fn dropped_index<R: rand::Rng>(&self, rng: &mut R, length: usize) -> Res<Vec<usize>> {
        let drop_nobs = match self.drop_nobs {
            Some(drop_nobs) => drop_nobs,
            None => {
                return Err(Exception::RuntimeError(String::from(
                    "dropping is not required: drop_nobs = 0",
                )))
            }
        };
        let drop_nobs = match drop_nobs {
            DropNObsType::Int(x) => x,
            DropNObsType::Float(x) => f64::round(x * length as f64) as usize,
        };
        if drop_nobs >= length {
            return Err(Exception::ValueError(format!(
                "cannot drop {} observations from light curve containing {} points",
                drop_nobs, length
            )));
        }
        if drop_nobs == 0 {
            return Ok((0..length).collect());
        }
        let mut idx = rand::seq::index::sample(rng, length, length - drop_nobs).into_vec();
        idx.sort_unstable();
        Ok(idx)
    }
}

macro_rules! dmdt_batches {
    ($worker: expr, $generic: ty, $name_batches: ident, $name_iter: ident, $t: ty $(,)?) => {
        #[pyclass]
        struct $name_batches {
            dmdt_batches: Arc<$generic>,
        }

        #[pyproto]
        impl PyIterProtocol for $name_batches {
            fn __iter__(slf: PyRef<Self>) -> PyResult<Py<$name_iter>> {
                let iter = $name_iter::new(slf.dmdt_batches.clone());
                Py::new(slf.py(), iter)
            }
        }

        #[pyclass]
        struct $name_iter {
            dmdt_batches: Arc<$generic>,
            lcs_order: Vec<usize>,
            range: Range<usize>,
            worker_thread: RefCell<Option<JoinHandle<Res<ndarray::Array3<$t>>>>>,
            rng: Option<Xoshiro256PlusPlus>,
        }

        impl $name_iter {
            fn child_rng(rng: Option<&mut Xoshiro256PlusPlus>) -> Option<Xoshiro256PlusPlus> {
                rng.map(|rng| Xoshiro256PlusPlus::from_rng(rng).unwrap())
            }

            fn new(dmdt_batches: Arc<$generic>) -> Self {
                let range = 0..usize::min(dmdt_batches.batch_size, dmdt_batches.lcs.len());

                let lcs_order = match dmdt_batches.shuffle {
                    false => (0..dmdt_batches.lcs.len()).collect(),
                    true => rand::seq::index::sample(
                        dmdt_batches.rng.lock().unwrap().deref_mut(),
                        dmdt_batches.lcs.len(),
                        dmdt_batches.lcs.len(),
                    )
                    .into_vec(),
                };

                let mut rng = match dmdt_batches.drop_nobs {
                    Some(_) => {
                        let mut parent_rng = dmdt_batches.rng.lock().unwrap();
                        Self::child_rng(Some(parent_rng.deref_mut()))
                    }
                    None => None,
                };

                let worker_thread = RefCell::new(Some(Self::run_worker_thread(
                    &dmdt_batches,
                    &lcs_order[range.start..range.end],
                    Self::child_rng(rng.as_mut()),
                )));

                Self {
                    dmdt_batches,
                    lcs_order,
                    range,
                    worker_thread,
                    rng,
                }
            }

            fn run_worker_thread(
                dmdt_batches: &Arc<$generic>,
                indexes: &[usize],
                rng: Option<Xoshiro256PlusPlus>,
            ) -> JoinHandle<Res<ndarray::Array3<$t>>> {
                let dmdt_batches = dmdt_batches.clone();
                let indexes = indexes.to_vec();
                std::thread::spawn(move || Self::worker(dmdt_batches, &indexes, rng))
            }

            fn worker(
                dmdt_batches: Arc<$generic>,
                indexes: &[usize],
                rng: Option<Xoshiro256PlusPlus>,
            ) -> Res<ndarray::Array3<$t>> {
                $worker(dmdt_batches, indexes, rng)
            }
        }

        impl Drop for $name_iter {
            fn drop(&mut self) {
                let t = self.worker_thread.replace(None);
                t.into_iter().for_each(|t| drop(t.join().unwrap()));
            }
        }

        #[pyproto]
        impl PyIterProtocol for $name_iter {
            fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
                slf
            }

            fn __next__(mut slf: PyRefMut<Self>) -> Res<Option<PyObject>> {
                if slf.worker_thread.borrow().is_none() {
                    return Ok(None);
                }

                let current_range = match slf.dmdt_batches.yield_index {
                    true => Some(slf.range.clone()),
                    false => None,
                };

                slf.range = slf.range.end
                    ..usize::min(
                        slf.range.end + slf.dmdt_batches.batch_size,
                        slf.dmdt_batches.lcs.len(),
                    );

                // Replace with None temporary to not have two workers running simultaneously
                let array = slf.worker_thread.replace(None).unwrap().join().unwrap()?;
                if !slf.range.is_empty() {
                    let rng = Self::child_rng(slf.rng.as_mut());
                    slf.worker_thread.replace(Some(Self::run_worker_thread(
                        &slf.dmdt_batches,
                        &slf.lcs_order[slf.range.start..slf.range.end],
                        rng,
                    )));
                }

                let py_array = array.into_pyarray(slf.py()).into_py(slf.py());
                match current_range {
                    Some(range) => {
                        let py_index =
                            PyArray1::from_slice(slf.py(), &slf.lcs_order[range.start..range.end])
                                .into_py(slf.py());
                        let tuple = PyTuple::new(slf.py(), &[py_index, py_array]).into_py(slf.py());
                        Ok(Some(tuple))
                    }
                    None => Ok(Some(py_array)),
                }
            }
        }
    };
}

macro_rules! py_dmdt_batches {
    ($worker: expr, $($generic: ty, $name_batches: ident, $name_iter: ident, $t: ty),* $(,)?) => {
        $(
            dmdt_batches!($worker, $generic, $name_batches, $name_iter, $t);
        )*
    };
}

py_dmdt_batches!(
    |dmdt_batches: Arc<GenericDmDtBatches<_, (ndarray::Array1<_>, ndarray::Array1<_>)>>, indexes: &[usize], rng: Option<Xoshiro256PlusPlus>| {
        let mut lcs: Vec<_> = indexes
            .iter()
            .map(|&i| {
                let (t, m) = &dmdt_batches.lcs[i];
                (
                    t.as_slice_memory_order().unwrap(),
                    m.as_slice_memory_order().unwrap(),
                )
            })
            .collect();
        let dropped_owned_lcs = match (dmdt_batches.drop_nobs, rng) {
            (Some(_), Some(mut rng)) => {
                let owned_lcs: Vec<(Vec<_>, Vec<_>)> = lcs.iter().map(|(t, m)| {
                    Ok(dmdt_batches.dropped_index(&mut rng, t.len())?.iter().map(|&i| (t[i], m[i])).unzip())
                }).collect::<Res<_>>()?;
                Some(owned_lcs)
            },
            (None, None) => None,
            (_, _) => return Err(Exception::RuntimeError(String::from("cannot be here, please report an issue")))
        };
        if let Some(owned_lcs) = &dropped_owned_lcs {
            for (ref_lc, lc) in lcs.iter_mut().zip(owned_lcs) {
                *ref_lc = (&lc.0, &lc.1);
            }
        }
        Ok(dmdt_batches.dmdt.points_many(lcs, Some(true))?)
    },
    GenericDmDtBatches<f32, (ndarray::Array1<f32>, ndarray::Array1<f32>)>,
    DmDtPointsBatchesF32,
    DmDtPointsIterF32,
    f32,
    GenericDmDtBatches<f64, (ndarray::Array1<f64>, ndarray::Array1<f64>)>,
    DmDtPointsBatchesF64,
    DmDtPointsIterF64,
    f64,
);

py_dmdt_batches!(
    |dmdt_batches: Arc<GenericDmDtBatches<_, (ndarray::Array1<_>, ndarray::Array1<_>, ndarray::Array1<_>)>>, indexes: &[usize], rng: Option<Xoshiro256PlusPlus>| {
        let mut lcs: Vec<_> = indexes
            .iter()
            .map(|&i| {
                let (t, m, err2) = &dmdt_batches.lcs[i];
                (
                    t.as_slice_memory_order().unwrap(),
                    m.as_slice_memory_order().unwrap(),
                    err2.as_slice_memory_order().unwrap(),
                )
            })
            .collect();
        let dropped_owned_lcs = match (dmdt_batches.drop_nobs, rng) {
            (Some(_), Some(mut rng)) => {
                let owned_lcs: Vec<(Vec<_>, Vec<_>, Vec<_>)> = lcs.iter().map(|(t, m, err2)| {
                    Ok(dmdt_batches.dropped_index(&mut rng, t.len())?.iter().map(|&i| (t[i], m[i], err2[i])).unzip3())
                }).collect::<Res<_>>()?;
                Some(owned_lcs)
            },
            (None, None) => None,
            (_, _) => return Err(Exception::RuntimeError(String::from("cannot be here, please report an issue")))
        };
        if let Some(owned_lcs) = &dropped_owned_lcs {
            for (ref_lc, lc) in lcs.iter_mut().zip(owned_lcs) {
                *ref_lc = (&lc.0, &lc.1, &lc.2);
            }
        }
        Ok(dmdt_batches.dmdt.gausses_many(lcs, Some(true))?)
    },
    GenericDmDtBatches<f32, (ndarray::Array1<f32>, ndarray::Array1<f32>, ndarray::Array1<f32>)>,
    DmDtGaussesBatchesF32,
    DmDtGaussesIterF32,
    f32,
    GenericDmDtBatches<f64, (ndarray::Array1<f64>, ndarray::Array1<f64>, ndarray::Array1<f64>)>,
    DmDtGaussesBatchesF64,
    DmDtGaussesIterF64,
    f64,
);

/// dm-lg(dt) map producer
///
/// Each pair of observations is mapped to dm-lg(dt) plane bringing unity
/// value. dmdt-map is a rectangle on this plane consisted of
/// `lgdt_size` x `dm_size` cells, and limited by `[min_lgdt; max_lgdt)` and
/// `[-max_abs_dm; max_abs_dm)` intervals. `.points*()` methods assigns unity
/// value of each observation to a single cell, while `.gausses*()` methods
/// smears this unity value over all cells with given lg(t2 - t1) value using
/// normal distribution `N(m2 - m1, sigma1^2 + sigma2^2)`, where
/// `(t1, m1, sigma1)` and `(t2, m2, sigma2)` is a pair of observations
/// including uncertainties. Optionally after the map is built, normalisation
/// is performed ("norm" parameter): "lgdt" means divide each lg(dt) = const
/// column by the total number of all observations corresponded to given dt;
/// "max" means divide all values by the maximum value; both options can be
/// combined, then "max" is performed after "lgdt".
///
/// Parameters
/// ----------
/// min_lgdt : float
///     Left border of lg(dt) interval
/// max_lgdt : float
///     Right border of lg(dt) interval
/// max_abs_dm : float
///     Absolute values of dm interval borders
/// lgdt_size : int
///     Number of lg(dt) cells
/// dm_size : int
///     Number of dm cells
/// norm : list of str, opional
///     Types of normalisation, cab be any combination of "lgdt" and "max",
///     default is an empty list `[]` which means no normalisation
/// n_jobs : int, optional
///     Number of parallel threads to run in bulk transformation methods such
///     as `points_many()`, `gausses_batches()` or `points_from_columnar()`,
///     default is `-1` which means to use as many threads as CPU cores
/// approx_erf : bool, optional
///     Use approximation normal CDF in `gausses*` methods, reduces accuracy,
///     but has better performance, default is `False`
///
/// Attributes
/// ----------
/// n_jobs : int
/// shape : (int, int)
///     Shape of a single dmdt map, `(lgdt_size, dm_size)`
/// min_lgdt : float
/// max_lgdt : float
/// max_abs_dm : float
///
/// Methods
/// -------
/// points(t, m, sorted=None)
///     Produces dmdt-maps from light curve
/// gausses(t, m, sigma, sorted=None)
///     Produces smeared dmdt-map from noisy light curve
/// points_many(lcs, sorted=None)
///     Produces dmdt-maps from a list of light curves
/// gausses_many(lcs, sorted=None)
///     Produces smeared dmdt-maps from a list of light curves
/// points_batches(lcs, sorted=None, batch_size=1, yield_index=False, shuffle=False, drop_nobs=0, random_seed=None)
///     Gives a reusable iterable which yields dmdt-maps
/// gausses_batches(lcs, sorted=None, batch_size=1, yield_index=False, shuffle=False, drop_nobs=0, random_seed=None)
///     Gives a reusable iterable which yields smeared dmdt-maps
/// points_from_columnar(edges, t, m, sorted=None)
///     Produces dmdt-maps from light curves given in columnar form
/// gausses_from_columnar(edges, t, m, sigma, sorted=None)
///     Produces smeared dmdt-maps from columnar light curves
///
#[pyclass]
pub struct DmDt {
    dmdt_f64: GenericDmDt<f64>,
    dmdt_f32: GenericDmDt<f32>,
}

#[pymethods]
impl DmDt {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[args(
    min_lgdt,
    max_lgdt,
    max_abs_dm,
    lgdt_size,
    dm_size,
    norm = "vec![]",
    n_jobs = -1,
    approx_erf = "false"
    )]
    fn __new__(
        min_lgdt: f64,
        max_lgdt: f64,
        max_abs_dm: f64,
        lgdt_size: usize,
        dm_size: usize,
        norm: Vec<&str>,
        n_jobs: i64,
        approx_erf: bool,
    ) -> Res<Self> {
        let dmdt_f32 = lcdmdt::DmDt {
            lgdt_grid: lcdmdt::Grid::new(min_lgdt as f32, max_lgdt as f32, lgdt_size),
            dm_grid: lcdmdt::Grid::new(-max_abs_dm as f32, max_abs_dm as f32, dm_size),
        };
        let dmdt_f64 = lcdmdt::DmDt {
            lgdt_grid: lcdmdt::Grid::new(min_lgdt, max_lgdt, lgdt_size),
            dm_grid: lcdmdt::Grid::new(-max_abs_dm, max_abs_dm, dm_size),
        };
        let norm = norm
            .iter()
            .map(|&s| match s {
                "lgdt" => Ok(NormFlag::LgDt),
                "max" => Ok(NormFlag::Max),
                _ => Err(Exception::ValueError(format!(
                    "normalisation name {:?} is unknown, known names are: \"lgdt\", \"norm\"",
                    s
                ))),
            })
            .collect::<Res<BitFlags<NormFlag>>>()?;
        let error_func = match approx_erf {
            true => lcdmdt::ErrorFunction::Eps1Over1e3,
            false => lcdmdt::ErrorFunction::Exact,
        };
        let n_jobs = if n_jobs <= 0 {
            num_cpus::get()
        } else {
            n_jobs as usize
        };
        Ok(Self {
            dmdt_f32: GenericDmDt {
                dmdt: dmdt_f32,
                norm,
                error_func,
                n_jobs,
            },
            dmdt_f64: GenericDmDt {
                dmdt: dmdt_f64,
                norm,
                error_func,
                n_jobs,
            },
        })
    }

    #[getter]
    fn get_n_jobs(&self) -> usize {
        self.dmdt_f32.n_jobs
    }

    #[setter]
    fn set_n_jobs(&mut self, value: i64) -> Res<()> {
        if value <= 0 {
            Err(Exception::ValueError(
                "cannot set non-positive n_jobs value".to_owned(),
            ))
        } else {
            self.dmdt_f32.n_jobs = value as usize;
            self.dmdt_f64.n_jobs = value as usize;
            Ok(())
        }
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.dmdt_f32.dmdt.shape()
    }

    #[getter]
    fn min_lgdt(&self) -> f64 {
        self.dmdt_f64.dmdt.lgdt_grid.get_start()
    }

    #[getter]
    fn max_lgdt(&self) -> f64 {
        self.dmdt_f64.dmdt.lgdt_grid.get_end()
    }

    #[getter]
    fn min_dm(&self) -> f64 {
        self.dmdt_f64.dmdt.dm_grid.get_start()
    }

    #[getter]
    fn max_dm(&self) -> f64 {
        self.dmdt_f64.dmdt.dm_grid.get_end()
    }

    #[getter]
    fn max_abs_dm(&self) -> f64 {
        self.dmdt_f64.dmdt.dm_grid.get_end()
    }

    /// Number of observations per each lg(dt) interval
    ///
    /// Parameters
    /// ----------
    /// t : 1d-ndarray of float
    ///     Time moments, must be sorted
    /// sorted : bool or None, optional
    ///     `True` guarantees that `t` is sorted
    ///
    /// Returns
    /// 1d-array of float
    ///
    #[args(t, sorted = "None")]
    fn count_lgdt(&self, py: Python, t: GenericFloatArray1, sorted: Option<bool>) -> Res<PyObject> {
        match t {
            GenericFloatArray1::Float32(t) => self
                .dmdt_f32
                .count_lgdt(&ArrWrapper::new(t, true), sorted)
                .map(|a| a.into_pyarray(py).into_py(py)),
            GenericFloatArray1::Float64(t) => self
                .dmdt_f64
                .count_lgdt(&ArrWrapper::new(t, true), sorted)
                .map(|a| a.into_pyarray(py).into_py(py)),
        }
    }

    /// Produces dmdt-map from light curve
    ///
    /// Parameters
    /// ----------
    /// t : 1d-ndarray of float
    ///     Time moments, must be sorted
    /// m : 1d-ndarray of float
    ///     Magnitudes
    /// sorted : bool or None, optional
    ///     `True` guarantees that the light curve is sorted
    ///
    /// Returns
    /// -------
    /// 2d-ndarray of float
    ///
    #[args(t, m, sorted = "None")]
    fn points(
        &self,
        py: Python,
        t: GenericFloatArray1,
        m: GenericFloatArray1,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        match (t, m) {
            (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => Ok(self
                .dmdt_f32
                .points(&ArrWrapper::new(t, true), &ArrWrapper::new(m, true), sorted)?
                .into_pyarray(py)
                .into_py(py)),
            (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => Ok(self
                .dmdt_f64
                .points(&ArrWrapper::new(t, true), &ArrWrapper::new(m, true), sorted)?
                .into_pyarray(py)
                .into_py(py)),
            _ => Err(Exception::TypeError(
                "t and m must have the same dtype".to_owned(),
            )),
        }
    }

    /// Produces dmdt-map from a collection of light curves
    ///
    /// The method is performed in parallel using `n_jobs` threads
    ///
    /// Parameters
    /// ----------
    /// lcs : list of (ndarray, ndarray)
    ///     List or tuple of tuple pairs (t, m) represented individual light
    ///     curves. All arrays must have the same dtype
    /// sorted : bool or None, optional
    ///     `True` guarantees that all light curves is sorted
    ///
    /// Returns
    /// -------
    /// 3d-ndarray of float
    ///    
    #[args(lcs, sorted = "None")]
    fn points_many(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1, GenericFloatArray1)>,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_owned()))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => {
                    let wrapped_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => Ok((ArrWrapper::new(t, true), ArrWrapper::new(m, true))),
                        _ => Err(Exception::TypeError(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float32", i))),
                    }).collect::<Res<Vec<_>>>()?;
                    let typed_lcs = wrapped_lcs.iter().map(|(t, m)| (&t[..], &m[..])).collect();
                    Ok(self
                        .dmdt_f32
                        .points_many(typed_lcs, sorted)?
                        .into_pyarray(py)
                        .into_py(py))
                }
                GenericFloatArray1::Float64(_) => {
                    let wrapped_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => Ok((ArrWrapper::new(t, true), ArrWrapper::new(m, true))),
                        _ => Err(Exception::TypeError(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float64", i))),
                    }).collect::<Res<Vec<_>>>()?;
                    let typed_lcs = wrapped_lcs.iter().map(|(t, m)| (&t[..], &m[..])).collect();
                    Ok(self
                        .dmdt_f64
                        .points_many(typed_lcs, sorted)?
                        .into_pyarray(py)
                        .into_py(py))
                }
            }
        }
    }

    /// Reusable iterable yielding dmdt-maps
    ///
    /// The dmdt-maps are produced in parallel using `n_jobs` threads, batches
    /// are being generated in background, so the next batch is started to
    /// generate just after the previous one is yielded. Note that light curves
    /// data are copied, so from the performance point of view it is better to
    /// use `points_many` if you don't need observation dropping
    ///
    /// Parameters
    /// ----------
    /// lcs : list of (ndarray, ndarray)
    ///     List or tuple of tuple pairs (t, m) represented individual light
    ///     curves. All arrays must have the same dtype
    /// sorted : bool or None, optional
    ///     `True` guarantees that all light curves is sorted, default is
    ///     `None`
    /// batch_size : int, optional
    ///     The number of dmdt-maps to yield. The last batch can be smaller.
    ///     Default is 1
    /// yield_index : bool, optional
    ///     Yield a tuple of (indexes, maps) instead of just maps. Could be
    ///     useful when shuffle is `True`. Default is `False`
    /// shuffle : bool, optional
    ///     If `True`, shuffle light curves (not individual observations) on
    ///     each creating of new iterator. Default is `False`
    /// drop_nobs : int or float, optional
    ///     Drop observations from every light curve. If it is a positive
    ///     integer, it is a number of observations to drop. If it is a
    ///     floating point between 0 and 1, it is a part of observation to
    ///     drop. Default is `0`, which means usage of the original data
    /// random_seed : int or None, optional
    ///     Random seed for shuffling and dropping. Default is `None` which
    ///     means random seed
    ///
    /// Returns
    /// -------
    /// Iterable of 3d-ndarray or (1d-ndarray, 3d-ndarray)
    ///
    #[allow(clippy::too_many_arguments)]
    #[args(
        lcs,
        sorted = "None",
        batch_size = 1,
        yield_index = false,
        shuffle = false,
        drop_nobs = "DropNObsType::Int(0)",
        random_seed = "None"
    )]
    fn points_batches(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1, GenericFloatArray1)>,
        sorted: Option<bool>,
        batch_size: usize,
        yield_index: bool,
        shuffle: bool,
        drop_nobs: DropNObsType,
        random_seed: Option<u64>,
    ) -> Res<PyObject> {
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_owned()))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => {
                    let typed_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => {
                            let t = t.to_owned_array();
                            check_sorted(t.as_slice().unwrap(), sorted)?;
                            let m = m.to_owned_array();
                            Ok((t, m))
                        },
                        _ => Err(Exception::TypeError(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float32", i))),
                    }).collect::<Res<Vec<_>>>()?;
                    Ok(DmDtPointsBatchesF32 {
                        dmdt_batches: Arc::new(GenericDmDtBatches::new(
                            self.dmdt_f32.clone(),
                            typed_lcs,
                            batch_size,
                            yield_index,
                            shuffle,
                            drop_nobs,
                            random_seed,
                        )?),
                    }
                    .into_py(py))
                }
                GenericFloatArray1::Float64(_) => {
                    let typed_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => {
                            let t = t.to_owned_array();
                            check_sorted(t.as_slice().unwrap(), sorted)?;
                            let m = m.to_owned_array();
                            Ok((t, m))
                        },
                        _ => Err(Exception::TypeError(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float64", i))),
                    }).collect::<Res<Vec<_>>>()?;
                    Ok(DmDtPointsBatchesF64 {
                        dmdt_batches: Arc::new(GenericDmDtBatches::new(
                            self.dmdt_f64.clone(),
                            typed_lcs,
                            batch_size,
                            yield_index,
                            shuffle,
                            drop_nobs,
                            random_seed,
                        )?),
                    }
                    .into_py(py))
                }
            }
        }
    }

    /// Produces dmdt-maps from light curves given in columnar form
    ///
    /// The method is performed in parallel using `n_jobs` threads
    ///
    /// Parameters
    /// ----------
    /// edges : 1d-ndarray of np.int64
    ///     Indices of light curve edges: each light curve is described by
    ///     `t[edges[i]:edges[i+1]], m[edges[i]:edges[i+1]]`, i.e. edges must
    ///     have `n + 1` elements, where `n` is a number of light curves
    /// t : 1d-ndarray of float
    ///     Time moments, must be sorted
    /// m : 1d-ndarray of float
    ///     Magnitudes
    /// sorted : bool or None, optional
    ///     `True` guarantees that the light curve is sorted
    ///
    /// Returns
    /// -------
    /// 3d-array of float
    ///     First axis is light curve index, two other axes are dmdt map
    ///
    #[args(edges, t, m, sorted = "None")]
    fn points_from_columnar(
        &self,
        py: Python,
        edges: &PyArray1<i64>,
        t: GenericFloatArray1,
        m: GenericFloatArray1,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        match (t, m) {
            (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => Ok(self
                .dmdt_f32
                .points_from_columnar(edges, t, m, sorted)?
                .into_pyarray(py)
                .into_py(py)),
            (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => Ok(self
                .dmdt_f64
                .points_from_columnar(edges, t, m, sorted)?
                .into_pyarray(py)
                .into_py(py)),
            _ => Err(Exception::TypeError(
                "t and m must have the same dtype".to_owned(),
            )),
        }
    }

    /// Produces smeared dmdt-map from light curve
    ///
    /// Parameters
    /// ----------
    /// t : 1d-ndarray of float
    ///     Time moments, must be sorted
    /// m : 1d-ndarray of float
    ///     Magnitudes
    /// sigma : 1d-ndarray of float
    ///     Uncertainties
    /// sorted : bool or None, optional
    ///     `True` guarantees that the light curve is sorted
    ///
    /// Returns
    /// -------
    /// 2d-array of float
    ///
    #[args(t, m, sigma, sorted = "None")]
    fn gausses(
        &self,
        py: Python,
        t: GenericFloatArray1,
        m: GenericFloatArray1,
        sigma: GenericFloatArray1,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        match (t, m, sigma) {
            (
                GenericFloatArray1::Float32(t),
                GenericFloatArray1::Float32(m),
                GenericFloatArray1::Float32(sigma),
            ) => {
                let err2 = GenericDmDt::sigma_to_err2(sigma);
                Ok(self
                    .dmdt_f32
                    .gausses(
                        &ArrWrapper::new(t, true),
                        &ArrWrapper::new(m, true),
                        err2.as_slice_memory_order().unwrap(),
                        sorted,
                    )?
                    .into_pyarray(py)
                    .into_py(py))
            }
            (
                GenericFloatArray1::Float64(t),
                GenericFloatArray1::Float64(m),
                GenericFloatArray1::Float64(sigma),
            ) => {
                let err2 = GenericDmDt::sigma_to_err2(sigma);
                Ok(self
                    .dmdt_f64
                    .gausses(
                        &ArrWrapper::new(t, true),
                        &ArrWrapper::new(m, true),
                        err2.as_slice_memory_order().unwrap(),
                        sorted,
                    )?
                    .into_pyarray(py)
                    .into_py(py))
            }
            _ => Err(Exception::ValueError(
                "t, m and sigma must have the same dtype".to_owned(),
            )),
        }
    }

    /// Produces smeared dmdt-map from a collection of light curves
    ///
    /// The method is performed in parallel using `n_jobs` threads
    ///
    /// Parameters
    /// ----------
    /// lcs : list of (ndarray, ndarray, ndarray)
    ///     List or tuple of tuple pairs (t, m, sigma) represented individual
    ///     light curves. All arrays must have the same dtype
    /// sorted : bool or None, optional
    ///     `True` guarantees that all light curves are sorted
    ///
    /// Returns
    /// -------
    /// 3d-ndarray of float
    ///    
    #[args(lcs, sorted = "None")]
    fn gausses_many(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1, GenericFloatArray1, GenericFloatArray1)>,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_owned()))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => {
                    let wrapped_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m), GenericFloatArray1::Float32(sigma)) => Ok((ArrWrapper::new(t, true), ArrWrapper::new(m, true), GenericDmDt::sigma_to_err2(sigma))),
                        _ => Err(Exception::TypeError(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float32", i))),
                    }).collect::<Res<Vec<_>>>()?;
                    let typed_lcs = wrapped_lcs
                        .iter()
                        .map(|(t, m, err2)| (&t[..], &m[..], err2.as_slice_memory_order().unwrap()))
                        .collect();
                    Ok(self
                        .dmdt_f32
                        .gausses_many(typed_lcs, sorted)?
                        .into_pyarray(py)
                        .into_py(py))
                }
                GenericFloatArray1::Float64(_) => {
                    let wrapped_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m), GenericFloatArray1::Float64(sigma)) => Ok((ArrWrapper::new(t, true), ArrWrapper::new(m, true), GenericDmDt::sigma_to_err2(sigma))),
                        _ => Err(Exception::TypeError(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float32", i))),
                    }).collect::<Res<Vec<_>>>()?;
                    let typed_lcs = wrapped_lcs
                        .iter()
                        .map(|(t, m, err2)| (&t[..], &m[..], err2.as_slice_memory_order().unwrap()))
                        .collect();
                    Ok(self
                        .dmdt_f64
                        .gausses_many(typed_lcs, sorted)?
                        .into_pyarray(py)
                        .into_py(py))
                }
            }
        }
    }

    /// Reusable iterable yielding dmdt-maps
    ///
    /// The dmdt-maps are produced in parallel using `n_jobs` threads, batches
    /// are being generated in background, so the next batch is started to
    /// generate just after the previous one is yielded. Note that light curves
    /// data are copied, so from the performance point of view it is better to
    /// use `gausses_many` if you don't need observation dropping
    ///
    /// Parameters
    /// ----------
    /// lcs : list of (ndarray, ndarray, ndarray)
    ///     List or tuple of tuple pairs (t, m, sigma) represented individual
    ///     light curves. All arrays must have the same dtype
    /// sorted : bool or None, optional
    ///     `True` guarantees that all light curves is sorted, default is
    ///     `None`
    /// batch_size : int, optional
    ///     The number of dmdt-maps to yield. The last batch can be smaller.
    ///     Default is 1
    /// yield_index : bool, optional
    ///     Yield a tuple of (indexes, maps) instead of just maps. Could be
    ///     useful when shuffle is `True`. Default is `False`    
    /// shuffle : bool, optional
    ///     If `True`, shuffle light curves (not individual observations) on
    ///     each creating of new iterator. Default is `False`
    /// drop_nobs : int or float, optional
    ///     Drop observations from every light curve. If it is a positive
    ///     integer, it is a number of observations to drop. If it is a
    ///     floating point between 0 and 1, it is a part of observation to
    ///     drop. Default is `0`, which means usage of the original data
    /// random_seed : int or None, optional
    ///     Random seed for shuffling and dropping. Default is `None` which
    ///     means random seed
    ///
    #[allow(clippy::too_many_arguments)]
    #[args(
        lcs,
        sorted = "None",
        batch_size = 1,
        yield_index = false,
        shuffle = false,
        drop_nobs = "DropNObsType::Int(0)",
        random_seed = "None"
    )]
    fn gausses_batches(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1, GenericFloatArray1, GenericFloatArray1)>,
        sorted: Option<bool>,
        batch_size: usize,
        yield_index: bool,
        shuffle: bool,
        drop_nobs: DropNObsType,
        random_seed: Option<u64>,
    ) -> Res<PyObject> {
        if lcs.is_empty() {
            Err(Exception::ValueError("lcs is empty".to_owned()))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => {
                    let typed_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m), GenericFloatArray1::Float32(sigma)) => {
                            let t = t.to_owned_array();
                            check_sorted(t.as_slice().unwrap(), sorted)?;
                            let m = m.to_owned_array();
                            let err2 = GenericDmDt::sigma_to_err2(sigma);
                            Ok((t, m, err2))
                        },
                        _ => Err(Exception::TypeError(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float32", i))),
                    }).collect::<Res<Vec<_>>>()?;
                    Ok(DmDtGaussesBatchesF32 {
                        dmdt_batches: Arc::new(GenericDmDtBatches::new(
                            self.dmdt_f32.clone(),
                            typed_lcs,
                            batch_size,
                            yield_index,
                            shuffle,
                            drop_nobs,
                            random_seed,
                        )?),
                    }
                    .into_py(py))
                }
                GenericFloatArray1::Float64(_) => {
                    let typed_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m), GenericFloatArray1::Float64(sigma)) => {
                            let t = t.to_owned_array();
                            check_sorted(t.as_slice().unwrap(), sorted)?;
                            let m = m.to_owned_array();
                            let err2 = GenericDmDt::sigma_to_err2(sigma);
                            Ok((t, m, err2))
                        },
                        _ => Err(Exception::TypeError(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float64", i))),
                    }).collect::<Res<Vec<_>>>()?;
                    Ok(DmDtGaussesBatchesF64 {
                        dmdt_batches: Arc::new(GenericDmDtBatches::new(
                            self.dmdt_f64.clone(),
                            typed_lcs,
                            batch_size,
                            yield_index,
                            shuffle,
                            drop_nobs,
                            random_seed,
                        )?),
                    }
                    .into_py(py))
                }
            }
        }
    }

    /// Produces smeared dmdt-maps from light curves given in columnar form
    ///
    /// The method is performed in parallel using `n_jobs` threads
    ///
    /// Parameters
    /// ----------
    /// edges : 1d-ndarray of np.int64
    ///     Indices of light curve edges: each light curve is described by
    ///     `t[edges[i]:edges[i+1]], m[edges[i]:edges[i+1]], sigma[edges[i]:edges[i+1]]`,
    ///     i.e. edges must have `n + 1` elements, where `n` is a number of
    ///     light curves
    /// t : 1d-ndarray of float
    ///     Time moments, must be sorted
    /// m : 1d-ndarray of float
    ///     Magnitudes
    /// sigma : 1d-ndarray of float
    ///     Observation uncertainties
    /// sorted : bool or None, optional
    ///     `True` guarantees that the light curve is sorted
    ///
    /// Returns
    /// -------
    /// 3d-array of float
    ///     First axis is light curve index, two other axes are dmdt map
    ///
    #[args(edges, t, m, sigma, sorted = "None")]
    fn gausses_from_columnar(
        &self,
        py: Python,
        edges: &PyArray1<i64>,
        t: GenericFloatArray1,
        m: GenericFloatArray1,
        sigma: GenericFloatArray1,
        sorted: Option<bool>,
    ) -> Res<PyObject> {
        match (t, m, sigma) {
            (
                GenericFloatArray1::Float32(t),
                GenericFloatArray1::Float32(m),
                GenericFloatArray1::Float32(sigma),
            ) => Ok(self
                .dmdt_f32
                .gausses_from_columnar(edges, t, m, sigma, sorted)?
                .into_pyarray(py)
                .into_py(py)),
            (
                GenericFloatArray1::Float64(t),
                GenericFloatArray1::Float64(m),
                GenericFloatArray1::Float64(sigma),
            ) => Ok(self
                .dmdt_f64
                .gausses_from_columnar(edges, t, m, sigma, sorted)?
                .into_pyarray(py)
                .into_py(py)),
            _ => Err(Exception::TypeError(String::from(
                "t, m and sigma must have the same dtype",
            ))),
        }
    }
}
