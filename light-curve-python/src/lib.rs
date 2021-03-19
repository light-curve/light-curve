use conv::ConvUtil;
use enumflags2::{bitflags, BitFlags};
use itertools::Itertools;
use light_curve_dmdt as lcdmdt;
use light_curve_feature as lcf;
use ndarray::Array1 as NDArray;
use ndarray::IntoNdProducer;
use numpy::{Element, IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::class::iter::PyIterProtocol;
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pymodule;
use rayon::prelude::*;
use std::cell::RefCell;
use std::ops::{Deref, Range};
use std::sync::Arc;
use std::thread::JoinHandle;

type F = f64;
type Arr<T> = PyArray1<T>;
type PyArrF = Py<Arr<F>>;

enum ArrWrapper<'a, T> {
    Readonly(PyReadonlyArray1<'a, T>),
    Owned(NDArray<T>),
}

impl<'a, T> ArrWrapper<'a, T>
where
    T: Element + num_traits::identities::Zero,
{
    fn new(a: &'a PyArray1<T>, required: bool) -> Self {
        match (a.is_contiguous(), required) {
            (true, _) => Self::Readonly(a.readonly()),
            (false, true) => Self::Owned(a.to_owned_array()),
            (false, false) => Self::Owned(ndarray::Array1::<T>::zeros(a.len())),
        }
    }
}

impl<'a, T> Deref for ArrWrapper<'a, T>
where
    T: Element,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Readonly(a) => a.as_slice().unwrap(),
            Self::Owned(a) => a.as_slice().unwrap(),
        }
    }
}

fn is_sorted<T>(a: &[T]) -> bool
where
    T: PartialOrd,
{
    a.iter().tuple_windows().all(|(a, b)| a < b)
}

fn check_sorted<T>(a: &[T], sorted: Option<bool>) -> PyResult<()>
where
    T: PartialOrd,
{
    match sorted {
        Some(true) => Ok(()),
        Some(false) => Err(PyNotImplementedError::new_err(
            "sorting is not implemented, please provide time-sorted arrays",
        )),
        None => {
            if is_sorted(&a) {
                Ok(())
            } else {
                Err(PyValueError::new_err("t must be in ascending order"))
            }
        }
    }
}

#[derive(FromPyObject)]
enum GenericFloatArray1<'a> {
    #[pyo3(transparent, annotation = "np.ndarray[float32]")]
    Float32(&'a Arr<f32>),
    #[pyo3(transparent, annotation = "np.ndarray[float64]")]
    Float64(&'a Arr<f64>),
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
    fn sigma_to_err2(sigma: &Arr<T>) -> ndarray::Array1<T> {
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

    fn count_lgdt(&self, t: &[T], sorted: Option<bool>) -> PyResult<ndarray::Array1<T>> {
        check_sorted(t, sorted)?;
        Ok(self
            .dmdt
            .lgdt_points(t)
            .mapv(|x| x.value_as::<T>().unwrap()))
    }

    fn points(&self, t: &[T], m: &[T], sorted: Option<bool>) -> PyResult<ndarray::Array2<T>> {
        check_sorted(t, sorted)?;

        let mut result = self.dmdt.points(t, m).mapv(|x| x.value_as::<T>().unwrap());
        self.normalize(&mut result, &t);
        Ok(result)
    }

    fn points_many(
        &self,
        lcs: Vec<(&[T], &[T])>,
        sorted: Option<bool>,
    ) -> PyResult<ndarray::Array3<T>> {
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
                    .try_for_each::<_, PyResult<_>>(|(mut map, (t, m))| {
                        map.assign(&self.points(t, m, sorted)?);
                        Ok(())
                    })
            })?;
        Ok(result)
    }

    fn points_from_columnar(
        &self,
        edges: &Arr<i64>,
        t: &Arr<T>,
        m: &Arr<T>,
        sorted: Option<bool>,
    ) -> PyResult<ndarray::Array3<T>> {
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
                .collect::<PyResult<Vec<_>>>()
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
    ) -> PyResult<ndarray::Array2<T>> {
        check_sorted(t, sorted)?;

        let mut result = self.dmdt.gausses(&t, &m, err2, &self.error_func);
        self.normalize(&mut result, &t);
        Ok(result)
    }

    fn gausses_many(
        &self,
        lcs: Vec<(&[T], &[T], &[T])>,
        sorted: Option<bool>,
    ) -> PyResult<ndarray::Array3<T>> {
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
                    .try_for_each::<_, PyResult<_>>(|(mut map, (t, m, err2))| {
                        map.assign(&self.gausses(t, m, err2, sorted)?);
                        Ok(())
                    })
            })?;
        Ok(result)
    }

    fn gausses_from_columnar(
        &self,
        edges: &Arr<i64>,
        t: &Arr<T>,
        m: &Arr<T>,
        sigma: &Arr<T>,
        sorted: Option<bool>,
    ) -> PyResult<ndarray::Array3<T>> {
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
                .collect::<PyResult<Vec<_>>>()
        })?;
        let map_views: Vec<_> = maps.iter().map(|a| a.view()).collect();
        Ok(ndarray::concatenate(ndarray::Axis(0), &map_views).unwrap())
    }
}

#[derive(Clone)]
struct GenericDmDtPointsBatches<T> {
    dmdt: GenericDmDt<T>,
    lcs: Vec<(ndarray::Array1<T>, ndarray::Array1<T>)>,
    sorted: Option<bool>,
    batch_size: usize,
}

#[derive(Clone)]
struct GenericDmDtGaussesBatches<T> {
    dmdt: GenericDmDt<T>,
    lcs: Vec<(ndarray::Array1<T>, ndarray::Array1<T>, ndarray::Array1<T>)>,
    sorted: Option<bool>,
    batch_size: usize,
}

macro_rules! dmdt_points_batches {
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
            range: Range<usize>,
            worker_thread: RefCell<Option<JoinHandle<PyResult<ndarray::Array3<$t>>>>>,
        }

        impl $name_iter {
            fn new(dmdt_batches: Arc<$generic>) -> Self {
                let range = 0..usize::min(dmdt_batches.batch_size, dmdt_batches.lcs.len());

                let worker_thread = RefCell::new(Some(Self::worker_thread(&dmdt_batches, &range)));

                Self {
                    dmdt_batches,
                    range,
                    worker_thread,
                }
            }

            fn worker_thread(
                dmdt_batches: &Arc<$generic>,
                range: &Range<usize>,
            ) -> JoinHandle<PyResult<ndarray::Array3<$t>>> {
                let dmdt_batches = dmdt_batches.clone();
                let range = range.clone();
                std::thread::spawn(move || Self::worker(dmdt_batches, range))
            }

            fn worker(
                dmdt_batches: Arc<$generic>,
                range: Range<usize>,
            ) -> PyResult<ndarray::Array3<$t>> {
                $worker(dmdt_batches, range)
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

            fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<Py<numpy::PyArray3<$t>>>> {
                if slf.worker_thread.borrow().is_none() {
                    return Ok(None);
                }

                slf.range = slf.range.end
                    ..usize::min(
                        slf.range.end + slf.dmdt_batches.batch_size,
                        slf.dmdt_batches.lcs.len(),
                    );

                // Replace with None temporary to not have two workers running simultaneously
                let result = slf.worker_thread.replace(None).unwrap().join().unwrap()?;
                if !slf.range.is_empty() {
                    slf.worker_thread
                        .replace(Some(Self::worker_thread(&slf.dmdt_batches, &slf.range)));
                }

                Ok(Some(result.into_pyarray(slf.py()).to_owned()))
            }
        }
    };
}

macro_rules! py_dmdt_batches {
    ($worker: expr, $($generic: ty, $name_batches: ident, $name_iter: ident, $t: ty),* $(,)?) => {
        $(
            dmdt_points_batches!($worker, $generic, $name_batches, $name_iter, $t);
        )*
    };
}

py_dmdt_batches!(
    |dmdt_batches: Arc<GenericDmDtPointsBatches<_>>, range: Range<usize>| {
        let lcs: Vec<_> = dmdt_batches.lcs[range]
            .iter()
            .map(|(t, m)| {
                (
                    t.as_slice_memory_order().unwrap(),
                    m.as_slice_memory_order().unwrap(),
                )
            })
            .collect();
        Ok(dmdt_batches.dmdt.points_many(lcs, dmdt_batches.sorted)?)
    },
    GenericDmDtPointsBatches<f32>,
    DmDtPointsBatchesF32,
    DmDtPointsIterF32,
    f32,
    GenericDmDtPointsBatches<f64>,
    DmDtPointsBatchesF64,
    DmDtPointsIterF64,
    f64,
);

py_dmdt_batches!(
    |dmdt_batches: Arc<GenericDmDtGaussesBatches<_>>, range: Range<usize>| {
        let lcs: Vec<_> = dmdt_batches.lcs[range]
            .iter()
            .map(|(t, m, err2)| {
                (
                    t.as_slice_memory_order().unwrap(),
                    m.as_slice_memory_order().unwrap(),
                    err2.as_slice_memory_order().unwrap(),
                )
            })
            .collect();
        Ok(dmdt_batches.dmdt.gausses_many(lcs, dmdt_batches.sorted)?)
    },
    GenericDmDtGaussesBatches<f32>,
    DmDtGaussesBatchesF32,
    DmDtGaussesIterF32,
    f32,
    GenericDmDtGaussesBatches<f64>,
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
///     as `points_many()` or `gausses_from_columnar()`, default is `-1` which
///     means to use as many threads as CPU cores
/// approx_erf : bool, optional
///     Use approximation normal CDF in `gausses*` methods, reduces accuracy,
///     but has better performance, default is `False`
///
/// Attributes
/// ----------
/// n_jons : int
///     Number of threads to use in paralleled methods
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
/// points_from_columnar(edges, t, m, sorted=None)
///     Produces dmdt-maps from light curves given in columnar form
/// gausses_from_columnar(edges, t, m, sigma, sorted=None)
///     Produces smeared dmdt-maps from columnar light curves
///
#[pyclass]
struct DmDt {
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
    ) -> PyResult<Self> {
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
                _ => Err(PyValueError::new_err(format!(
                    "normalisation name {:?} is unknown, known names are: \"lgdt\", \"norm\"",
                    s
                ))),
            })
            .collect::<PyResult<BitFlags<NormFlag>>>()?;
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
    fn set_n_jobs(&mut self, value: i64) -> PyResult<()> {
        if value <= 0 {
            Err(PyValueError::new_err(
                "cannot set non-positive n_jobs value",
            ))
        } else {
            self.dmdt_f32.n_jobs = value as usize;
            self.dmdt_f64.n_jobs = value as usize;
            Ok(())
        }
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
    fn count_lgdt(
        &self,
        py: Python,
        t: GenericFloatArray1,
        sorted: Option<bool>,
    ) -> PyResult<PyObject> {
        match t {
            GenericFloatArray1::Float32(t) => self
                .dmdt_f32
                .count_lgdt(&ArrWrapper::new(t, true), sorted)
                .map(|a| a.into_pyarray(py).to_owned().into_py(py)),
            GenericFloatArray1::Float64(t) => self
                .dmdt_f64
                .count_lgdt(&ArrWrapper::new(t, true), sorted)
                .map(|a| a.into_pyarray(py).to_owned().into_py(py)),
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
    ) -> PyResult<PyObject> {
        match (t, m) {
            (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => Ok(self
                .dmdt_f32
                .points(&ArrWrapper::new(t, true), &ArrWrapper::new(m, true), sorted)?
                .into_pyarray(py)
                .to_owned()
                .into_py(py)),
            (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => Ok(self
                .dmdt_f64
                .points(&ArrWrapper::new(t, true), &ArrWrapper::new(m, true), sorted)?
                .into_pyarray(py)
                .to_owned()
                .into_py(py)),
            _ => Err(PyTypeError::new_err("t and m must have the same dtype")),
        }
    }

    /// Produces dmdt-map from a collection of light curves
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
    ) -> PyResult<PyObject> {
        if lcs.is_empty() {
            Err(PyValueError::new_err("lcs is empty"))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => {
                    let wrapped_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => Ok((ArrWrapper::new(t, true), ArrWrapper::new(m, true))),
                        _ => Err(PyTypeError::new_err(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float32", i))),
                    }).collect::<PyResult<Vec<_>>>()?;
                    let typed_lcs = wrapped_lcs.iter().map(|(t, m)| (&t[..], &m[..])).collect();
                    Ok(self
                        .dmdt_f32
                        .points_many(typed_lcs, sorted)?
                        .into_pyarray(py)
                        .to_owned()
                        .into_py(py))
                }
                GenericFloatArray1::Float64(_) => {
                    let wrapped_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => Ok((ArrWrapper::new(t, true), ArrWrapper::new(m, true))),
                        _ => Err(PyTypeError::new_err(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float64", i))),
                    }).collect::<PyResult<Vec<_>>>()?;
                    let typed_lcs = wrapped_lcs.iter().map(|(t, m)| (&t[..], &m[..])).collect();
                    Ok(self
                        .dmdt_f64
                        .points_many(typed_lcs, sorted)?
                        .into_pyarray(py)
                        .to_owned()
                        .into_py(py))
                }
            }
        }
    }

    #[args(lcs, sorted = "None", batch_size = 1)]
    fn points_batches(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1, GenericFloatArray1)>,
        sorted: Option<bool>,
        batch_size: usize,
    ) -> PyResult<PyObject> {
        if lcs.is_empty() {
            Err(PyValueError::new_err("lcs is empty"))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => {
                    let typed_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => Ok((t.to_owned_array(), m.to_owned_array())),
                        _ => Err(PyTypeError::new_err(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float32", i))),
                    }).collect::<PyResult<Vec<_>>>()?;
                    Ok(DmDtPointsBatchesF32 {
                        dmdt_batches: Arc::new(GenericDmDtPointsBatches {
                            dmdt: self.dmdt_f32.clone(),
                            lcs: typed_lcs,
                            sorted,
                            batch_size,
                        }),
                    }
                    .into_py(py))
                }
                GenericFloatArray1::Float64(_) => {
                    let typed_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => Ok((t.to_owned_array(), m.to_owned_array())),
                        _ => Err(PyTypeError::new_err(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float64", i))),
                    }).collect::<PyResult<Vec<_>>>()?;
                    Ok(DmDtPointsBatchesF64 {
                        dmdt_batches: Arc::new(GenericDmDtPointsBatches {
                            dmdt: self.dmdt_f64.clone(),
                            lcs: typed_lcs,
                            sorted,
                            batch_size,
                        }),
                    }
                    .into_py(py))
                }
            }
        }
    }

    /// Produces dmdt-maps from light curves given in columnar form
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
        edges: &Arr<i64>,
        t: GenericFloatArray1,
        m: GenericFloatArray1,
        sorted: Option<bool>,
    ) -> PyResult<PyObject> {
        match (t, m) {
            (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m)) => Ok(self
                .dmdt_f32
                .points_from_columnar(edges, t, m, sorted)?
                .into_pyarray(py)
                .to_owned()
                .into_py(py)),
            (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m)) => Ok(self
                .dmdt_f64
                .points_from_columnar(edges, t, m, sorted)?
                .into_pyarray(py)
                .to_owned()
                .into_py(py)),
            _ => Err(PyTypeError::new_err("t and m must have the same dtype")),
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
    ) -> PyResult<PyObject> {
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
                    .to_owned()
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
                    .to_owned()
                    .into_py(py))
            }
            _ => Err(PyValueError::new_err(
                "t, m and sigma must have the same dtype",
            )),
        }
    }

    /// Produces smeared dmdt-map from a collection of light curves
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
    ) -> PyResult<PyObject> {
        if lcs.is_empty() {
            Err(PyValueError::new_err("lcs is empty"))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => {
                    let wrapped_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m), GenericFloatArray1::Float32(sigma)) => Ok((ArrWrapper::new(t, true), ArrWrapper::new(m, true), GenericDmDt::sigma_to_err2(sigma))),
                        _ => Err(PyTypeError::new_err(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float32", i))),
                    }).collect::<PyResult<Vec<_>>>()?;
                    let typed_lcs = wrapped_lcs
                        .iter()
                        .map(|(t, m, err2)| (&t[..], &m[..], err2.as_slice_memory_order().unwrap()))
                        .collect();
                    Ok(self
                        .dmdt_f32
                        .gausses_many(typed_lcs, sorted)?
                        .into_pyarray(py)
                        .to_owned()
                        .into_py(py))
                }
                GenericFloatArray1::Float64(_) => {
                    let wrapped_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m), GenericFloatArray1::Float64(sigma)) => Ok((ArrWrapper::new(t, true), ArrWrapper::new(m, true), GenericDmDt::sigma_to_err2(sigma))),
                        _ => Err(PyTypeError::new_err(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float32", i))),
                    }).collect::<PyResult<Vec<_>>>()?;
                    let typed_lcs = wrapped_lcs
                        .iter()
                        .map(|(t, m, err2)| (&t[..], &m[..], err2.as_slice_memory_order().unwrap()))
                        .collect();
                    Ok(self
                        .dmdt_f64
                        .gausses_many(typed_lcs, sorted)?
                        .into_pyarray(py)
                        .to_owned()
                        .into_py(py))
                }
            }
        }
    }

    #[args(lcs, sorted = "None", batch_size = 1)]
    fn gausses_batches(
        &self,
        py: Python,
        lcs: Vec<(GenericFloatArray1, GenericFloatArray1, GenericFloatArray1)>,
        sorted: Option<bool>,
        batch_size: usize,
    ) -> PyResult<PyObject> {
        if lcs.is_empty() {
            Err(PyValueError::new_err("lcs is empty"))
        } else {
            match lcs[0].0 {
                GenericFloatArray1::Float32(_) => {
                    let typed_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float32(t), GenericFloatArray1::Float32(m), GenericFloatArray1::Float32(sigma)) => Ok((t.to_owned_array(), m.to_owned_array(), GenericDmDt::sigma_to_err2(sigma))),
                        _ => Err(PyTypeError::new_err(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float32", i))),
                    }).collect::<PyResult<Vec<_>>>()?;
                    Ok(DmDtGaussesBatchesF32 {
                        dmdt_batches: Arc::new(GenericDmDtGaussesBatches {
                            dmdt: self.dmdt_f32.clone(),
                            lcs: typed_lcs,
                            sorted,
                            batch_size,
                        }),
                    }
                    .into_py(py))
                }
                GenericFloatArray1::Float64(_) => {
                    let typed_lcs = lcs.iter().enumerate().map(|(i, lc)| match lc {
                        (GenericFloatArray1::Float64(t), GenericFloatArray1::Float64(m), GenericFloatArray1::Float64(sigma)) => Ok((t.to_owned_array(), m.to_owned_array(), GenericDmDt::sigma_to_err2(sigma))),
                        _ => Err(PyTypeError::new_err(format!("lcs[{}] elements have mismatched dtype with the lc[0][0] which is float64", i))),
                    }).collect::<PyResult<Vec<_>>>()?;
                    Ok(DmDtGaussesBatchesF64 {
                        dmdt_batches: Arc::new(GenericDmDtGaussesBatches {
                            dmdt: self.dmdt_f64.clone(),
                            lcs: typed_lcs,
                            sorted,
                            batch_size,
                        }),
                    }
                    .into_py(py))
                }
            }
        }
    }

    /// Produces smeared dmdt-maps from light curves given in columnar form
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
        edges: &Arr<i64>,
        t: GenericFloatArray1,
        m: GenericFloatArray1,
        sigma: GenericFloatArray1,
        sorted: Option<bool>,
    ) -> PyResult<PyObject> {
        match (t, m, sigma) {
            (
                GenericFloatArray1::Float32(t),
                GenericFloatArray1::Float32(m),
                GenericFloatArray1::Float32(sigma),
            ) => Ok(self
                .dmdt_f32
                .gausses_from_columnar(edges, t, m, sigma, sorted)?
                .into_pyarray(py)
                .to_owned()
                .into_py(py)),
            (
                GenericFloatArray1::Float64(t),
                GenericFloatArray1::Float64(m),
                GenericFloatArray1::Float64(sigma),
            ) => Ok(self
                .dmdt_f64
                .gausses_from_columnar(edges, t, m, sigma, sorted)?
                .into_pyarray(py)
                .to_owned()
                .into_py(py)),
            _ => Err(PyTypeError::new_err(
                "t, m and sigma must have the same dtype",
            )),
        }
    }
}

#[pyclass(subclass, name = "_FeatureEvaluator")]
struct PyFeatureEvaluator {
    feature_evaluator: Box<dyn lcf::FeatureEvaluator<F>>,
}

#[pymethods]
impl PyFeatureEvaluator {
    #[call]
    #[args(t, m, sigma = "None", sorted = "None", fill_value = "None")]
    fn __call__(
        &self,
        py: Python,
        t: &Arr<F>,
        m: &Arr<F>,
        sigma: Option<&Arr<F>>,
        sorted: Option<bool>,
        fill_value: Option<F>,
    ) -> PyResult<Py<Arr<F>>> {
        if t.len() != m.len() {
            return Err(PyValueError::new_err("t and m must have the same size"));
        }
        if let Some(sigma) = sigma {
            if t.len() != sigma.len() {
                return Err(PyValueError::new_err("t and sigma must have the same size"));
            }
        }
        if t.len() < self.feature_evaluator.min_ts_length() {
            return Err(PyValueError::new_err(format!(
                "input arrays must have size not smaller than {}, but having {}",
                self.feature_evaluator.min_ts_length(),
                t.len()
            )));
        }

        let is_t_required = match (
            self.feature_evaluator.is_t_required(),
            self.feature_evaluator.is_sorting_required(),
            sorted,
        ) {
            // feature requires t
            (true, _, _) => true,
            // t is required because sorting is required and data can be unsorted
            (false, true, Some(false)) | (false, true, None) => true,
            // sorting is required but user guarantees that data is already sorted
            (false, true, Some(true)) => false,
            // neither t or sorting is required
            (false, false, _) => false,
        };
        let t = ArrWrapper::new(t, is_t_required);
        match sorted {
            Some(true) => {}
            Some(false) => {
                return Err(PyNotImplementedError::new_err(
                    "sorting is not implemented, please provide time-sorted arrays",
                ))
            }
            None => {
                if self.feature_evaluator.is_sorting_required() & !is_sorted(&t) {
                    return Err(PyValueError::new_err("t must be in ascending order"));
                }
            }
        }

        let m = ArrWrapper::new(m, self.feature_evaluator.is_m_required());

        let w = sigma.and_then(|sigma| {
            if self.feature_evaluator.is_w_required() {
                let mut w = sigma.to_owned_array();
                w.mapv_inplace(|x| x.powi(-2));
                Some(ArrWrapper::Owned(w))
            } else {
                None
            }
        });

        let mut ts = lcf::TimeSeries::new(&t, &m, w.as_deref());

        let result = match fill_value {
            Some(x) => self.feature_evaluator.eval_or_fill(&mut ts, x),
            None => self
                .feature_evaluator
                .eval(&mut ts)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        };
        Ok(result.into_pyarray(py).to_owned())
    }

    /// Feature names
    #[getter]
    fn names(&self) -> Vec<&str> {
        self.feature_evaluator.get_names()
    }

    /// Feature descriptions
    #[getter]
    fn descriptions(&self) -> Vec<&str> {
        self.feature_evaluator.get_descriptions()
    }
}

/// Features extractor
///
/// Extract multiple features simultaneously, which should be more
/// performant than calling them separately
///
/// Parameters
/// ----------
/// *features : iterable
///     Feature objects
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(*args)"]
struct Extractor {}

#[pymethods]
impl Extractor {
    #[new]
    #[args(args = "*")]
    fn __new__(args: &PyTuple) -> PyResult<(Self, PyFeatureEvaluator)> {
        let evals = args
            .iter()
            .map(|arg| {
                arg.downcast::<PyCell<PyFeatureEvaluator>>()
                    .map(|fe| fe.borrow().feature_evaluator.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::FeatureExtractor::new(evals)),
            },
        ))
    }
}

macro_rules! evaluator {
    ($name: ident, $eval: ty $(,)?) => {
        #[pyclass(extends = PyFeatureEvaluator)]
        #[text_signature = "()"]
        struct $name {}

        #[pymethods]
        impl $name {
            #[new]
            fn __new__() -> (Self, PyFeatureEvaluator) {
                (
                    Self {},
                    PyFeatureEvaluator {
                        feature_evaluator: Box::new(<$eval>::new()),
                    },
                )
            }
        }
    };
}

evaluator!(Amplitude, lcf::Amplitude);

evaluator!(AndersonDarlingNormal, lcf::AndersonDarlingNormal);

/// Fraction of observations beyond N*std from mean
///
/// Parameters
/// ----------
/// nstd : positive float
///     N
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(nstd, /)"]
struct BeyondNStd {}

#[pymethods]
impl BeyondNStd {
    #[new]
    fn __new__(nstd: F) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::BeyondNStd::new(nstd)),
            },
        )
    }
}

#[cfg(feature = "nonlinear-fit")]
evaluator!(BazinFit, lcf::BazinFit);

/// Binned time-series
///
/// Parameters
/// ----------
/// features : iterable
///     Features to extract from binned time-series
/// window : positive float
///     Width of binning interval in units of time
/// offset : float
///     Zero time moment
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(features, window, offset)"]
struct Bins {}

#[pymethods]
impl Bins {
    #[new]
    #[args(features, window, offset)]
    fn __new__(
        py: Python,
        features: PyObject,
        window: F,
        offset: F,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        let mut eval = lcf::Bins::default();
        for x in features.extract::<&PyAny>(py)?.iter()? {
            let feature = x?
                .downcast::<PyCell<PyFeatureEvaluator>>()?
                .borrow()
                .feature_evaluator
                .clone();
            eval.add_feature(feature);
        }
        eval.set_window(window);
        eval.set_offset(offset);
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(eval),
            },
        ))
    }
}

evaluator!(Cusum, lcf::Cusum);

evaluator!(Eta, lcf::Eta);

evaluator!(EtaE, lcf::EtaE);

evaluator!(ExcessVariance, lcf::ExcessVariance);

/// Inner percentile range
///
/// Parameters
/// ----------
/// quantile : positive float
///     Range is (100% * quantile, 100% * (1 - quantile))
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile)"]
struct InterPercentileRange {}

#[pymethods]
impl InterPercentileRange {
    #[new]
    #[args(quantile)]
    fn __new__(quantile: f32) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::InterPercentileRange::new(quantile)),
            },
        )
    }
}

evaluator!(Kurtosis, lcf::Kurtosis);

evaluator!(LinearFit, lcf::LinearFit);

evaluator!(LinearTrend, lcf::LinearTrend);

/// Ratio of two inter-percentile ranges
///
/// Parameters
/// ----------
/// quantile_numerator: positive float
///     Numerator is inter-percentile range (100% * q, 100% (1 - q))
/// quantile_denominator: positive float
///     Denominator is inter-percentile range (100% * q, 100% (1 - q))
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile_numerator, quantile_denominator)"]
struct MagnitudePercentageRatio {}

#[pymethods]
impl MagnitudePercentageRatio {
    #[new]
    #[args(quantile_numerator, quantile_denominator)]
    fn __new__(
        quantile_numerator: f32,
        quantile_denominator: f32,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        if !(0.0..0.5).contains(&quantile_numerator) {
            return Err(PyValueError::new_err(
                "quantile_numerator must be between 0.0 and 0.5",
            ));
        }
        if !(0.0..0.5).contains(&quantile_denominator) {
            return Err(PyValueError::new_err(
                "quantile_denumerator must be between 0.0 and 0.5",
            ));
        }
        Ok((
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::MagnitudePercentageRatio::new(
                    quantile_numerator,
                    quantile_denominator,
                )),
            },
        ))
    }
}

evaluator!(MaximumSlope, lcf::MaximumSlope);

evaluator!(Mean, lcf::Mean);

evaluator!(MeanVariance, lcf::MeanVariance);

evaluator!(Median, lcf::Median);

evaluator!(MedianAbsoluteDeviation, lcf::MedianAbsoluteDeviation,);

/// Median Buffer Range Percentage
///
/// Parameters
/// ----------
/// quantile : positive float
///     Relative range size
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile)"]
struct MedianBufferRangePercentage {}

#[pymethods]
impl MedianBufferRangePercentage {
    #[new]
    #[args(quantile)]
    fn __new__(quantile: F) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::MedianBufferRangePercentage::new(quantile)),
            },
        )
    }
}

evaluator!(PercentAmplitude, lcf::PercentAmplitude);

/// Percent Difference Magnitude Percentile
///
/// Parameters
/// ----------
/// quantile : positive float
///     Relative range size
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(quantile)"]
struct PercentDifferenceMagnitudePercentile {}

#[pymethods]
impl PercentDifferenceMagnitudePercentile {
    #[new]
    #[args(quantile)]
    fn __new__(quantile: f32) -> (Self, PyFeatureEvaluator) {
        (
            Self {},
            PyFeatureEvaluator {
                feature_evaluator: Box::new(lcf::PercentDifferenceMagnitudePercentile::new(
                    quantile,
                )),
            },
        )
    }
}

/// Periodogram-based features
///
/// Parameters
/// ----------
/// peaks : int or None, optional
///     Number of peaks to find
///
/// resolution : float or None, optional
///     Resolution of frequency grid
///
/// max_freq_factor : float or None, optional
///     Mulitplier for Nyquist frequency
///
/// nyquist : str or float or None, optional
///     Type of Nyquist frequency. Could be one of:
///      - 'average': "Average" Nyquist frequency
///      - 'median': Nyquist frequency is defined by median time interval
///         between observations
///      - float: Nyquist frequency is defined by given quantile of time
///         intervals between observations
///
/// fast : bool or None, optional
///     Use "Fast" (approximate and FFT-based) or direct periodogram algorithm
///
/// features : iterable or None, optional
///     Features to extract from periodogram considering it as a time-series
///
#[pyclass(extends = PyFeatureEvaluator)]
#[text_signature = "(peaks=None, resolution=None, max_freq_factor=None, nyquist=None, fast=None, features=None)"]
struct Periodogram {
    eval: lcf::Periodogram<F>,
}

impl Periodogram {
    fn create_eval(
        py: Python,
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<PyObject>,
        fast: Option<bool>,
        features: Option<PyObject>,
    ) -> PyResult<lcf::Periodogram<F>> {
        let mut eval = match peaks {
            Some(peaks) => lcf::Periodogram::new(peaks),
            None => lcf::Periodogram::default(),
        };
        if let Some(resolution) = resolution {
            eval.set_freq_resolution(resolution);
        }
        if let Some(max_freq_factor) = max_freq_factor {
            eval.set_max_freq_factor(max_freq_factor);
        }
        if let Some(nyquist) = nyquist {
            let nyquist_freq: Box<dyn lcf::NyquistFreq<F>> =
                if let Ok(s) = nyquist.extract::<&str>(py) {
                    match s {
                        "average" => Box::new(lcf::AverageNyquistFreq {}),
                        "median" => Box::new(lcf::MedianNyquistFreq {}),
                        _ => return Err(PyValueError::new_err(
                            "nyquist must be one of: None, 'average', 'median' or quantile value",
                        )),
                    }
                } else if let Ok(quantile) = nyquist.extract::<f32>(py) {
                    Box::new(lcf::QuantileNyquistFreq { quantile })
                } else {
                    return Err(PyValueError::new_err(
                        "nyquist must be one of: None, 'average', 'median' or quantile value",
                    ));
                };
            eval.set_nyquist(nyquist_freq);
        }
        if let Some(fast) = fast {
            if fast {
                eval.set_periodogram_algorithm(move || Box::new(lcf::PeriodogramPowerFft));
            } else {
                eval.set_periodogram_algorithm(move || Box::new(lcf::PeriodogramPowerDirect));
            }
        }
        if let Some(features) = features {
            for x in features.extract::<&PyAny>(py)?.iter()? {
                let feature = x?
                    .downcast::<PyCell<PyFeatureEvaluator>>()?
                    .borrow()
                    .feature_evaluator
                    .clone();
                eval.add_feature(feature);
            }
        }
        Ok(eval)
    }
}

#[pymethods]
impl Periodogram {
    #[new]
    #[args(
        peaks = "None",
        resolution = "None",
        max_freq_factor = "None",
        nyquist = "None",
        fast = "None",
        features = "None"
    )]
    fn __new__(
        py: Python,
        peaks: Option<usize>,
        resolution: Option<f32>,
        max_freq_factor: Option<f32>,
        nyquist: Option<PyObject>,
        fast: Option<bool>,
        features: Option<PyObject>,
    ) -> PyResult<(Self, PyFeatureEvaluator)> {
        Ok((
            Self {
                eval: Self::create_eval(
                    py,
                    peaks,
                    resolution,
                    max_freq_factor,
                    nyquist.as_ref().map(|x| x.clone_ref(py)),
                    fast,
                    features.as_ref().map(|x| x.clone_ref(py)),
                )?,
            },
            PyFeatureEvaluator {
                feature_evaluator: Box::new(Self::create_eval(
                    py,
                    peaks,
                    resolution,
                    max_freq_factor,
                    nyquist,
                    fast,
                    features,
                )?),
            },
        ))
    }

    /// Angular frequencies and periodogram values
    #[text_signature = "(t, m)"]
    fn freq_power(&self, py: Python, t: &Arr<F>, m: &Arr<F>) -> (PyArrF, PyArrF) {
        let t = ArrWrapper::new(t, true);
        let m = ArrWrapper::new(m, true);
        let mut ts = lcf::TimeSeries::new(&t, &m, None);
        let (freq, power) = self.eval.freq_power(&mut ts);
        (
            freq.into_pyarray(py).to_owned(),
            power.into_pyarray(py).to_owned(),
        )
    }
}

evaluator!(ReducedChi2, lcf::ReducedChi2);

evaluator!(Skew, lcf::Skew);

evaluator!(StandardDeviation, lcf::StandardDeviation);

evaluator!(StetsonK, lcf::StetsonK);

evaluator!(WeightedMean, lcf::WeightedMean);

evaluator!(Duration, lcf::antifeatures::Duration);

evaluator!(MaximumTimeInterval, lcf::antifeatures::MaximumTimeInterval);

evaluator!(MinimumTimeInterval, lcf::antifeatures::MinimumTimeInterval);

evaluator!(ObservationCount, lcf::antifeatures::ObservationCount);

evaluator!(TimeMean, lcf::antifeatures::TimeMean);

evaluator!(
    TimeStandardDeviation,
    lcf::antifeatures::TimeStandardDeviation
);

/// Features highly dependent on time-series cadence
///
/// See feature interface documentation in the top-level module
#[pymodule]
fn antifeatures(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Duration>()?;
    m.add_class::<MaximumTimeInterval>()?;
    m.add_class::<MinimumTimeInterval>()?;
    m.add_class::<ObservationCount>()?;
    m.add_class::<TimeMean>()?;
    m.add_class::<TimeStandardDeviation>()?;

    Ok(())
}

/// High-performance time-series feature extractor
///
/// The module provides a collection of features to be extracted from time-series.
/// This module if based on a Rust crate `light-curve-feature`, features
/// documentation can be found on https://docs.rs/light-curve-feature
///
/// All features are represented by classes with callable instances, which have
/// the same attributes and call signature:
///
/// Attributes
/// ----------
/// names : list of feature names
///
/// Methods
/// -------
/// __call__(t, m, sigma=None, sorted=None, fill_value=None)
///     Extract features and return them as numpy array.
///
///     Parameters
///     ----------
///     t : numpy.ndarray of np.float64 dtype
///         Time moments
///     m : numpy.ndarray of np.float64 dtype
///         Power of observed signal (magnitude or flux)
///     sigma : numpy.ndarray of np.float64 dtype or None, optional
///         Observation error, if None it is assumed to be unity
///     sorted : bool or None, optional
///         Specifies if input array are sorted by time moments.
///         True is for certainly sorted, False is for unsorted.
///         If None is specified than sorting is checked and an exception is
///         raised for unsorted `t`
///     fill_value : float or None, optional
///         Value to fill invalid feature values, for example if count of
///         observations is not enough to find a proper value.
///         None causes exception for invalid features
///
#[pymodule]
fn light_curve(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<DmDt>()?;

    m.add_class::<PyFeatureEvaluator>()?;

    m.add_class::<Extractor>()?;

    m.add_class::<Amplitude>()?;
    m.add_class::<AndersonDarlingNormal>()?;
    m.add_class::<BeyondNStd>()?;
    m.add_class::<Bins>()?;
    m.add_class::<Cusum>()?;
    m.add_class::<Eta>()?;
    m.add_class::<EtaE>()?;
    m.add_class::<ExcessVariance>()?;
    m.add_class::<InterPercentileRange>()?;
    m.add_class::<Kurtosis>()?;
    m.add_class::<LinearFit>()?;
    m.add_class::<LinearTrend>()?;
    m.add_class::<MagnitudePercentageRatio>()?;
    m.add_class::<MaximumSlope>()?;
    m.add_class::<Mean>()?;
    m.add_class::<MeanVariance>()?;
    m.add_class::<Median>()?;
    m.add_class::<MedianAbsoluteDeviation>()?;
    m.add_class::<MedianBufferRangePercentage>()?;
    m.add_class::<PercentAmplitude>()?;
    m.add_class::<PercentDifferenceMagnitudePercentile>()?;
    m.add_class::<Periodogram>()?;
    m.add_class::<ReducedChi2>()?;
    m.add_class::<Skew>()?;
    m.add_class::<StandardDeviation>()?;
    m.add_class::<StetsonK>()?;
    m.add_class::<WeightedMean>()?;

    #[cfg(feature = "nonlinear-fit")]
    m.add_class::<BazinFit>()?;

    m.add_wrapped(wrap_pymodule!(antifeatures))?;

    Ok(())
}
