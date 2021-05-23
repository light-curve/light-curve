use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::PyErr;
use std::fmt::Debug;
use std::result::Result;
use thiserror::Error;

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Error, Debug)]
#[error("{0}")]
pub enum Exception {
    NotImplementedError(String),
    RuntimeError(String),
    TypeError(String),
    ValueError(String),
}

impl std::convert::From<Exception> for PyErr {
    fn from(err: Exception) -> PyErr {
        match err {
            Exception::NotImplementedError(err) => PyNotImplementedError::new_err(err),
            Exception::RuntimeError(err) => PyRuntimeError::new_err(err),
            Exception::TypeError(err) => PyTypeError::new_err(err),
            Exception::ValueError(err) => PyValueError::new_err(err),
        }
    }
}

pub type Res<T> = Result<T, Exception>;
