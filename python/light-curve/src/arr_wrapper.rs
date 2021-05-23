use ndarray::Array1;
use numpy::{Element, PyArray1, PyReadonlyArray1};
use std::ops::Deref;

pub enum ArrWrapper<'a, T> {
    Readonly(PyReadonlyArray1<'a, T>),
    Owned(Array1<T>),
}

impl<'a, T> ArrWrapper<'a, T>
where
    T: Element + num_traits::identities::Zero,
{
    pub fn new(a: &'a PyArray1<T>, required: bool) -> Self {
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
