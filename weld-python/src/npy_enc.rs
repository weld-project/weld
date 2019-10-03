//! Encoders and decoders for some common NumPy data types.
//!
//! This module supports zero-copy encoding/decoding using NumPy arrays of the following Weld
//! types:
//!
//! * vec[T] where T is an unsigned or signed integer.
//! * vec[T] where T is a float or double.
//! * vec[T] where T is a boolean.
//! * vec[T] where T is a fixed-size string (dtype='|Sx')
//!
//! In addition, this module supports encoding Python string objects, but requires copying data.
//!
//! 2D NumPy arrays are no longer support because their representation as vec[vec[T]] is quite
//! inefficient. Weld will eventually include a tensor[T,shape] type that will support this.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use numpy::PyArray1;

use weld::data;

#[pyclass]
struct WeldVec {
    vec: data::WeldVec<i32>,
}

impl WeldVec {
    fn new(vec: data::WeldVec<i32>) -> Self {
        WeldVec {
            vec
        }
    }
}

/// Converts a 1D NumPy array into a `WeldVec` that can be passed to the Weld runtime.
fn to_weld_1d<T>(array: &PyArray1<T>) -> data::WeldVec<T> {
    let array_obj = array.as_array_ptr();
    data::WeldVec {
        data: unsafe { (*array_obj).data as *mut T },
        len: array.len() as i64
    }
}

/// Converts a 1D NumPy array into a `WeldVec` that can be passed to the Weld runtime.
fn to_numpy_1d<T>(vec: &data::WeldVec<T>) -> PyArray1<T> {
    unimplemented!("This needs to be implemented");
}

#[pyfunction]
fn to_weld_1d_i32(array: &PyArray1<i32>) -> PyResult<WeldVec> {
    Ok(WeldVec::new(to_weld_1d::<i32>(array)))
}

#[pymodule]
fn numpy_encoders(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(to_weld_1d_i32)).unwrap();
    Ok(())
}
