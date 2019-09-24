
use pyo3::prelude::*;

use pyo3::wrap_pyfunction;

#[pyclass]
struct WeldConf {
    x: i32
}

#[pymodule]
fn weld_python(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<WeldConf>()?;
    Ok(())
}
