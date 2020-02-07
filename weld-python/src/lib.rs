//! Python bindings for the core Weld API.

use pyo3::prelude::*;
use pyo3::import_exception;

use weld;

import_exception!(weld, WeldError);

/// Converts a `Result` to `PyResult`.
///
/// This is needed since we can't implement `From` for a struct that isn't implemented in this
/// module.
trait ToPyErr<T, E> {
    /// Performs the conversion.
    fn to_py(self) -> PyResult<T>;
}

impl<T> ToPyErr<T, weld::WeldError> for weld::WeldResult<T>  {
    fn to_py(self) -> PyResult<T> {
        self.map_err(|e| WeldError::py_err(e.message().to_str().unwrap().to_string()))
    }
}

#[pyclass]
struct WeldContext {
    context: weld::WeldContext,
}

#[pyclass]
struct WeldConf {
    conf: weld::WeldConf,
}

#[pyclass]
struct WeldValue {
    value: weld::WeldValue,
}

#[pyclass]
struct WeldModule {
    module: weld::WeldModule,
}

#[pyclass]
struct WeldType {
    ty: weld::ast::Type,
}

impl WeldType {
    fn new(ty: weld::ast::Type) -> WeldType {
        WeldType {
            ty,
        }
    }
}

#[pyproto]
impl pyo3::class::basic::PyObjectProtocol for WeldType {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.ty.to_string())
    }
}

#[pymethods]
impl WeldConf {
    #[new]
    fn new(obj: &PyRawObject) {
        obj.init({
            WeldConf {
                conf: weld::WeldConf::new(),
            }
        });
    }

    fn get(&self, key: String) -> PyResult<String> {
        let s = self.conf.get(&key)
            .map(|e| e.clone().into_string().unwrap());
        if let Some(s) = s {
            Ok(s)
        } else {
            Err(pyo3::exceptions::KeyError::py_err(format!("'{}'", key)))
        }
    }

    fn set(&mut self, key: String, value: String) -> PyResult<()> {
        self.conf.set(key, value);
        Ok(())
    }
}

#[pymethods]
impl WeldContext {
    #[new]
    fn new(obj: &PyRawObject, conf: &WeldConf) -> PyResult<()> {
        obj.init({
            WeldContext {
                context: weld::WeldContext::new(&conf.conf).to_py()?,
            }
        });
        Ok(())
    }

    fn memory_usage(&self) -> PyResult<i64> {
        Ok(self.context.memory_usage())
    }
}

#[pymethods]
impl WeldModule {
    #[new]
    fn new(obj: &PyRawObject, code: String, conf: &WeldConf) -> PyResult<()> {
        let module = weld::WeldModule::compile(code, &conf.conf).to_py()?;
        obj.init({
            WeldModule {
                module
            }
        });
        Ok(())
    }

    unsafe fn run(&self, context: &mut WeldContext, value: &WeldValue) -> PyResult<WeldValue> {
        Ok(WeldValue::from_weld(self.module.run(&mut context.context, &value.value).to_py()?))
    }

    fn return_type(&self) -> PyResult<WeldType> {
        Ok(WeldType::new(self.module.return_type()))
    }
}

impl WeldValue {
    fn from_weld(value: weld::WeldValue) -> WeldValue {
        WeldValue {
            value
        }
    }
}

#[pymethods]
impl WeldValue {
    #[new]
    fn new(obj: &PyRawObject, pointer: usize) {
        obj.init({
            WeldValue {
                value: weld::WeldValue::new_from_data(pointer as _)
            }
        });
    }

    fn context(&self) -> PyResult<()> {
        Ok(())
    }

    fn data(&self) -> PyResult<usize> {
        Ok(self.value.data() as usize)
    }
}


#[pymodule]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<WeldConf>()?;
    m.add_class::<WeldContext>()?;
    m.add_class::<WeldModule>()?;
    m.add_class::<WeldValue>()?;
    Ok(())
}
