//! Python bindings for the core Weld API.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use weld;

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
            .map(|e| e.clone().into_string().unwrap())
            .unwrap_or("".to_string());
        Ok(s)
    }

    fn set(&mut self, key: String, value: String) -> PyResult<()> {
        self.conf.set(key, value);
        Ok(())
    }
}

#[pymethods]
impl WeldContext {
    #[new]
    fn new(obj: &PyRawObject, conf: &WeldConf) {
        obj.init({
            WeldContext {
                context: weld::WeldContext::new(&conf.conf).unwrap(),
            }
        });
    }

    fn memory_usage(&self) -> PyResult<i64> {
        Ok(self.context.memory_usage())
    }
}

#[pymethods]
impl WeldModule {
    #[new]
    fn new(obj: &PyRawObject, code: String, conf: &WeldConf) {
        // TODO Throw an error here.
        let module = weld::WeldModule::compile(code, &conf.conf).unwrap();
        obj.init({
            WeldModule {
                module
            }
        });
    }

    unsafe fn run(&self, context: &mut WeldContext, value: &WeldValue) -> PyResult<WeldValue> {
        Ok(WeldValue::from_weld(self.module.run(&mut context.context, &value.value).unwrap()))
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
fn weld_python(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<WeldConf>()?;
    m.add_class::<WeldContext>()?;
    m.add_class::<WeldValue>()?;
    m.add_class::<WeldModule>()?;
    Ok(())
}
