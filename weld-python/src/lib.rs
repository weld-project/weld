
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
        Ok(self.conf.get(&key).unwrap_or("".to_string()).into())
    }

    fn set(&self, key: String, value: String) -> PyResult<()> {
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
    fn from_weld(value: weld::WeldValue) {
        WeldValue {
            value
        }
    }
}

#[pymethods]
impl WeldValue {
    #[new]
    fn new(obj: &PyRawObject, x: i32) {
        obj.init({
            WeldValue {
                // TODO
                weld::WeldValue::new(std::ptr::null_mut())
            }
        });
    }
}


#[pymodule]
fn weld_python(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<WeldConf>()?;
    m.add_class::<WeldContext>()?;
    Ok(())
}
