// Disable dead code macros in "build" mode but keep them on in "test" builds so that we don't
// get spurious warnings for functions that are currently only used in tests. This is useful for
// development but can be removed later.
#![cfg_attr(not(test), allow(dead_code))]

#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate easy_ll;
extern crate libc;

use std::error::Error;
use libc::{c_char, c_void};
use std::ffi::{CString, CStr};

/// Utility macro to create an Err result with a WeldError from a format string.
macro_rules! weld_err {
    ( $($arg:tt)* ) => ({
        ::std::result::Result::Err($crate::error::WeldError::new(format!($($arg)*)))
    })
}

// TODO: Not all of these should be public
pub mod ast;
pub mod code_builder;
pub mod error;
pub mod llvm;
pub mod macro_processor;
pub mod parser;
pub mod partial_types;
pub mod pretty_print;
pub mod program;
pub mod sir;
pub mod tokenizer;
pub mod transforms;
pub mod type_inference;
pub mod util;

pub struct WeldError {
    success: bool,
    message: CString,
}

pub struct WeldObject {
    data: *const c_void,
    owned: bool,
}

impl WeldError {
    fn new(success: bool, message: &str) -> WeldError {
        WeldError {
            success: success,
            message: CString::new(message).unwrap(),
        }
    }
}

#[no_mangle]
pub extern "C" fn weld_object_new(data: *const c_void) -> *mut WeldObject {
    Box::into_raw(Box::new(WeldObject {
        data: data,
        owned: false,
    }))
}


#[no_mangle]
pub extern "C" fn weld_object_owned(obj: *const WeldObject) -> i32 {
    let obj = unsafe {
        assert!(!obj.is_null());
        &*obj
    };
    obj.owned as i32
}

#[no_mangle]
pub extern "C" fn weld_object_data(obj: *const WeldObject) -> *const c_void {
    let obj = unsafe {
        assert!(!obj.is_null());
        &*obj
    };
    obj.data
}

#[no_mangle]
pub extern "C" fn weld_object_free(obj: *mut WeldObject) {
    if obj.is_null() {
        return;
    }
    unsafe { Box::from_raw(obj) };
}

#[no_mangle]
pub extern "C" fn weld_module_compile(code: *const c_char,
                                      conf: *const c_char,
                                      err: *mut *mut WeldError)
                                      -> *mut easy_ll::CompiledModule {
    let code = unsafe {
        assert!(!code.is_null());
        CStr::from_ptr(code)
    };
    let code = code.to_str().unwrap().trim();

    let conf = unsafe {
        assert!(!conf.is_null());
        CStr::from_ptr(conf)
    };
    let conf = conf.to_str().unwrap();

    // TODO(shoumik): If the err is already holding an object, will it get dropped?
    let err = unsafe {
        *err = Box::into_raw(Box::new(WeldError::new(true, "Success")));
        &mut **err
    };

    let module = llvm::compile_program(&parser::parse_program(code).unwrap());
    if let Err(ref e) = module {
        // TODO(shoumik): Error codes instead?
        err.success = false;
        err.message = CString::new(e.description()).unwrap();
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(module.unwrap()))
}

#[no_mangle]
pub extern "C" fn weld_module_run(module: *mut easy_ll::CompiledModule,
                                  arg: *const WeldObject,
                                  err: *mut *mut WeldError)
                                  -> *mut WeldObject {
    let module = unsafe {
        assert!(!module.is_null());
        &mut *module
    };

    let arg = unsafe {
        assert!(!arg.is_null());
        &*arg
    };

    // TODO(shoumik): Error reporting - at least basic crashes
    let result = module.run(arg.data as i64) as *const c_void;
    Box::into_raw(Box::new(WeldObject {
        data: result,
        owned: true,
    }))
}

#[no_mangle]
pub extern "C" fn weld_module_free(ptr: *mut easy_ll::CompiledModule) {
    if ptr.is_null() {
        return;
    }
    unsafe { Box::from_raw(ptr) };

}

#[no_mangle]
pub extern "C" fn weld_error_success(err: *mut WeldError) -> i32 {
    let err = unsafe {
        if err.is_null() {
            return 1;
        }
        &mut *err
    };
    err.success as i32
}

#[no_mangle]
pub extern "C" fn weld_error_message(err: *mut WeldError) -> *const c_char {
    let err = unsafe {
        if err.is_null() {
            return std::ptr::null();
        }
        &mut *err
    };
    err.message.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn weld_error_free(err: *mut WeldError) {
    if err.is_null() {
        return;
    }
    unsafe { Box::from_raw(err) };
}

#[cfg(test)]
mod tests;
