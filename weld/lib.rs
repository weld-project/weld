// Disable dead code macros in "build" mode but keep them on in "test" builds so that we don't
// get spurious warnings for functions that are currently only used in tests. This is useful for
// development but can be removed later.
#![cfg_attr(not(test), allow(dead_code))]

#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate easy_ll;
extern crate libc;

use libc::c_char;
use std::ffi::CStr;

use llvm::LlvmGenerator;
use pretty_print::*;
use type_inference::*;
use sir::ast_to_sir;

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

#[no_mangle]
pub extern "C" fn weld_module_compile(prog: *const c_char,
                                      conf: *const c_char)
                                      -> *mut easy_ll::CompiledModule {
    // Parse stuff.
    let prog = unsafe {
        assert!(!prog.is_null());
        CStr::from_ptr(prog)
    };
    let conf = unsafe {
        assert!(!conf.is_null());
        CStr::from_ptr(conf)
    };

    let prog = prog.to_str().unwrap().trim();
    let conf = conf.to_str().unwrap();

    println!("Got Program {}", prog);
    println!("Got Conf {}", conf);

    let program = parser::parse_program(prog);
    if let Err(ref e) = program {
        println!("Error during parsing: {:?}", e);
    }

    let program = program.unwrap();

    let expr = macro_processor::process_program(&program);
    if let Err(ref e) = expr {
        println!("Error during macro substitution: {}", e);
        return std::ptr::null_mut();
    }
    let mut expr = expr.unwrap();
    println!("After macro substitution:\n{}\n", print_expr(&expr));

    transforms::inline_apply(&mut expr);
    println!("After inlining applies:\n{}\n", print_expr(&expr));

    if let Err(ref e) = infer_types(&mut expr) {
        println!("Error during type inference: {}\n", e);
        println!("Partially inferred types:\n{}\n", print_typed_expr(&expr));
        return std::ptr::null_mut();
    }

    println!("After type inference:\n{}\n", print_typed_expr(&expr));
    println!("Expression type: {}\n", print_type(&expr.ty));

    let mut expr = expr.to_typed().unwrap();

    transforms::fuse_loops_horizontal(&mut expr);
    println!("After horizontal loop fusion:\n{}\n",
             print_typed_expr(&expr));

    transforms::fuse_loops_vertical(&mut expr);
    println!("After vertical loop fusion:\n{}\n", print_typed_expr(&expr));

    let sir_result = ast_to_sir(&expr);
    match sir_result {
        Ok(sir) => {
            println!("SIR representation:\n{}\n", &sir);
            let mut llvm_gen = LlvmGenerator::new();
            if let Err(ref e) = llvm_gen.add_function_on_pointers("run", &sir) {
                println!("Error during LLVM code gen:\n{}\n", e);
            } else {
                let llvm_code = llvm_gen.result();
                println!("LLVM code:\n{}\n", llvm_code);

                let module = easy_ll::compile_module(&llvm_code);
                match module {
                    Err(ref e) => {
                        println!("Error during LLVM compilation:\n{}\n", e);
                        return std::ptr::null_mut();
                    }
                    _ => {
                        println!("LLVM module compiled successfully\n");
                    }
                }

                return Box::into_raw(Box::new(module.unwrap()));
            }
        }
        Err(ref e) => {
            println!("Error during SIR code gen:\n{}\n", e);
        }
    }

    return std::ptr::null_mut();
}

#[no_mangle]
pub extern "C" fn weld_module_run(ptr: *mut easy_ll::CompiledModule, arg: *const u8) -> i64 {
    let module = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let result = module.run(arg as i64) as *const u8 as i64;
    return result;
}

#[no_mangle]
pub extern "C" fn weld_module_free(ptr: *mut easy_ll::CompiledModule) {
    if ptr.is_null() {
        return;
    }
    unsafe { Box::from_raw(ptr) };

}
#[no_mangle]
pub extern "C" fn weld_error_message() {}

#[no_mangle]
pub extern "C" fn weld_error_free() {}

#[cfg(test)]
mod tests;
