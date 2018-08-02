use std::error::Error;

use codegen::llvm::easy_ll::compile_module;
use codegen::Runnable;

#[test]
fn basic_use() {
    let module = compile_module("
       define i64 @bar(i64 %arg) {
           %1 = add i64 %arg, 1
           ret i64 %1
       }

       define i64 @run(i64 %arg) {
           %1 = add i64 %arg, 1
           %2 = call i64 @bar(i64 %1)
           ret i64 %2
       }
    ", 2, false, None);
    assert!(module.is_ok());
    assert_eq!(module.unwrap().module.run(42), 44);
}

#[test]
fn compile_error() {
    let module = compile_module("
       define ZZZZZZZZ @run(i64 %arg) {
           ret i64 0
       }
    ", 2, false, None);
    assert!(!module.is_ok());
    assert!(module.unwrap_err().description().contains("Compile"));
}

#[test]
fn no_run_function() {
    let module = compile_module("
       define i64 @ZZZZZZZ(i64 %arg) {
           ret i64 0
       }
    ", 2, false, None);
    assert!(!module.is_ok());
    assert!(module.unwrap_err().description().contains("run function"));
}

#[test]
fn wrong_function_type() {
    let module = compile_module("
       define i64 @run() {
           ret i64 0
       }
    ", 2, false, None);
    assert!(!module.is_ok());
    assert!(module.unwrap_err().description().contains("wrong type"));
}
