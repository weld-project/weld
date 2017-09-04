use std::error::Error;

use easy_ll::compile_module;

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
    ", None);
    assert!(module.is_ok());
    assert_eq!(module.unwrap().run(42), 44);
}

#[test]
fn compile_error() {
    let module = compile_module("
       define ZZZZZZZZ @run(i64 %arg) {
           ret i64 0
       }
    ", None);
    assert!(!module.is_ok());
    assert!(module.unwrap_err().description().contains("Compile"));
}

#[test]
fn no_run_function() {
    let module = compile_module("
       define i64 @ZZZZZZZ(i64 %arg) {
           ret i64 0
       }
    ", None);
    assert!(!module.is_ok());
    assert!(module.unwrap_err().description().contains("run function"));
}

#[test]
fn wrong_function_type() {
    let module = compile_module("
       define i64 @run() {
           ret i64 0
       }
    ", None);
    assert!(!module.is_ok());
    assert!(module.unwrap_err().description().contains("wrong type"));
}
