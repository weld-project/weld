use std::error::Error;
use std::result::Result;

use super::{initialize, compile_module};

#[test]
fn basic_use() {
    initialize().unwrap();

    let module = compile_module("test_module", "
       define i64 @foo(i64 %arg) {
           ret i64 0
       }
    ");

    assert!(!module.is_ok());
    assert_eq!(module.unwrap_err().description(), "EEK");
    //assert_eq!(
}
