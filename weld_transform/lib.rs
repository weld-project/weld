#[macro_use] extern crate lazy_static;

extern crate weld_ast;
#[macro_use] extern crate weld_error;
extern crate weld_parser;

// TODO: Not all of these should be public
pub mod macro_processor;
pub mod transforms;
pub mod type_inference;
