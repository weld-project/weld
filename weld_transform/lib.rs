#[macro_use] extern crate lazy_static;

extern crate weld_ast;
extern crate weld_parser;
#[macro_use] extern crate weld_error;

// TODO: Not all of these should be public
pub mod macro_processor;
pub mod transforms;
pub mod type_inference;
