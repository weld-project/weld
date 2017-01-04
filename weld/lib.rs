// Disable dead code macros in "build" mode but keep them on in "test" builds so that we don't
// get spurious warnings for functions that are currently only used in tests. This is useful for
// development but can be removed later.
#![cfg_attr(not(test), allow(dead_code))]

#[macro_use] extern crate lazy_static;
extern crate regex;

extern crate llvm;

// TODO: Not all of these should be public
pub mod ast;
pub mod code_builder;
#[macro_use] pub mod error;
#[macro_use] pub mod codegen;
pub mod macro_processor;
pub mod parser;
pub mod partial_types;
pub mod pretty_print;
pub mod program;
pub mod tokenizer;
pub mod transforms;
pub mod type_inference;
pub mod util;

#[cfg(test)] mod tests;
