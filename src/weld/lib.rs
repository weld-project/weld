extern crate lalrpop_util;

#[macro_use]
extern crate lazy_static;

/// Utility macro to create an Err result with a WeldError.
macro_rules! weld_err {
    ( $($arg:tt)* ) => ({
        ::std::result::Result::Err($crate::error::WeldError(format!($($arg)*)))
    })
}

// TODO: Not all of these should be public
pub mod ast;
pub mod eval;
pub mod error;
pub mod grammar;
pub mod macro_processor;
pub mod partial_types;
pub mod pretty_print;
pub mod program;
pub mod transforms;
pub mod type_inference;

#[cfg(test)]
mod tests;
