extern crate lalrpop_util;

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
pub mod parser;
pub mod grammar;
pub mod partial_types;
pub mod pretty_print;
pub mod type_inference;

#[cfg(test)]
mod tests;
