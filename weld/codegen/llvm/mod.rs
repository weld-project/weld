//! LLVM-based backend for Weld.

use error::*;

pub use self::easy_ll::load_library;
pub use self::llvm::*;

mod easy_ll;
mod llvm;

impl From<easy_ll::LlvmError> for WeldCompileError {
    fn from(err: easy_ll::LlvmError) -> WeldCompileError {
        WeldCompileError::new(err.to_string())
    }
}
