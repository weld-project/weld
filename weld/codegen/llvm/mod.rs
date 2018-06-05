//! LLVM-based backend for Weld.

use error::*;

pub use self::llvm::*;
pub use self::easy_ll::load_library;

mod llvm;
mod easy_ll;

impl From<easy_ll::LlvmError> for WeldCompileError {
    fn from(err: easy_ll::LlvmError) -> WeldCompileError {
        WeldCompileError::new(err.to_string())
    }
}
