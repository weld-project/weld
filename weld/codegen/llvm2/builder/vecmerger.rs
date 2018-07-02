//! Trait implementing the VecMerger builder.
//!
//! This builder builds on top of the `Vector` data structure.

extern crate llvm_sys;

use error::*;

use self::llvm_sys::prelude::*;
// use self::llvm_sys::core::*;

use codegen::llvm2::CodeGenExt;
use codegen::llvm2::vector::Vector;

/// Struct encapsulating additional methods specific to `VecMerger`.
///
/// Vectors contain a reference to their vecmerger. The type is generated dynamically if necessary.
pub struct VecMerger {
    new: Option<LLVMValueRef>,
    merge: Option<LLVMValueRef>,
    result: Option<LLVMValueRef>,
}

impl VecMerger {
    pub fn define() -> VecMerger {
        VecMerger {
            new: None,
            merge: None,
            result: None,
        }
    }
}

pub trait VecMergerGen {
    unsafe fn gen_new_vecmerger(&mut self) -> WeldResult<LLVMValueRef>;
}

impl VecMergerGen for Vector {
    unsafe fn gen_new_vecmerger(&mut self) -> WeldResult<LLVMValueRef> {
        Ok(self.i32(0))
    }
}
