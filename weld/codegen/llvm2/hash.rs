//! Implements hashing for Weld types.

extern crate llvm_sys;

use std::ffi::CStr;

use ast::Type;
use error::*;

use super::llvm_exts::*;
use super::llvm_exts::LLVMExtAttribute::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

use super::llvm_exts::*;
use super::llvm_exts::LLVMExtAttribute::*;

use super::CodeGenExt;
use super::LlvmGenerator;

/// Trait for generating hashing code.
pub trait GenHash {
    /// Generates a hash function for a type.
    ///
    /// Hash functions have the type signature `T -> i32`. This method returns the generated
    /// function.
    unsafe fn gen_hash_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef>;
}

impl GenHash for LlvmGenerator {
    unsafe fn gen_hash_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef> {
        use ast::Type::*;
        let result = self.hash_fns.get(ty).cloned();
        if let Some(result) = result {
            return Ok(result)
        }

        let llvm_ty = self.llvm_type(ty)?;
        let mut arg_tys = [LLVMPointerType(llvm_ty, 0)];

        // TODO maybe put this in CodeGenExt as `hash_type`?
        let ret_ty = self.i32_type();

        let c_prefix = LLVMPrintTypeToString(llvm_ty);
        let prefix = CStr::from_ptr(c_prefix);
        let prefix = prefix.to_str().unwrap();
        let name = format!("{}.hash", prefix);
        // Free the allocated string.
        LLVMDisposeMessage(c_prefix);

        let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

        LLVMExtAddAttrsOnFunction(self.context, function, &[InlineHint]);

        LLVMBuildRet(builder, self.i32(0));
        LLVMDisposeBuilder(builder);

        self.hash_fns.insert(ty.clone(), function);
        Ok(function)
    }
}
