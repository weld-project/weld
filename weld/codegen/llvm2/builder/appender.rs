//! Code generation for the appender builder type.
//!
//! Much of this code mirrors the implementation of the `vector` type, and it may be worth merging
//! this module with `llvm2::vector` one day. The main difference between a vector and an appender
//! is that an appender has a third capacity field (in addition to the vector's data pointer and
//! size). The appender also contains methods for dynamic resizing.

use llvm_sys;

use std::ffi::CString;

use crate::error::*;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;
use self::llvm_sys::LLVMIntPredicate::*;
use self::llvm_sys::LLVMTypeKind;

use crate::codegen::llvm2::intrinsic::Intrinsics;
use crate::codegen::llvm2::llvm_exts::*;
use crate::codegen::llvm2::CodeGenExt;
use crate::codegen::llvm2::LLVM_VECTOR_WIDTH;

pub const POINTER_INDEX: u32 = 0;
pub const SIZE_INDEX: u32 = 1;
pub const CAPACITY_INDEX: u32 = 2;

/// The default Appender capacity.
///
/// This *must* be larger than the `LLVM_VECTOR_WIDTH`.
pub const DEFAULT_CAPACITY: i64 = 16;

pub struct Appender {
    pub appender_ty: LLVMTypeRef,
    pub elem_ty: LLVMTypeRef,
    pub name: String,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    new: Option<LLVMValueRef>,
    merge: Option<LLVMValueRef>,
    vmerge: Option<LLVMValueRef>,
    result: Option<LLVMValueRef>,
}

impl CodeGenExt for Appender {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }
}

impl Appender {
    pub unsafe fn define<T: AsRef<str>>(
        name: T,
        elem_ty: LLVMTypeRef,
        context: LLVMContextRef,
        module: LLVMModuleRef,
    ) -> Appender {
        let c_name = CString::new(name.as_ref()).unwrap();
        // An appender is struct with a pointer, size, and capacity.
        let mut layout = [
            LLVMPointerType(elem_ty, 0),
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context),
        ];
        let appender = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(appender, layout.as_mut_ptr(), layout.len() as u32, 0);
        Appender {
            appender_ty: appender,
            elem_ty,
            name: c_name.into_string().unwrap(),
            context,
            module,
            new: None,
            merge: None,
            vmerge: None,
            result: None,
        }
    }

    /// Returns a pointer to the `index`th element in the appender.
    ///
    /// If the `index` is `None`, thie method returns the base pointer. This method does not
    /// perform any bounds checking.
    unsafe fn gen_index(
        &mut self,
        builder: LLVMBuilderRef,
        appender: LLVMValueRef,
        index: Option<LLVMValueRef>,
    ) -> WeldResult<LLVMValueRef> {
        let pointer = LLVMBuildStructGEP(builder, appender, POINTER_INDEX, c_str!(""));
        let pointer = LLVMBuildLoad(builder, pointer, c_str!(""));
        if let Some(index) = index {
            Ok(LLVMBuildGEP(
                builder,
                pointer,
                [index].as_mut_ptr(),
                1,
                c_str!(""),
            ))
        } else {
            Ok(pointer)
        }
    }

    /// Generates code for a new appender.
    pub unsafe fn gen_new(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        run: LLVMValueRef,
        capacity: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            let mut arg_tys = [self.i64_type(), self.run_handle_type()];
            let ret_ty = self.appender_ty;

            let name = format!("{}.new", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let capacity = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);
            let elem_size = self.size_of(self.elem_ty);
            let alloc_size = LLVMBuildMul(builder, elem_size, capacity, c_str!("capacity"));
            let bytes =
                intrinsics.call_weld_run_malloc(builder, run, alloc_size, Some(c_str!("bytes")));
            let elements = LLVMBuildBitCast(
                builder,
                bytes,
                LLVMPointerType(self.elem_ty, 0),
                c_str!("elements"),
            );

            let mut result = LLVMGetUndef(self.appender_ty);
            result = LLVMBuildInsertValue(builder, result, elements, POINTER_INDEX, c_str!(""));
            result = LLVMBuildInsertValue(builder, result, self.i64(0), SIZE_INDEX, c_str!(""));
            result = LLVMBuildInsertValue(builder, result, capacity, CAPACITY_INDEX, c_str!(""));
            LLVMBuildRet(builder, result);

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [capacity, run];
        Ok(LLVMBuildCall(
            builder,
            self.new.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }

    /// Internal merge function generation that supports vectorization.
    ///
    /// Returns an `LLVMValueRef` representing the generated merge function.
    unsafe fn gen_merge_internal(
        &mut self,
        intrinsics: &mut Intrinsics,
        vectorized: bool,
    ) -> WeldResult<LLVMValueRef> {
        // Number of elements merged in at once.
        let (merge_ty, num_elements) = if vectorized {
            (
                LLVMVectorType(self.elem_ty, LLVM_VECTOR_WIDTH),
                LLVM_VECTOR_WIDTH,
            )
        } else {
            (self.elem_ty, 1)
        };

        let name = if vectorized {
            format!("{}.vmerge", self.name)
        } else {
            format!("{}.merge", self.name)
        };

        let mut arg_tys = [
            LLVMPointerType(self.appender_ty, 0),
            merge_ty,
            self.run_handle_type(),
        ];
        let ret_ty = LLVMVoidTypeInContext(self.context);
        let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

        LLVMExtAddAttrsOnFunction(self.context, function, &[LLVMExtAttribute::AlwaysInline]);

        let full_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!("isFull"));
        let finish_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!("finish"));

        // Builder is positioned at the entry block - attempt to merge in value.

        let appender = LLVMGetParam(function, 0);
        let merge_value = LLVMGetParam(function, 1);
        let run_handle = LLVMGetParam(function, 2);

        let size_slot = LLVMBuildStructGEP(builder, appender, SIZE_INDEX, c_str!(""));
        let size = LLVMBuildLoad(builder, size_slot, c_str!("size"));

        let capacity_slot = LLVMBuildStructGEP(builder, appender, CAPACITY_INDEX, c_str!(""));
        let capacity = LLVMBuildLoad(builder, capacity_slot, c_str!("capacity"));

        let new_size = LLVMBuildNSWAdd(
            builder,
            self.i64(i64::from(num_elements)),
            size,
            c_str!("newSize"),
        );

        let full = LLVMBuildICmp(builder, LLVMIntSGT, new_size, capacity, c_str!("full"));
        LLVMBuildCondBr(builder, full, full_block, finish_block);

        // Build the case where the appender is full and we need to alloate more memory.
        LLVMPositionBuilderAtEnd(builder, full_block);
        let new_capacity = LLVMBuildNSWMul(builder, capacity, self.i64(2), c_str!("newCapacity"));
        let elem_size = self.size_of(self.elem_ty);
        let alloc_size = LLVMBuildMul(builder, elem_size, new_capacity, c_str!("allocSize"));
        let base_pointer = self.gen_index(builder, appender, None)?;
        let raw_pointer = LLVMBuildBitCast(
            builder,
            base_pointer,
            LLVMPointerType(self.i8_type(), 0),
            c_str!("rawPtr"),
        );
        let bytes = intrinsics.call_weld_run_realloc(
            builder,
            run_handle,
            raw_pointer,
            alloc_size,
            Some(c_str!("bytes")),
        );
        let typed_bytes =
            LLVMBuildBitCast(builder, bytes, LLVMTypeOf(base_pointer), c_str!("typed"));
        let pointer_slot = LLVMBuildStructGEP(builder, appender, POINTER_INDEX, c_str!(""));
        LLVMBuildStore(builder, typed_bytes, pointer_slot);
        LLVMBuildStore(builder, new_capacity, capacity_slot);
        LLVMBuildBr(builder, finish_block);

        // Build the finish block, which merges the value.
        LLVMPositionBuilderAtEnd(builder, finish_block);

        let mut merge_pointer = self.gen_index(builder, appender, Some(size))?;
        if vectorized {
            merge_pointer = LLVMBuildBitCast(
                builder,
                merge_pointer,
                LLVMPointerType(merge_ty, 0),
                c_str!(""),
            );
        }
        let store_inst = LLVMBuildStore(builder, merge_value, merge_pointer);
        if vectorized {
            LLVMSetAlignment(store_inst, 1);
        }
        LLVMBuildStore(builder, new_size, size_slot);
        LLVMBuildRetVoid(builder);

        LLVMDisposeBuilder(builder);
        Ok(function)
    }

    /// Generates code to merge a value into an appender.
    pub unsafe fn gen_merge(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        run_arg: LLVMValueRef,
        builder_arg: LLVMValueRef,
        value_arg: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        let vectorized = LLVMGetTypeKind(LLVMTypeOf(value_arg)) == LLVMTypeKind::LLVMVectorTypeKind;
        if vectorized && self.vmerge.is_none() {
            self.vmerge = Some(self.gen_merge_internal(intrinsics, true)?);
        } else if !vectorized && self.merge.is_none() {
            self.merge = Some(self.gen_merge_internal(intrinsics, false)?);
        }

        let mut args = [builder_arg, value_arg, run_arg];
        if vectorized {
            Ok(LLVMBuildCall(
                builder,
                self.vmerge.unwrap(),
                args.as_mut_ptr(),
                args.len() as u32,
                c_str!(""),
            ))
        } else {
            Ok(LLVMBuildCall(
                builder,
                self.merge.unwrap(),
                args.as_mut_ptr(),
                args.len() as u32,
                c_str!(""),
            ))
        }
    }

    /// Generates code to get the result from an appender.
    ///
    /// The Appender's result is a vector.
    pub unsafe fn gen_result(
        &mut self,
        builder: LLVMBuilderRef,
        vector_ty: LLVMTypeRef,
        builder_arg: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        // The vector type that the appender generates.
        use crate::codegen::llvm2::vector;
        if self.result.is_none() {
            let mut arg_tys = [LLVMPointerType(self.appender_ty, 0)];
            let ret_ty = vector_ty;
            let name = format!("{}.result", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let appender = LLVMGetParam(function, 0);

            let pointer = self.gen_index(builder, appender, None)?;
            let size_slot = LLVMBuildStructGEP(builder, appender, SIZE_INDEX, c_str!(""));
            let size = LLVMBuildLoad(builder, size_slot, c_str!("size"));

            let mut result = LLVMGetUndef(vector_ty);
            result =
                LLVMBuildInsertValue(builder, result, pointer, vector::POINTER_INDEX, c_str!(""));
            result = LLVMBuildInsertValue(builder, result, size, vector::SIZE_INDEX, c_str!(""));
            LLVMBuildRet(builder, result);

            self.result = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [builder_arg];
        Ok(LLVMBuildCall(
            builder,
            self.result.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }
}
