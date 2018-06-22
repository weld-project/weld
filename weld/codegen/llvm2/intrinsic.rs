//! Defines intrinsics in the LLVM IR.

extern crate libc;
extern crate llvm_sys;
extern crate fnv;

use fnv::FnvHashMap;

use libc::c_char;

use error::*;

use std::default::Default;
use std::ffi::CString;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

/// Intrinsics defined in the code generator.
///
/// An intrinsic is any function that appears without a definition in the generated module. Code
/// generators must ensure that instrinics are properly linked upon compilation.
pub struct Intrinsics {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    intrinsics: FnvHashMap<String, LLVMValueRef>, 
}

impl Intrinsics {
    pub unsafe fn defaults(context: LLVMContextRef, module: LLVMModuleRef) -> Intrinsics {
        let mut map = FnvHashMap::default();

        // Define some common types.
        let int32 = LLVMInt32TypeInContext(context);
        let int64 = LLVMInt64TypeInContext(context);
        let int8p = LLVMPointerType(LLVMInt8TypeInContext(context), 0);
        let void = LLVMVoidTypeInContext(context);

        // Defines the default intrinsics used by the Weld runtime.
        let mut params = vec![int32, int64];
        let name = CString::new("weld_run_init").unwrap();
        let fn_type = LLVMFunctionType(int64, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        let mut params = vec![int64];
        let name = CString::new("weld_run_get_result").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        let mut params = vec![int8p];
        let name = CString::new("weld_run_set_result").unwrap();
        let fn_type = LLVMFunctionType(void, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        let mut params = vec![];
        let name = CString::new("weld_run_get_run_id").unwrap();
        let fn_type = LLVMFunctionType(int64, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        let mut params = vec![int64, int64];
        let name = CString::new("weld_run_malloc").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        let mut params = vec![int64, int8p, int64];
        let name = CString::new("weld_run_realloc").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        let mut params = vec![int8p];
        let name = CString::new("weld_run_free").unwrap();
        let fn_type = LLVMFunctionType(void, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        Intrinsics {
            context: context,
            module: module,
            intrinsics: map,
        }
    }

    /// Get the intrinsic function with the given name.
    pub fn get<T: AsRef<str>>(&self, key: T) -> Option<LLVMValueRef> {
        return self.intrinsics.get(key.as_ref()).map(|r| *r)
    }

    /// Add a new intrinsic function with the given name, return type, and argument types.
    pub unsafe fn add<T: AsRef<str>>(&mut self, name: T, ret_ty: LLVMTypeRef, arg_tys: &mut [LLVMTypeRef]) {
        if !self.intrinsics.contains_key(name.as_ref()) {
            let name = CString::new(name.as_ref()).unwrap();
            let fn_type = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
            let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
            self.intrinsics.insert(name.into_string().unwrap(), function);
        }
    }

    /// Generate code to call an intrinsic function with the given name and arguments.
    ///
    /// If the intrinsic is not defined, this function throws an error.
    pub unsafe fn call<T: AsRef<str>>(&mut self,
                                      builder: LLVMBuilderRef,
                                      name: T,
                                      args: &mut [LLVMValueRef]) -> WeldResult<LLVMValueRef> {
        if let Some(func) = self.intrinsics.get(name.as_ref()) {
            if args.len() != LLVMCountParams(*func) as usize {
                unreachable!()
            }
            Ok(LLVMBuildCall(builder, *func, args.as_mut_ptr(), args.len() as u32, c_str!("")))
        } else {
            unreachable!()
        }
    }

    /// Convinience wrapper for calling the `weld_run_init` intrinsic.
    pub unsafe fn call_weld_run_init(&mut self,
                                          builder: LLVMBuilderRef,
                                          nworkers: LLVMValueRef,
                                          memlimit: LLVMValueRef,
                                          name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [nworkers, memlimit];
        LLVMBuildCall(builder,
                      self.get("weld_run_init").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_get_result` intrinsic.
    pub unsafe fn call_weld_run_get_result(&mut self,
                                          builder: LLVMBuilderRef,
                                          run_id: LLVMValueRef,
                                          name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run_id];
        LLVMBuildCall(builder,
                      self.get("weld_run_get_result").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_set_result` intrinsic.
    pub unsafe fn call_weld_run_set_result(&mut self,
                                          builder: LLVMBuilderRef,
                                          pointer: LLVMValueRef,
                                          name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [pointer];
        LLVMBuildCall(builder,
                      self.get("weld_run_set_result").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_get_run_id` intrinsic.
    pub unsafe fn call_weld_run_get_run_id(&mut self,
                                          builder: LLVMBuilderRef,
                                          name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [];
        LLVMBuildCall(builder,
                      self.get("weld_run_get_run_id").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_malloc` intrinsic.
    pub unsafe fn call_weld_run_malloc(&mut self,
                                      builder: LLVMBuilderRef,
                                      run_id: LLVMValueRef,
                                      size: LLVMValueRef,
                                      name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run_id, size];
        LLVMBuildCall(builder,
                      self.get("weld_run_malloc").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_remalloc` intrinsic.
    pub unsafe fn call_weld_run_realloc(&mut self,
                                      builder: LLVMBuilderRef,
                                      run_id: LLVMValueRef,
                                      pointer: LLVMValueRef,
                                      size: LLVMValueRef,
                                      name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run_id, pointer, size];
        LLVMBuildCall(builder,
                      self.get("weld_run_realloc").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_free` intrinsic.
    pub unsafe fn call_weld_run_free(&mut self,
                                      builder: LLVMBuilderRef,
                                      run_id: LLVMValueRef,
                                      pointer: LLVMValueRef,
                                      name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run_id, pointer];
        LLVMBuildCall(builder,
                      self.get("weld_run_free").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

}
