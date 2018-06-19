//! Defines intrinsics in the IR.

extern crate libc;
extern crate llvm_sys;
extern crate fnv;

use fnv::FnvHashMap;

use libc::c_char;

use std::default::Default;
use std::ffi::CString;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

/// Intrinsics defined in the code generator.
pub struct Intrinsics {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    intrinsics: FnvHashMap<String, LLVMValueRef>, 
}

impl Intrinsics {
    pub unsafe fn defaults(context: LLVMContextRef, module: LLVMModuleRef) -> Intrinsics {
        let mut map = FnvHashMap::default();

        // Define some common types.
        let int64 = LLVMInt64TypeInContext(context);
        let int8p = LLVMPointerType(LLVMInt8TypeInContext(context), 0);
        let void = LLVMVoidTypeInContext(context);

        // Defines the default intrinsics used by the Weld runtime.
        let mut params = vec![int64];
        let name = CString::new("weld_rt_get_result").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        let mut params = vec![];
        let name = CString::new("weld_rt_get_run_id").unwrap();
        let fn_type = LLVMFunctionType(int64, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        let mut params = vec![int64, int64];
        let name = CString::new("weld_rt_malloc").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        let mut params = vec![int64, int8p, int64];
        let name = CString::new("weld_rt_realloc").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(module, name.as_ptr(), fn_type);
        map.insert(name.into_string().unwrap(), function);

        let mut params = vec![int8p];
        let name = CString::new("weld_rt_free").unwrap();
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
        let name = CString::new(name.as_ref()).unwrap();
        let fn_type = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        self.intrinsics.insert(name.into_string().unwrap(), function);
    }

    /// Convinience wrapper for calling the `weld_rt_get_run_id` intrinsic.
    pub unsafe fn call_weld_rt_get_run_id(&mut self,
                                          builder: LLVMBuilderRef,
                                          name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [];
        LLVMBuildCall(builder,
                      self.get("weld_rt_get_run_id").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_rt_malloc` intrinsic.
    pub unsafe fn call_weld_rt_malloc(&mut self,
                                      builder: LLVMBuilderRef,
                                      run_id: LLVMValueRef,
                                      size: LLVMValueRef,
                                      name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run_id, size];
        LLVMBuildCall(builder,
                      self.get("weld_rt_malloc").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_rt_remalloc` intrinsic.
    pub unsafe fn call_weld_rt_realloc(&mut self,
                                      builder: LLVMBuilderRef,
                                      run_id: LLVMValueRef,
                                      pointer: LLVMValueRef,
                                      size: LLVMValueRef,
                                      name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run_id, pointer, size];
        LLVMBuildCall(builder,
                      self.get("weld_rt_realloc").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_rt_free` intrinsic.
    pub unsafe fn call_weld_rt_free(&mut self,
                                      builder: LLVMBuilderRef,
                                      run_id: LLVMValueRef,
                                      pointer: LLVMValueRef,
                                      name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run_id, pointer];
        LLVMBuildCall(builder,
                      self.get("weld_rt_free").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

}
