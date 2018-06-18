//! Defines intrinsics in the IR.

extern crate llvm_sys;
extern crate fnv;

use fnv::FnvHashMap;

use std::default::Default;
use std::ffi::CString;

use error::*;

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
}
