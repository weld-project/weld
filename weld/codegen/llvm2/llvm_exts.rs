//! Extensions to the LLVM sys module.
//!
//! For consistency the names here are consistent with the LLVM API (and suffixed with `Ext`)
//! instead of the Rust naming convention.

#![allow(non_snake_case)]

extern crate llvm_sys;
extern crate lazy_static;
extern crate libc;

use libc::{c_char, c_uint};

use std::ffi::{CStr, CString};
use std::fmt;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::{LLVMAttributeReturnIndex, LLVMAttributeFunctionIndex};
use self::llvm_sys::transforms::pass_manager_builder::LLVMPassManagerBuilderRef;

use self::llvm_sys::target_machine::LLVMTargetMachineRef;

// Preload the target-specific features.
lazy_static! {
    pub static ref PROCESS_TRIPLE: CString = unsafe {
        let c_str = LLVMExtGetProcessTriple();
        CStr::from_ptr(c_str).to_owned()
    };
    pub static ref HOST_CPU_NAME: CString = unsafe {
        let c_str = LLVMExtGetHostCPUName();
        CStr::from_ptr(c_str).to_owned()
    };
    pub static ref HOST_CPU_FEATURES: CString = unsafe {
        let c_str = LLVMExtGetHostCPUFeatures();
        CStr::from_ptr(c_str).to_owned()
    };
}

/// LLVM attributes.
///
/// For some reason, these are not exposed in `llvm_sys` at the moment, so we provide hardcoded
/// versions of them here.
/// 
/// The list of available attributes is here:
/// See http://llvm.org/docs/LangRef.html#parameter-attributes
/// and http://llvm.org/docs/LangRef.html#function-attributes
/// for the list of available attributes.
///
#[derive(Debug,Copy,Clone,Eq,PartialEq)]
pub enum LLVMExtAttribute {
    AlwaysInline,
    InlineHint,
    NoAlias,
    NoCapture,
    NonNull,
    NoReturn,
    NoUnwind,
    ReadOnly,
}

impl fmt::Display for LLVMExtAttribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::LLVMExtAttribute::*;
        let ref string = match *self {
            AlwaysInline => "alwaysinline",
            InlineHint => "inlinehint",
            NoAlias => "noalias",
            NoCapture => "nocapture",
            NonNull => "nonnull",
            NoReturn => "noreturn",
            NoUnwind => "nounwind",
            ReadOnly => "readonly",
        };
        f.write_str(string)
    }
}

/// Add attributes on an LLVM function, parameter, or return value.
///
/// The `index` passed determines what the attribute is added on.
unsafe fn add_attrs(context: LLVMContextRef,
                             function: LLVMValueRef,
                             attrs: &[LLVMExtAttribute],
                             index: u32) {
    for attr in attrs {
        let name = CString::new(attr.to_string()).unwrap();
        let kind = LLVMGetEnumAttributeKindForName(name.as_ptr(), name.as_bytes().len());
        assert_ne!(kind, 0);

        let attr = LLVMCreateEnumAttribute(context, kind, 0);
        // This function uses parameter numbers 1..N
        LLVMAddAttributeAtIndex(function, index, attr);
    }
}

/// Add attributes on an LLVM function parameter.
///
/// `param` indicates the parameter index.
pub fn LLVMExtAddAttrsOnParameter(context: LLVMContextRef,
                             function: LLVMValueRef,
                             attrs: &[LLVMExtAttribute],
                             index: u32) {
    // Parameters are indexed 1..N
    unsafe { add_attrs(context, function, attrs, index + 1); }
}

/// Add attributes on an LLVM function.
pub fn LLVMExtAddAttrsOnFunction(context: LLVMContextRef,
                             function: LLVMValueRef,
                             attrs: &[LLVMExtAttribute]) {
    unsafe { add_attrs(context, function, attrs, LLVMAttributeFunctionIndex); }
}

/// Add attributes on an LLVM return value.
pub fn LLVMExtAddAttrsOnReturn(context: LLVMContextRef,
                             function: LLVMValueRef,
                             attrs: &[LLVMExtAttribute]) {
    unsafe { add_attrs(context, function, attrs, LLVMAttributeReturnIndex); }
}

/// Add the host-specific attributes to a function.
///
/// These attributes should be added to each function, since they define target-specific features
/// that enhance generated machine code quality.
pub fn LLVMExtAddDefaultAttrs(context: LLVMContextRef, function: LLVMValueRef) {
    unsafe {
        let cpu_name_attr = LLVMCreateStringAttribute(context,
                                                      c_str!("target-cpu"), 10,
                                                      HOST_CPU_NAME.as_ptr(),
                                                      HOST_CPU_NAME.as_bytes().len() as u32);
        let cpu_features_attr = LLVMCreateStringAttribute(context,
                                                          c_str!("target-features"), 15,
                                                          HOST_CPU_FEATURES.as_ptr(),
                                                          HOST_CPU_FEATURES.as_bytes().len() as u32);

        LLVMAddAttributeAtIndex(function, LLVMAttributeFunctionIndex, cpu_name_attr);
        LLVMAddAttributeAtIndex(function, LLVMAttributeFunctionIndex, cpu_features_attr);
    }
}

#[link(name="llvmext", kind="static")]
extern "C" {
    #[no_mangle]
    pub fn LLVMExtGetProcessTriple() -> *const c_char;
    #[no_mangle]
    pub fn LLVMExtGetHostCPUName() -> *const c_char;
    #[no_mangle]
    pub fn LLVMExtGetHostCPUFeatures() -> *const c_char;
    #[no_mangle]
    pub fn LLVMExtAddTargetLibraryInfo(manager: LLVMPassManagerRef);
    #[no_mangle]
    pub fn LLVMExtAddTargetPassConfig(target: LLVMTargetMachineRef,
                                      manager: LLVMPassManagerRef);
    #[no_mangle]
    pub fn LLVMExtPassManagerBuilderSetDisableVectorize(builder: LLVMPassManagerBuilderRef, disabled: c_uint);


}
