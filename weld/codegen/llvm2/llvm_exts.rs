//! Extensions to the LLVM sys module.
//!
//! For consistency the names here are consistent with the LLVM API (and suffixed with `Ext`)
//! instead of the Rust naming convention.

#![allow(non_snake_case)]

extern crate llvm_sys;

use std::ffi::CString;
use std::fmt;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::{LLVMAttributeReturnIndex, LLVMAttributeFunctionIndex};

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
    ByVal,
    Cold,
    InlineHint,
    MinSize,
    Naked,
    NoAlias,
    NoCapture,
    NoInline,
    NonNull,
    NoRedZone,
    NoReturn,
    NoUnwind,
    OptimizeForSize,
    ReadOnly,
    SExt,
    StructRet,
    UWTable,
    ZExt,
    InReg,
    SanitizeThread,
    SanitizeAddress,
    SanitizeMemory,
}

impl fmt::Display for LLVMExtAttribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::LLVMExtAttribute::*;
        let ref string = match *self {
            AlwaysInline => "alwaysinline",
            ByVal => "byval",
            Cold => "cold",
            InlineHint => "inlinehint",
            MinSize => "minsize",
            Naked => "naked",
            NoAlias => "noalias",
            NoCapture => "nocapture",
            NoInline => "noinline",
            NonNull => "nonnull",
            NoRedZone => "noredzone",
            NoReturn => "noreturn",
            NoUnwind => "nounwind",
            OptimizeForSize => "optsize",
            ReadOnly => "readonly",
            SExt => "sext",
            StructRet => "sret",
            UWTable => "uwtable",
            ZExt => "zext",
            InReg => "inreg",
            SanitizeThread => "sanitize_thread", 
            SanitizeAddress => "sanitize_address",
            SanitizeMemory => "santiize_memory",
        };
        f.write_str(string)
    }
}

/// Add an attribute on an LLVM function, parameter, or return value.
///
/// The `index` passed determines what the attribute is added on.
unsafe fn add_attr(context: LLVMContextRef,
                             function: LLVMValueRef,
                             attr: LLVMExtAttribute,
                             index: u32) {
    // TODO just make these constants...
    let name = CString::new(attr.to_string()).unwrap();
    let kind = LLVMGetEnumAttributeKindForName(name.as_ptr(), name.as_bytes().len());
    assert_ne!(kind, 0);

    let attr = LLVMCreateEnumAttribute(context, kind, 0);
    // This function uses parameter numbers 1..N
    LLVMAddAttributeAtIndex(function, index, attr);
}



/// Add an attribute on an LLVM function parameter.
///
/// `param` indicates the parameter index.
pub fn LLVMExtAddAttrOnParameter(context: LLVMContextRef,
                             function: LLVMValueRef,
                             attr: LLVMExtAttribute,
                             index: u32) {
    // Parameters are indexed 1..N
    unsafe { add_attr(context, function, attr, index + 1); }
}

/// Add an attribute on an LLVM function.
pub fn LLVMExtAddAttrOnFunction(context: LLVMContextRef,
                             function: LLVMValueRef,
                             attr: LLVMExtAttribute) {
    unsafe { add_attr(context, function, attr, LLVMAttributeFunctionIndex); }
}

/// Add an attribute on an LLVM return value.
pub fn LLVMExtAddAttrOnReturn(context: LLVMContextRef,
                             function: LLVMValueRef,
                             attr: LLVMExtAttribute) {
    unsafe { add_attr(context, function, attr, LLVMAttributeReturnIndex); }
}
