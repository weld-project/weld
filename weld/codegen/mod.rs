//! Target-specific code generation from Weld.
//!
//! This module supports code generation from Weld to a traditional compiler IR. Usually, code
//! generation for converts the Weld AST to the Weld SIR to facilitate code generation, and then
//! maps each SIR instruction to a compiler IR instruction (e.g., LLVM).
//!
//! For now, we only support LLVM, so this module is just a wrapper for the LLVM module.
//! Eventually we will refactor it so common tasks such as Weld AST optimizations and AST-to-SIR
//! conversion occur here and submodules implement backends for different hardware platforms. For
//! example, an NVPTX GPU-based backend would be a new submodule that emits NVPTX code, and this
//! module will provide the shared Weld optimization and SIR conversion logic that currently lives in the
//! `llvm` module.

pub mod llvm;
pub mod llvm2;
