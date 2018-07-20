/**
 * Extensions to the LLVM C API.
 */
#ifndef _LLVM_EXT_H_
#define _LLVM_EXT_H_

/**
 * Returns a target triple for the current process. 
 * Suitable for use with a JIT.
 */
extern "C" const char *LLVMExtGetProcessTriple(); 

/**
 * Returns the host CPU name.
 * The return value can be used as a value for the "target-cpu" attribute in LLVM.
 */
extern "C" const char *LLVMExtGetHostCPUName();

/**
 * Returns the target features of the current machine, formatted for use as an
 * attribute in LLVM IR.
 * The return value can be used as a value for the "target-features" attribute in LLVM.
 */
extern "C" const char *LLVMExtGetHostCPUFeatures(); 

#endif
