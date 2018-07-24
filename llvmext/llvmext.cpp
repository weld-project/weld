/**
 * llvm_exts.cpp: Extensions for LLVM callable from Rust.
 * 
 * This file (built as a statically linked library) contains a number of LLVM
 * extensions callable from Rust that do not exist in `llvm_sys`.
 *
 * The library must be dynamically linked with the LLVM libraries.
 */
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>

#include <llvm/Support/Host.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/TargetLibraryInfo.h>

#include <llvm-c/Target.h>

#define BUFSZ 4096

using namespace std;

/**
 * Returns a target triple for the JIT. 
 */
extern "C" const char *LLVMExtGetProcessTriple() {
  static char buf[BUFSZ];
  string triple = llvm::sys::getProcessTriple();
  strncpy(buf, triple.c_str(), BUFSZ);
  return buf;
}

/**
 * Returns the host CPU name.
 * The return value can be used as a value for the "target-cpu" attribute in LLVM.
 */
extern "C" const char *LLVMExtGetHostCPUName() {
  static char buf[BUFSZ];
  string host_name = llvm::sys::getHostCPUName();
  strncpy(buf, host_name.c_str(), BUFSZ);
  return buf;
}

/**
 * Returns the target features of the current machine, formatted for use as an
 * attribute in LLVM IR.
 * The return value can be used as a value for the "target-features" attribute in LLVM.
 */
extern "C" const char *LLVMExtGetHostCPUFeatures() {
  static char buf[BUFSZ];
  llvm::StringMap<bool> features;
  llvm::sys::getHostCPUFeatures(features);
  vector<string> features_present;
  vector<string> features_missing;
  for (auto it = features.begin(); it != features.end(); it++) {
    if (it->second) {
      features_present.push_back("+" + it->first().str());
    } else {
      features_missing.push_back("-" + it->first().str());
    }
  }

  sort(features_present.begin(), features_present.end());
  sort(features_missing.begin(), features_missing.end());

  string result;
  for (auto it = features_present.begin(); it != features_present.end(); it++) {
    result += *it;
    result += ",";
  }
  for (auto it = features_missing.begin(); it != features_missing.end(); it++) {
    result += *it;
    result += ",";
  }
  result.pop_back();
  strncpy(buf, result.c_str(), BUFSZ);
  return buf;
}

extern "C" LLVMTargetLibraryInfoRef LLVMExtTargetLibraryInfo() {
  llvm::Triple triple = llvm::Triple(LLVMExtGetProcessTriple());
  // TODO this currently leaks!!
  llvm::TargetLibraryInfoImpl *P = new llvm::TargetLibraryInfoImpl(triple);
  llvm::TargetLibraryInfoImpl *X = const_cast<llvm::TargetLibraryInfoImpl*>(P);
  return reinterpret_cast<LLVMTargetLibraryInfoRef>(X);
}
