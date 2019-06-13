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
#include <llvm/Target/TargetMachine.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>


#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <llvm-c/Types.h>
#include <llvm-c/Transforms/PassManagerBuilder.h>

#define BUFSZ 4096

using namespace std;
using namespace llvm;

/**
 * Returns a target triple for the JIT. 
 */
extern "C" const char *LLVMExtGetProcessTriple() {
  static char buf[BUFSZ];
  string triple = sys::getProcessTriple();
  strncpy(buf, triple.c_str(), BUFSZ);
  return buf;
}

/**
 * Returns the host CPU name.
 * The return value can be used as a value for the "target-cpu" attribute in LLVM.
 */
extern "C" const char *LLVMExtGetHostCPUName() {
  static char buf[BUFSZ];
  string host_name = sys::getHostCPUName();
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
  StringMap<bool> features;
  sys::getHostCPUFeatures(features);
  vector<string> features_present;
  vector<string> features_missing;
  for (auto it = features.begin(); it != features.end(); it++) {
    if (it->second) {
      features_present.push_back("+" + it->first().str());
    } else {
      features_missing.push_back("-" + it->first().str());
    }
  }

  std::sort(features_present.begin(), features_present.end());
  std::sort(features_missing.begin(), features_missing.end());

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


extern "C" void LLVMExtAddTargetPassConfig(LLVMTargetMachineRef target, LLVMPassManagerRef manager) {
  legacy::PassManagerBase *Passes = reinterpret_cast<legacy::PassManagerBase *>(manager);
  TargetMachine *TM = reinterpret_cast<TargetMachine *>(target);
  auto &LTM = static_cast<LLVMTargetMachine &>(*TM);
  Pass *TPC = LTM.createPassConfig(static_cast<PassManagerBase &>(*Passes));
  Passes->add(TPC);
}

extern "C" void LLVMExtAddTargetLibraryInfo(LLVMPassManagerRef manager){
  legacy::PassManagerBase *Passes = reinterpret_cast<legacy::PassManagerBase *>(manager);
  Triple triple = Triple(LLVMExtGetProcessTriple());
  TargetLibraryInfoWrapperPass *D = new TargetLibraryInfoWrapperPass(triple);
  Passes->add(D);
}


/** Disable or enable vectorization.
 *
 * The naming scheme of this function mirrors the LLVM `SetDisableUnrollLoops` function.
 */
extern "C" void LLVMExtPassManagerBuilderSetDisableVectorize(LLVMPassManagerBuilderRef P, unsigned i) {
  PassManagerBuilder *b = reinterpret_cast<PassManagerBuilder*>(P);
  b->LoopVectorize = (i == 0);
  b->SLPVectorize = (i == 0);
}
