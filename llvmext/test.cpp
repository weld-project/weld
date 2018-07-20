// Simple test program that prints the results of the LLVM extension API.

#include <iostream>
#include "llvmext.h"

using namespace std;

int main() {
  cout << "Process Triple: " << LLVMExtGetProcessTriple() << endl;
  cout << "Host CPU Name:  " << LLVMExtGetHostCPUName() << endl;
  cout << "Host Features:  " << LLVMExtGetHostCPUFeatures() << endl;
}
