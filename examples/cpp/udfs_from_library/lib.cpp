// This file will be built as a dynamically loaded library, called
// libudf.so or libudf.dylib, to show use of weld_load_library in udfs.cpp.

#include <stdint.h>
#include <stdio.h>

extern "C" void add_five(int64_t *x, int64_t *result) {
    *result = *x + 5;
    printf("called add_five: %lld + 5 = %lld\n", *x, *result);
}
