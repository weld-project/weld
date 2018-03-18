#ifndef _GROUPBUILDER_H_
#define _GROUPBUILDER_H_

#include "dict.h"

extern "C" void *weld_rt_gb_new(int32_t key_size,
    KeyComparator keys_eq,
    int32_t val_size,
    int64_t max_local_bytes,
    int64_t capacity);

extern "C" void weld_rt_gb_merge(void *b, void *key, int32_t hash, void *value);
extern "C" void *weld_rt_gb_result(void *b);
extern "C" void weld_rt_gb_free(void *gb);

#endif

