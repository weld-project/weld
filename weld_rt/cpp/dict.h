#ifndef _WELD_DICT_H_
#define _WELD_DICT_H_


// A callback which checks whether two keys are equal. Returns a non-zero value
// if the keys are equal.
//
// arg1: key, arg2: other_key
typedef int32_t (*KeyComparator)(void *, void *);

/**
 * @param metadata
 * @param filled
 * @param value to update.
 * @pram value to merge.
 */
typedef void (*MergeFn)(void *, int32_t, void *, void *);

extern "C" void *weld_rt_dict_new(int32_t key_size,
    KeyComparator keys_eq,
    MergeFn merge_fn,
    MergeFn finalize_merge_fn,
    void *metadata,
    int32_t val_size,
    int32_t packed_value_size,
    int64_t max_local_bytes,
    int64_t capacity);

extern "C" void weld_rt_dict_free(void *d);
extern "C" void *weld_rt_dict_lookup(void *d, int32_t hash, void *key);
extern "C" void weld_rt_dict_merge(void *d, int32_t hash, void *key, void *value);
extern "C" void weld_rt_dict_finalize(void *d);
extern "C" void *weld_rt_dict_to_array(void *d, int32_t value_offset, int32_t struct_size);
extern "C" int64_t weld_rt_dict_size(void *d);

#endif
