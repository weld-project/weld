#ifndef _WELD_DICT_H_
#define _WELD_DICT_H_

#include <stdint.h>

/** Comparator for two keys. Returns 0 if the keys are equal and 1 otherwise.
 * If the keys are equal, their hash values in the dictionary *must* match. It
 * is an error for the return value of this function to change for a given key
 * if the key is already in the dictionary.
 *
 * */
typedef int32_t (*KeyComparator)(void *, void *);

/** A merge function for merging a new value with an existing one in the dictionary.
 *
 * @param metadata that the merge function can use.
 * @param filled specifies whether a value was in this slot already.
 * @param value to updated
 * @pram value to merge.
 */
typedef void (*MergeFn)(void *, int32_t, void *, void *);

/** Creates a new Weld dictionary.
 *
 * @param key_size the size of the key in bytes.
 * @param keys_eq the comparator the check if two keys are equal
 * @param merge_fn specifies how a value is merged into the dictionary with an existing value.
 * @param finalize merge_fn specifies how a value is merged into the dictionary
 * with an existing value, during finalization.
 * @param metadata a pointer passed to the merge functions.
 * @param val_size the size of the value in bytes.
 * @param packed_value_size the size of the value when it is being copied into a KV-struct.
 * This can be used to trim unneeded bytes in the value.
 * @param max_local_bytes the maximum size of a *single* thread-local dictionary before
 * switching to performing writes in a global dictionary.
 * @param capacity the initial capacity of the local dictionaries.
 *
 * @return a handle to the created dictionary.
 */
extern "C" void *weld_rt_dict_new(int32_t key_size,
    KeyComparator keys_eq,
    MergeFn merge_fn,
    MergeFn finalize_merge_fn,
    void *metadata,
    int32_t val_size,
    int32_t packed_value_size,
    int64_t max_local_bytes,
    int64_t capacity);

/* Same as `weld_rt_dict_new` above, but always initialize the dictionary as
 * finalized (this allows multi-threaded writes, but still allocates space for
 * multi-threading in case the dictionary is converted into a builder).
 */
extern "C" void *weld_rt_dict_new_finalized(int32_t key_size,
    KeyComparator keys_eq,
    MergeFn merge_fn,
    MergeFn finalize_merge_fn,
    void *metadata,
    int32_t val_size,
    int32_t packed_value_size,
    int64_t max_local_bytes,
    int64_t capacity);

/** Frees a dictionary created using `weld_rt_dict_new`. */
extern "C" void weld_rt_dict_free(void *d);

/** Lookup a value in a dictionary given a key and its hash. */
extern "C" void *weld_rt_dict_lookup(void *d, int32_t hash, void *key);

/** Merge a value into a dictionary given a key, its hash, and a corresponding value. */
extern "C" void weld_rt_dict_merge(void *d, int32_t hash, void *key, void *value);

/** Finalize the dictionary, transitioning from "writes-only" mode to "read-only" mode. */
extern "C" void weld_rt_dict_finalize(void *d);

/** Return an array of structs, where structs contain the key and value. This allocates
 * memory that can be freed with `weld_run_free`.
 *
 * @param d the dictionary.
 * @param value_offset the offset in the KV-struct of the value. This should be greater than or equal to
 * the key size, to specify padding.
 * @param struct_size the total struct size. This is used to add padding after the value.
 *
 * @return a continguous array of KV-structs. The number of structs created can be retrieved using
 * `weld_rt_dict_size`.
 *
 * PRE-REQUISITES: The dictionary must be finalized.
 */
extern "C" void *weld_rt_dict_to_array(void *d, int32_t value_offset, int32_t struct_size);

/** Returns the size of the dictionary.
 *
 * PRE-REQUISITES: The dictionary must be finalized.
 */
extern "C" int64_t weld_rt_dict_size(void *d);

/** Writes serialized bytes representing the dictionary `d` into `buf`.
 *
 *
 * PRE-REQUISITES:
 * `d`, the dictionary, must be finalized.
 * `buf` must be a Weld growable vec[i8].
 *
 */
extern "C" void weld_rt_dict_serialize(void *d,
    void *buf,
    int32_t has_pointer,
    void* key_ser,
    void* val_ser);

#endif
