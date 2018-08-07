#ifndef _WELD_DICT_H_
#define _WELD_DICT_H_

#include <stdint.h>

#include "strt.h"

/** Comparator for two keys. Returns 0 if the keys are equal and 1 otherwise.
 * If the keys are equal, their hash values in the dictionary *must* match. It
 * is an error for the return value of this function to change for a given key
 * if the key is already in the dictionary.
 *
 * */
typedef int32_t (*KeyComparator)(void *, void *);

/**
 * Creates a new Weld dictionary.
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
extern "C" void *weld_st_dict_new(
    WeldRunHandleRef run,
		int32_t key_size,
		int32_t value_size,
    KeyComparator keys_eq,
    int64_t capacity);

/** Frees a dictionary created using `weld_rt_dict_new`. */
extern "C" void weld_rt_dict_free(WeldRunHandleRef, void *d);

/**
 * Returns the slot for the given hash/key. If the dictionary does not have an entry for hash/key
 * then its slot is initialized with the init_value and returned.
 * Returns NULL if no slot could be found or created.
 */
extern "C" void *weld_rt_upsert_slot(WeldRunHandleRef, void *d, int32_t hash, void *key, void *init_value);

/**
 * Allocates a new slot for the key if it does not exist.
 *
 * The value in the ne slot is uninitialized. This may resize the dictionary.
 */
extern "C" void *weld_st_dict_get_slot(
    WeldRunHandleRef run,
    void *d,
    void *key,
    int32_t hash);

/**
 * Returns the size of the dictionary.
 */
extern "C" int64_t weld_rt_dict_size(WeldRunHandleRef, void *d);

/**
 * Return an array of structs, where structs contain the key and value. This allocates
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
extern "C" void weld_rt_dict_tovec(WeldRunHandleRef,
    void *d,
    int32_t value_offset,
    int32_t struct_size,
    void *out);

/** Writes serialized bytes representing the dictionary `d` into `buf`.
 *
 * Returns the new offset to write at into the buffer.
 *
 *
 * PRE-REQUISITES:
 * `d`, the dictionary, must be finalized.
 * `buf` must be a Weld growable vec[i8].
 *
 */
extern "C" int64_t weld_rt_dict_serialize(
    WeldRunHandleRef run,
    void *d,
    void *buf,
    int32_t has_pointer,
    void* key_ser,
    void* val_ser);

#endif
