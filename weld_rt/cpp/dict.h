#ifndef _WELD_DICT_H_
#define _WELD_DICT_H_

extern "C" void *weld_rt_dict_lookup(void *d, int32_t hash, void *key);
extern "C" void weld_rt_dict_put(void *d, void *slot);
extern "C" void *weld_rt_dict_finalize_next_local_slot(void *d);
extern "C" void *weld_rt_dict_finalize_global_slot_for_local(void *d, void *local_slot);
extern "C" void *weld_rt_dict_to_array(void *d, int32_t value_offset_in_struct, int32_t struct_size); 
extern "C" int64_t weld_rt_dict_get_size(void *d);
extern "C" void weld_rt_dict_free(void *d);
extern "C" void *weld_rt_gb_new(int32_t key_size, int32_t (*keys_eq)(void *, void *),
  int32_t val_size, int64_t max_local_bytes, int64_t capacity);
extern "C" void weld_rt_gb_merge(void *b, void *key, int32_t hash, void *value);
extern "C" void *weld_rt_gb_result(void *b);
extern "C" void weld_rt_gb_free(void *gb);

#endif

