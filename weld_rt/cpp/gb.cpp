// See gb-new.cpp for a re-implementation of this dictionary.
#if 0

#include "runtime.h"
#include <algorithm>

struct weld_arr {
  void *data;
  int64_t size;
};

struct weld_arr_growable {
  weld_arr a;
  int64_t capacity;
};

inline void resize_weld_arr(weld_arr_growable *arr, int32_t val_size, int64_t new_cap) {
  void *old_data = arr->a.data;
  arr->a.data = weld_run_malloc(weld_rt_get_run_id(), new_cap * val_size);
  memcpy(arr->a.data, old_data, arr->a.size * val_size);
  arr->capacity = new_cap;
  weld_run_free(weld_rt_get_run_id(), old_data);
}

inline void *el_at(weld_arr_growable *arr, int32_t val_size, int64_t i) {
  return (void *)((uint8_t *)arr->a.data + i * val_size);
}

void gb_merge_new_val(void *metadata, int32_t is_filled, void *dst, void *value) {
  weld_arr_growable *arr = (weld_arr_growable *)dst;
  int32_t val_size = *(int32_t *)metadata;
  if (is_filled) {
    if (arr->capacity == arr->a.size) {
      resize_weld_arr(arr, val_size, arr->capacity * 2);
    }
  } else {
    arr->capacity = 16;
    arr->a.data = weld_run_malloc(weld_rt_get_run_id(), arr->capacity * val_size);
    arr->a.size = 0;
  }
  memcpy(el_at(arr, val_size, arr->a.size), value, val_size);
  arr->a.size++;
}

void gb_merge_vals_finalize(void *metadata, int32_t is_filled, void *dst, void *value) {
  weld_arr_growable *local_arr = (weld_arr_growable *)value;
  weld_arr_growable *global_arr = (weld_arr_growable *)dst;
  int32_t val_size = *(int32_t *)metadata;
  if (is_filled) {
    if (local_arr->a.size + global_arr->a.size > global_arr->capacity) {
      resize_weld_arr(global_arr, val_size,
        std::max(global_arr->capacity * 2, local_arr->a.size + global_arr->a.size));
    }
  } else {
    global_arr->capacity = std::max((int64_t)16, local_arr->a.size);
    global_arr->a.data = weld_run_malloc(weld_rt_get_run_id(),
      global_arr->capacity * val_size);
    global_arr->a.size = 0;
  }
  memcpy(el_at(global_arr, val_size, global_arr->a.size), local_arr->a.data,
    val_size * local_arr->a.size);
  global_arr->a.size += local_arr->a.size;
  weld_run_free(weld_rt_get_run_id(), local_arr->a.data);
}

extern "C" void *weld_rt_gb_new(int32_t key_size, int32_t (*keys_eq)(void *, void *),
  int32_t val_size, int64_t max_local_bytes, int64_t capacity) {
  int32_t *v_size = (int32_t *)weld_run_malloc(weld_rt_get_run_id(), sizeof(int32_t));
  *v_size = val_size;
  return weld_rt_dict_new(key_size, keys_eq, &gb_merge_new_val, &gb_merge_vals_finalize,
    (void *)v_size, sizeof(weld_arr_growable), sizeof(weld_arr), max_local_bytes, capacity);
}

extern "C" void weld_rt_gb_merge(void *b, void *key, int32_t hash, void *value) {
  weld_rt_dict_merge(b, hash, key, value);
}

extern "C" void *weld_rt_gb_result(void *b) {
  weld_rt_dict_finalize(b);
  return b;
}

extern "C" void weld_rt_gb_free(void *gb) {}

#endif
