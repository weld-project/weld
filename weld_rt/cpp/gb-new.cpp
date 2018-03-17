#include "runtime.h"
#include "dict.h"

#include <algorithm>

// The initial capacity of a growable vector.
const int INITIAL_CAPACITY = 16;

// The canonical layout for a weld vector.
class WeldVec {
public:
  void *data;
  int64_t size;
};

// A growable weld vector, which is a WeldVec with an additional capacity field.
class GrowableWeldVec {

  public:

    WeldVec wv;
    int64_t capacity;

  public:

    void resize(int32_t value_size, int64_t new_capacity) {
      wv.data = weld_run_realloc(weld_rt_get_run_id(), wv.data, new_capacity * value_size);
      capacity = new_capacity;
    }

    void *at(int32_t val_size, int64_t i) {
      return (void *)((uint8_t *)wv.data + i * val_size);
    }
};

// The merge function for the dictionary holding the groupbuilder vectors.
void _groupbuilder_merge_fn(void *metadata, int32_t is_filled, void *dst, void *value) {

  GrowableWeldVec *v = (GrowableWeldVec *)dst;
  int32_t val_size = *(int32_t *)metadata;

  if (is_filled && v->capacity == v->wv.size) {
      v->resize(val_size, v->capacity * 2);
  }
  
  if (!is_filled) {
    v->capacity = INITIAL_CAPACITY;
    v->wv.data = weld_run_malloc(weld_rt_get_run_id(), v->capacity * val_size);
    v->wv.size = 0;
  }
  memcpy(v->at(val_size, v->wv.size), value, val_size);
  v->wv.size++;
}


// The finalize merge function for the dictionary holding the groupbuilder vectors.
void _groupbuilder_finalize_merge_fn(void *metadata, int32_t is_filled, void *dst, void *value) {

  GrowableWeldVec *local_vec = (GrowableWeldVec *)value;
  GrowableWeldVec *global_vec = (GrowableWeldVec *)dst;
  int32_t val_size = *(int32_t *)metadata;

  if (is_filled) {
    if (local_vec->wv.size + global_vec->wv.size > global_vec->capacity) {
      global_vec->resize(val_size,
        std::max(global_vec->capacity * 2, local_vec->wv.size + global_vec->wv.size));
    }
  } else {
    global_vec->capacity = std::max((int64_t)INITIAL_CAPACITY, local_vec->wv.size);
    global_vec->wv.data = weld_run_malloc(weld_rt_get_run_id(), global_vec->capacity * val_size);
    global_vec->wv.size = 0;
  }

  memcpy(global_vec->at(val_size, global_vec->wv.size),
      local_vec->wv.data,
      val_size * local_vec->wv.size);

  global_vec->wv.size += local_vec->wv.size;
  weld_run_free(weld_rt_get_run_id(), local_vec->wv.data);
}

extern "C" void *weld_rt_gb_new(int32_t key_size, KeyComparator keys_eq,
  int32_t val_size, int64_t max_local_bytes, int64_t capacity) {

  int32_t *v_size = (int32_t *)weld_run_malloc(weld_rt_get_run_id(), sizeof(int32_t));
  *v_size = val_size;

  return weld_rt_dict_new(key_size,
      keys_eq,
      &_groupbuilder_merge_fn,
      &_groupbuilder_finalize_merge_fn,
      (void *)v_size,
      sizeof(GrowableWeldVec),
      sizeof(WeldVec),
      max_local_bytes,
      capacity);
}

extern "C" void weld_rt_gb_merge(void *b, void *key, int32_t hash, void *value) {
  weld_rt_dict_merge(b, hash, key, value);
}

extern "C" void *weld_rt_gb_result(void *b) {
  weld_rt_dict_finalize(b);
  return b;
}

// TODO need to free the metadata...
extern "C" void weld_rt_gb_free(void *gb) {
  weld_rt_dict_free(gb);
}
