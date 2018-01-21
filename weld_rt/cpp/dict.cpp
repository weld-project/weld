#include "runtime.h"
#include "assert.h"
#include "stdio.h"

// power of 2
#define LOCK_GRANULARITY 16
#define MAX_LOCAL_PROBES 5
#define GLOBAL_BATCH_SIZE 128

struct simple_dict {
  void *data;
  volatile int64_t size;
  volatile int64_t capacity;
  bool full; // only local dicts should be marked as full
};

// per-thread buffer for global writes to reduce overhead of taking rw_lock
struct global_buffer {
  void *data;
  int32_t size;
};

struct weld_dict {
  void *dicts; // one dict per thread plus a global dict
  void *global_buffers;
  int32_t key_size;
  int32_t (*keys_eq)(void *, void *);
  void (*merge_new_val)(void *, int32_t, void *, void *);
  void (*merge_vals_finalize)(void *, int32_t, void *, void *);
  void *metadata; // passed into the above two functions
  int32_t val_size;
  int32_t to_array_true_val_size; // might discard some trailing part of the value when
  // converting to array (useful for groupbuilder)
  int64_t max_local_bytes;
  bool finalized; // multithreaded writes by the client are complete; all subsequent writes will be by a single
  // thread to the global table
  pthread_rwlock_t global_lock;
  int32_t n_workers; // save this here so we don't have to repeatedly call runtime for it
};

inline int32_t slot_size(weld_dict *wd) {
  return sizeof(uint8_t) /* space for `filled` field */ + wd->key_size + wd->val_size +
    sizeof(int32_t) /* space for key hash */ + sizeof(uint8_t) /* space for slot lock */;
}

inline void *slot_at_with_data(int64_t slot_offset, weld_dict *wd, void *data) {
  return (void *)((uint8_t *)data + slot_offset * slot_size(wd));
}

inline void *slot_at(int64_t slot_offset, weld_dict *wd, simple_dict *sd) {
  return slot_at_with_data(slot_offset, wd, sd->data);
}

inline void *key_at(void *slot) {
  return (void *)((uint8_t *)slot + sizeof(uint8_t));
}

inline void *val_at(weld_dict *wd, void *slot) {
  return (void *)((uint8_t *)slot + sizeof(uint8_t) + wd->key_size);
}

inline int32_t *hash_at(weld_dict *wd, void *slot) {
  return (int32_t *)((uint8_t *)slot + sizeof(uint8_t) + wd->key_size + wd->val_size);
}

inline uint8_t *filled_at(void *slot) {
  return (uint8_t *)slot;
}

inline uint8_t *lock_at(weld_dict *wd, void *slot) {
  return (uint8_t *)((uint8_t *)slot + sizeof(uint8_t) + wd->key_size + wd->val_size
    + sizeof(int32_t));
}

inline bool slot_in_local(weld_dict *wd, void *slot) {
  simple_dict *local = (simple_dict *)weld_rt_get_merger_at_index(wd->dicts, sizeof(simple_dict),
    weld_rt_thread_id());
  intptr_t slot_int = (intptr_t)slot;
  return slot_int >= (intptr_t)local->data &&
    slot_int < (intptr_t)((uint8_t *)local->data + slot_size(wd) * local->capacity);
}

inline bool should_resize_dict_at_size(int64_t size, simple_dict *sd) {
  // 70% full
  return size * 10 >= sd->capacity * 7;
}

inline simple_dict *get_dict_at_index(weld_dict *wd, int32_t i) {
  return (simple_dict *)weld_rt_get_merger_at_index(wd->dicts, sizeof(simple_dict), i);
}

inline global_buffer *get_global_buffer_at_index(weld_dict *wd, int32_t i) {
  return (global_buffer *)weld_rt_get_merger_at_index(wd->global_buffers, sizeof(global_buffer), i);
}

inline global_buffer *get_my_global_buffer(weld_dict *wd) {
  return get_global_buffer_at_index(wd, weld_rt_thread_id());
}

inline bool slot_in_global_buffer(weld_dict *wd, void *slot) {
  global_buffer *my_buf = get_my_global_buffer(wd);
  intptr_t slot_int = (intptr_t)slot;
  return slot_int >= (intptr_t)my_buf->data &&
    slot_int < (intptr_t)((uint8_t *)my_buf->data + slot_size(wd) * GLOBAL_BATCH_SIZE);
}

inline simple_dict *get_local_dict(weld_dict *wd) {
  return get_dict_at_index(wd, weld_rt_thread_id());
}

inline simple_dict *get_global_dict(weld_dict *wd) {
  return get_dict_at_index(wd, wd->n_workers);
}

extern "C" void *weld_rt_dict_new(int32_t key_size, int32_t (*keys_eq)(void *, void *),
  void (*merge_new_val)(void *, int32_t, void *, void *),
  void (*merge_vals_finalize)(void *, int32_t, void *, void *), void *metadata,
  int32_t val_size, int32_t to_array_true_val_size, int64_t max_local_bytes, int64_t capacity) {
  assert(capacity > 0 && (capacity & (capacity - 1)) == 0); // power of 2 check
  weld_dict *wd = (weld_dict *)weld_run_malloc(weld_rt_get_run_id(), sizeof(weld_dict));
  memset(wd, 0, sizeof(weld_dict));
  wd->key_size = key_size;
  wd->keys_eq = keys_eq;
  wd->merge_new_val = merge_new_val;
  wd->merge_vals_finalize = merge_vals_finalize;
  wd->metadata = metadata;
  wd->val_size = val_size;
  wd->to_array_true_val_size = to_array_true_val_size;
  wd->max_local_bytes = max_local_bytes;
  wd->n_workers = weld_rt_get_nworkers();
  if (wd->n_workers == 1) {
    // should always write directly to global (unbatched, unlocked) in this situation
    wd->max_local_bytes = 0;
    wd->finalized = true;
  }
  wd->dicts = weld_rt_new_merger(sizeof(simple_dict), wd->n_workers + 1);
  wd->global_buffers = weld_rt_new_merger(sizeof(global_buffer), wd->n_workers);
  for (int32_t i = 0; i < wd->n_workers + 1; i++) {
    if (i != wd->n_workers) {
      global_buffer *b = get_global_buffer_at_index(wd, i);
      b->data = weld_run_malloc(weld_rt_get_run_id(), GLOBAL_BATCH_SIZE * slot_size(wd));
      memset(b->data, 0, GLOBAL_BATCH_SIZE * slot_size(wd));
      b->size = 0;
    }
    simple_dict *d = get_dict_at_index(wd, i);
    d->size = 0;
    d->capacity = capacity;
    d->data = weld_run_malloc(weld_rt_get_run_id(), d->capacity * slot_size(wd));
    d->full = wd->max_local_bytes == 0;
    memset(d->data, 0, d->capacity * slot_size(wd));
  }
  pthread_rwlock_init(&wd->global_lock, NULL);
  return (void *)wd;
}

inline bool lockable_slot_idx(int64_t idx) {
  return (idx & (LOCK_GRANULARITY - 1)) == 0;
}

inline uint8_t *lock_for_slot(weld_dict *wd, simple_dict *sd, void *slot) {
  int64_t idx = ((intptr_t)slot - (intptr_t)sd->data) / slot_size(wd);
  return lock_at(wd, slot_at(idx & ~(LOCK_GRANULARITY - 1), wd, sd));
}

inline void *simple_dict_lookup(weld_dict *wd, simple_dict *sd, int32_t hash, void *key,
  bool match_possible, bool lock_global_slots, int64_t max_probes) {
  bool is_global = get_global_dict(wd) == sd;
  // can do the bitwise and because capacity is always a power of two
  int64_t first_offset = hash & (sd->capacity - 1);
  uint8_t *prev_lock = NULL;
  for (int64_t i = 0; i < max_probes; i++) {
    int64_t idx = (first_offset + i) & (sd->capacity - 1);
    void *cur_slot = slot_at(idx, wd, sd);
    if (!wd->finalized && is_global && lock_global_slots && (i == 0 || lockable_slot_idx(idx))) {
      if (prev_lock != NULL) {
        *prev_lock = 0;
      }
      prev_lock = lock_for_slot(wd, sd, cur_slot);
      while (!__sync_bool_compare_and_swap(prev_lock, 0, 1)) {}
    }
    if (*filled_at(cur_slot)) {
      if (match_possible && *hash_at(wd, cur_slot) == hash &&
        wd->keys_eq(key, key_at(cur_slot))) {
        return cur_slot;
      }
    } else {
      return cur_slot;
    }
  }
  if (prev_lock != NULL) {
    *prev_lock = 0;
  }
  return NULL;
}

inline void resize_dict(weld_dict *wd, simple_dict *sd, int64_t target_cap) {
  if (sd->capacity >= target_cap) {
    return;
  }
  void *old_data = sd->data;
  int64_t old_capacity = sd->capacity;
  while (sd->capacity < target_cap) {
    sd->capacity *= 2;
  }
  sd->data = weld_run_malloc(weld_rt_get_run_id(), sd->capacity * slot_size(wd));
  memset(sd->data, 0, sd->capacity * slot_size(wd));
  for (int64_t i = 0; i < old_capacity; i++) {
    void *old_slot = slot_at_with_data(i, wd, old_data);
    if (*filled_at(old_slot)) {
      // will never compare the keys when collision_possible = false so can pass NULL
      void *new_slot = simple_dict_lookup(wd, sd, *hash_at(wd, old_slot), NULL, false, false,
        sd->capacity);
      memcpy(new_slot, old_slot, slot_size(wd));
    }
  }
  weld_run_free(weld_rt_get_run_id(), old_data);
}

extern "C" void *weld_rt_dict_lookup(void *d, int32_t hash, void *key) {
  weld_dict *wd = (weld_dict *)d;
  if (!wd->finalized && wd->max_local_bytes > 0) {
    simple_dict *sd = get_local_dict(wd);
    void *slot = simple_dict_lookup(wd, sd, hash, key, true, true,
      sd->full ? MAX_LOCAL_PROBES : sd->capacity);
    if (!sd->full || (slot != NULL && *filled_at(slot))) {
      return slot;
    }
  }
  if (!wd->finalized) {
    global_buffer *buf = get_my_global_buffer(wd);
    void *buf_slot = slot_at_with_data(buf->size, wd, buf->data);
    return buf_slot;
  } else {
    simple_dict *global = get_global_dict(wd);
    return simple_dict_lookup(wd, global, hash, key, true, true, global->capacity);
  }
}

static void dict_put_single(weld_dict *wd, void *slot) {
  uint8_t old_filled = *filled_at(slot);
  if (!old_filled) {
    *filled_at(slot) = 1;
    if (slot_in_local(wd, slot)) {
      simple_dict *local = get_local_dict(wd);
      local->size++;
      if (should_resize_dict_at_size(local->size, local)) {
        resize_dict(wd, local, local->capacity * 2);
      } else if (should_resize_dict_at_size(local->size + 1, local) &&
        local->capacity * 2 * slot_size(wd) > wd->max_local_bytes) {
        local->full = true;
      }
    } else {
      simple_dict *global = get_global_dict(wd);
      if (!wd->finalized) {
        *lock_for_slot(wd, global, slot) = 0;
        __sync_fetch_and_add(&global->size, 1);
      } else{
        global->size++;
      }
      if (should_resize_dict_at_size(global->size, global)) {
        if (!wd->finalized) {
          pthread_rwlock_unlock(&wd->global_lock);
          pthread_rwlock_wrlock(&wd->global_lock);
        }
        if (should_resize_dict_at_size(global->size, global)) {
          resize_dict(wd, global, global->capacity * 2);
        }
        if (!wd->finalized) {
          pthread_rwlock_unlock(&wd->global_lock);
          pthread_rwlock_rdlock(&wd->global_lock);
        }
      }
    }
  } else if (!wd->finalized && !slot_in_local(wd, slot)) {
    *lock_for_slot(wd, get_global_dict(wd), slot) = 0;
  }
}

static void merge_into_slot(weld_dict *wd, void (*merge_fn)(void *, int32_t, void *, void *),
  void *slot, int32_t hash, void *key, void *value) {
  if (!*filled_at(slot)) {
    memcpy(key_at(slot), key, wd->key_size);
    *hash_at(wd, slot) = hash;
  }
  merge_fn(wd->metadata, *filled_at(slot), val_at(wd, slot), value);
}

static void drain_global_buffer(weld_dict *wd, global_buffer *b) {
  simple_dict *global = get_global_dict(wd);
  if (!wd->finalized) {
    pthread_rwlock_rdlock(&wd->global_lock);
  }
  for (int32_t i = 0; i < b->size; i++) {
    void *buf_slot = slot_at_with_data(i, wd, b->data);
    int32_t hash = *hash_at(wd, buf_slot);
    // need simple_dict_lookup here instead of weld_rt_dict_lookup because we don't want to waste
    // any time scanning the local dict
    void *global_slot = simple_dict_lookup(wd, global, hash, key_at(buf_slot), true, true, global->capacity);
    merge_into_slot(wd, wd->merge_new_val, global_slot, hash, key_at(buf_slot), val_at(wd, buf_slot));
    dict_put_single(wd, global_slot);
  }
  if (!wd->finalized) {
    pthread_rwlock_unlock(&wd->global_lock);
  }
  b->size = 0;
}

extern "C" void weld_rt_dict_merge(void *d, int32_t hash, void *key, void *value) {
  weld_dict *wd = (weld_dict *)d;
  void *slot = weld_rt_dict_lookup(d, hash, key);
  if (slot_in_global_buffer(wd, slot)) {
    *hash_at(wd, slot) = hash;
    memcpy(key_at(slot), key, wd->key_size);
    memcpy(val_at(wd, slot), value, wd->val_size);
    global_buffer *my_buf = get_my_global_buffer(wd);
    my_buf->size++;
    if (my_buf->size == GLOBAL_BATCH_SIZE) {
      drain_global_buffer(wd, my_buf);
    }
  } else {
    merge_into_slot(wd, wd->merge_new_val, slot, hash, key, value);
    dict_put_single(wd, slot);
  }
}

// Merge tuples from the local dicts into the global dict and finalize this dictionary.
extern "C" void weld_rt_dict_finalize(void *d) {
  weld_dict *wd = (weld_dict *)d;
  if (wd->finalized) {
    return;
  }
  wd->finalized = true;
  int64_t max_cap = 0;
  for (int32_t i = 0; i < wd->n_workers; i++) {
    simple_dict *sd = get_dict_at_index(wd, i);
    if (sd->capacity > max_cap) {
      max_cap = sd->capacity;
    }
  }
  // set global dict capacity to maximum of local dict capacities to reduce
  // number of resizings during the merging process
  resize_dict(wd, get_global_dict(wd), max_cap);
  for (int32_t i = 0; i < wd->n_workers; i++) {
    drain_global_buffer(wd, get_global_buffer_at_index(wd, i));
  }

  for (int32_t i = 0; i < wd->n_workers; i++) {
    simple_dict *cur_dict = get_dict_at_index(wd, i);
    for (int64_t j = 0; j < cur_dict->capacity; j++) {
      void *next_slot = slot_at(j, wd, cur_dict);
      if (*filled_at(next_slot)) {
        void *global_slot = weld_rt_dict_lookup(wd, *hash_at(wd, next_slot), key_at(next_slot));
        merge_into_slot(wd, wd->merge_vals_finalize, global_slot, *hash_at(wd, next_slot), key_at(next_slot),
          val_at(wd, next_slot));
        dict_put_single(wd, global_slot);
      }
    }
  }
}

extern "C" void *weld_rt_dict_to_array(void *d, int32_t value_offset_in_struct, int32_t struct_size) {
  weld_dict *wd = (weld_dict *)d;
  int32_t post_key_padding = value_offset_in_struct - wd->key_size;
  assert(wd->finalized);
  simple_dict *global = get_global_dict(wd);
  void *array = weld_run_malloc(weld_rt_get_run_id(),
    global->size * struct_size);
  int64_t next_arr_slot = 0;
  for (int64_t i = 0; i < global->capacity; i++) {
    void *cur_slot = slot_at(i, wd, global);
    if (*filled_at(cur_slot)) {
      memcpy((uint8_t *)array + next_arr_slot * struct_size,
        key_at(cur_slot), wd->key_size);
      memcpy((uint8_t *)array + next_arr_slot * struct_size +
        wd->key_size + post_key_padding, val_at(wd, cur_slot), wd->to_array_true_val_size);
      next_arr_slot++;
    }
  }
  assert(next_arr_slot == global->size);
  return array;
}

extern "C" int64_t weld_rt_dict_get_size(void *d) {
  weld_dict *wd = (weld_dict *)d;
  assert(wd->finalized);
  return get_global_dict(wd)->size;
}

extern "C" void weld_rt_dict_free(void *d) {
  weld_dict *wd = (weld_dict *)d;
  for (int32_t i = 0; i < wd->n_workers + 1; i++) {
    simple_dict *d = get_dict_at_index(wd, i);
    weld_run_free(weld_rt_get_run_id(), d->data);
  }
  weld_rt_free_merger(wd->dicts);
  pthread_rwlock_destroy(&wd->global_lock);
  if (wd->metadata != NULL) {
    weld_run_free(weld_rt_get_run_id(), wd->metadata);
  }
  weld_run_free(weld_rt_get_run_id(), wd);
}
