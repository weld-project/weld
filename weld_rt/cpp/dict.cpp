#include "runtime.h"
#include "assert.h"

struct simple_dict {
  void *data;
  int64_t size;
  int64_t capacity;
  bool full; // only local dicts should be marked as full
};

struct weld_dict {
  void *dicts; // one dict per thread plus a global dict
  int32_t key_size;
  int32_t val_size;
  int64_t max_local_bytes;
  // following two fields are used when finalizing dict (merging all locals into global)
  int32_t cur_local_dict;
  int64_t cur_slot_in_dict;
  bool finalized; // all keys have been moved to global
  pthread_mutex_t global_lock;
};

inline int32_t slot_size(weld_dict *wd) {
  return sizeof(int32_t) /* space for key hash */ +
    sizeof(uint8_t) /* space for `filled` field */ + wd->key_size + wd->val_size;
}

inline int32_t kv_size(weld_dict *wd, int32_t post_key_padding) {
  return wd->key_size + post_key_padding + wd->val_size;
}

inline void *slot_at_with_data(int64_t slot_offset, weld_dict *wd, void *data) {
  return (void *)((uint8_t *)data + slot_offset * slot_size(wd));
}

inline void *slot_at(int64_t slot_offset, weld_dict *wd, simple_dict *sd) {
  return slot_at_with_data(slot_offset, wd, sd->data);
}

inline void *key_at(void *slot) {
  return (void *)((uint8_t *)slot + sizeof(int32_t) + sizeof(uint8_t));
}

inline void *val_at(weld_dict *wd, void *slot) {
  return (void *)((uint8_t *)slot + sizeof(int32_t) + sizeof(uint8_t) + wd->key_size);
}

inline int32_t *hash_at(void *slot) {
  return (int32_t *)slot;
}

inline uint8_t *filled_at(void *slot) {
  return (uint8_t *)slot + sizeof(int32_t);
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

inline simple_dict *get_local_dict(weld_dict *wd) {
  return get_dict_at_index(wd, weld_rt_thread_id());
}

inline simple_dict *get_global_dict(weld_dict *wd) {
  return get_dict_at_index(wd, weld_rt_get_nworkers());
}

extern "C" void *weld_rt_dict_new(int32_t key_size, int32_t val_size, int64_t max_local_bytes, int64_t capacity) {
  weld_dict *wd = (weld_dict *)calloc(1, sizeof(weld_dict));
  wd->key_size = key_size;
  wd->val_size = val_size;
  wd->max_local_bytes = max_local_bytes;
  wd->dicts = weld_rt_new_merger(sizeof(simple_dict), weld_rt_get_nworkers() + 1);
  for (int32_t i = 0; i < weld_rt_get_nworkers() + 1; i++) {
    simple_dict *d = get_dict_at_index(wd, i);
    d->size = 0;
    d->capacity = capacity;
    d->data = calloc(capacity, slot_size(wd));
  }
  pthread_mutex_init(&wd->global_lock, NULL);
  return (void *)wd;
}

inline void *simple_dict_lookup(weld_dict *wd, simple_dict *sd, int32_t hash, void *key,
  bool collision_possible) {
  int64_t first_offset = hash % sd->capacity;
  for (int64_t i = 0; i < sd->capacity; i++) {
    // can do the bitwise and because capacity is always a power of two
    void *cur_slot = slot_at((first_offset + i) & (sd->capacity - 1), wd, sd);
    if (*filled_at(cur_slot)) {
      if (collision_possible && *hash_at(cur_slot) == hash &&
        memcmp(key, key_at(cur_slot), wd->key_size) == 0) {
        return cur_slot;
      }
    } else {
      *hash_at(cur_slot) = hash; // store hash here in case we end up filling this slot (no
      // problem if not)
      return cur_slot;
    }
  }
  // should never reach this
  return NULL;
}

inline void resize_dict(weld_dict *wd, simple_dict *sd) {
  void *old_data = sd->data;
  int64_t old_capacity = sd->capacity;
  sd->capacity *= 2;
  sd->data = calloc(sd->capacity, slot_size(wd));
  for (int64_t i = 0; i < old_capacity; i++) {
    void *old_slot = slot_at_with_data(i, wd, old_data);
    if (*filled_at(old_slot)) {
      // will never compare the keys when collision_possible = false so can pass NULL
      void *new_slot = simple_dict_lookup(wd, sd, *hash_at(old_slot), NULL, false);
      memcpy(new_slot, old_slot, slot_size(wd));
    }
  }
}

// Can only be doing a lookup to examine value if dict is already finalized. Otherwise must
// be for the purposes of merging.
extern "C" void *weld_rt_dict_lookup(void *d, int32_t hash, void *key) {
  weld_dict *wd = (weld_dict *)d;
  if (!wd->finalized) {
    simple_dict *sd = get_local_dict(wd);
    void *slot = simple_dict_lookup(wd, sd, hash, key, true);
    if (!sd->full || *filled_at(slot)) {
      return slot;
    }
  }
  if (!wd->finalized) {
    pthread_mutex_lock(&wd->global_lock);
  }
  simple_dict *global = get_global_dict(wd);
  return simple_dict_lookup(wd, global, hash, key, true);
}

extern "C" void weld_rt_dict_put(void *d, void *slot) {
  weld_dict *wd = (weld_dict *)d;
  uint8_t old_filled = *filled_at(slot);
  if (!old_filled) {
    *filled_at(slot) = 1;
    if (slot_in_local(wd, slot)) {
      simple_dict *local = get_local_dict(wd);
      local->size++;
      if (should_resize_dict_at_size(local->size, local)) {
        resize_dict(wd, local);
      } else if (should_resize_dict_at_size(local->size + 1, local) &&
        local->capacity * 2 > wd->max_local_bytes) {
        local->full = true;
      }
    } else {
      simple_dict *global = get_global_dict(wd);
      global->size++;
      if (should_resize_dict_at_size(global->size, global)) {
        resize_dict(wd, global);
      }
      if (!wd->finalized) {
        pthread_mutex_unlock(&wd->global_lock);
      }
    }
  } else if (!wd->finalized && !slot_in_local(wd, slot)) {
    pthread_mutex_unlock(&wd->global_lock);
  }
}

inline void advance_finalize_iterator(weld_dict *wd) {
  wd->cur_slot_in_dict++;
  if (wd->cur_slot_in_dict == get_dict_at_index(wd, wd->cur_local_dict)->capacity) {
    wd->cur_local_dict++;
    wd->cur_slot_in_dict = 0;
  }
}

extern "C" void *weld_rt_dict_finalize_next_local_slot(void *d) {
  weld_dict *wd = (weld_dict *)d;
  wd->finalized = true; // immediately mark as finalized ... we expect the client to keep
  // calling this method until it returns NULL, and only after this perform other operations
  // on the dictionary
  while (wd->cur_local_dict != weld_rt_get_nworkers()) {
    simple_dict *cur_dict = get_dict_at_index(wd, wd->cur_local_dict);
    void *next_slot = slot_at(wd->cur_slot_in_dict, wd, cur_dict);
    advance_finalize_iterator(wd);
    if (*filled_at(next_slot)) {
      return next_slot;
    }
  }
  return NULL;
}

extern "C" void *weld_rt_dict_finalize_global_slot_for_local(void *d, void *local_slot) {
  void *global_slot = weld_rt_dict_lookup((weld_dict *)d, *hash_at(local_slot), key_at(local_slot));
  return global_slot;
}

extern "C" void *weld_rt_dict_to_array(void *d, int32_t value_offset_in_struct) {
  weld_dict *wd = (weld_dict *)d;
  int32_t post_key_padding = value_offset_in_struct - wd->key_size;
  assert(wd->finalized);
  simple_dict *global = get_global_dict(wd);
  void *array = malloc(global->size * kv_size(wd, post_key_padding));
  int64_t next_arr_slot = 0;
  for (int64_t i = 0; i < global->capacity; i++) {
    void *cur_slot = slot_at(i, wd, global);
    if (*filled_at(cur_slot)) {
      memcpy((uint8_t *)array + next_arr_slot * kv_size(wd, post_key_padding), key_at(cur_slot),
        wd->key_size);
      memcpy((uint8_t *)array + next_arr_slot * kv_size(wd, post_key_padding) + wd->key_size +
        post_key_padding, val_at(wd, cur_slot), wd->val_size);
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
  for (int32_t i = 0; i < weld_rt_get_nworkers() + 1; i++) {
    simple_dict *d = get_dict_at_index(wd, i);
    free(d->data);
  }
  weld_rt_free_merger(wd->dicts);
  pthread_mutex_destroy(&wd->global_lock);
  free(wd);
}