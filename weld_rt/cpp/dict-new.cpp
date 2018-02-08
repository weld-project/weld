#include "runtime.h"

// The dictionary API.
#include "dict.h"

#include <algorithm>
#include <assert.h>

// The number of slots to lock at once.
const int LOCK_GRANULARITY = 16;

// Resize threshold. Resizes when RESIZE_THRES * 10 percent of slots are full.
const int RESIZE_THRES = 7;

// A callback which checks whether two keys are equal. Returns a non-zero value
// if the keys are equal.
typedef int32_t (*KeyComparator)(void *, void *);

// A single slot. The header is followed by sizeof(key_size) + sizeof(value_size) bytes.
// The header + the key and value define a full slot.
struct Slot {
  public:

    // Slot metadata.
    struct SlotHeader {
      int32_t hash;
      bool filled;
      uint8_t lockvar;
    } header;

    // Returns the key, which immediately follows the header.
    inline void *key() {
      return reinterpret_cast<uint8_t *>(this) + sizeof(header);
    }

    // Returns the value, which follows the header, and then `key_size` bytes.
    inline void *value(size_t key_size) {
      return reinterpret_cast<uint8_t *>(this) + sizeof(header) + key_size;
    }

    // Lock a slot.
    inline void acquire_access() {
      while (!__sync_bool_compare_and_swap(&header.lockvar, 0, 1));
    }

    // Unlock a slot.
    inline void release_access() {
      header.lockvar = 0;
    }
};

// An internal dictionary which provides (optional) locked access to its slots.
class InternalDict {

  public:

    int64_t key_size;
    int64_t value_size;
    int64_t full_watermark;
    KeyComparator keys_eq;

  private:

    // An array of Slots.
    void *_data;
    volatile int64_t _size;
    volatile int64_t _capacity;
    bool _full;

    // Global lock for dictionaries accessed concurrently.
    pthread_rwlock_t _global_lock;

    inline size_t slot_size() {
      return sizeof(Slot::SlotHeader) + key_size + value_size;
    }

    /** Returns true if, with the configured locking granularity, the lock at the given
     * slot index should be used to lock a region of the hash table.
     *
     * As an example, if the LOCK_GRANULARITY is 4, only every 4th lock is utilized. Slots at
     * indices i where i % 4 != 0 use the slot at index (i / 4 * 4).
     *
     * */
    inline bool lockable_index(int64_t index) {
      return (index & (LOCK_GRANULARITY - 1)) == 0;
    }

    /** Given a slot, returns the Slot which holds the lock for the slot. If the locking
     * granularity is one slot, this will return the argument */
    inline Slot *lock_for_slot(Slot *slot) {
      // Compute the locked slot index, and then return the slot at the index
      int64_t index = ((intptr_t)slot - (intptr_t)_data) / slot_size();
      index &= ~(LOCK_GRANULARITY - 1);

      return slot_at_index(index);
    }

  public:

    void init_internal(int64_t a_key_size, int64_t a_value_size, KeyComparator a_keys_eq, int64_t a_capacity, int64_t a_full_watermark) {

      key_size = a_key_size;
      value_size = a_value_size;
      keys_eq = a_keys_eq;
      full_watermark = a_full_watermark;

      _capacity = a_capacity;
      _size = 0;

      // Power of 2 check.
      assert(_capacity > 0 && (_capacity & (_capacity - 1)) == 0);
      size_t data_size = _capacity * slot_size();

      _data = weld_run_malloc(weld_rt_get_run_id(), data_size);
      memset(_data, 0, data_size);
      _full = full_watermark == 0;

      pthread_rwlock_init(&_global_lock, NULL);
    }

    void free_internal() {
      pthread_rwlock_destroy(&_global_lock);
      weld_run_free(weld_rt_get_run_id(), _data);
    }

    int64_t size() {
      return _size;
    }

    int64_t capacity() {
      return _capacity;
    }

    bool full() {
      return _full;
    }

    // Returns true if this dictionary contains the slot, by checking whether the slot
    // address lies in the dictionary's allocated buffer.
    bool contains_slot(Slot *s) {
      intptr_t start = (intptr_t)_data;
      intptr_t end = (intptr_t)((uint8_t *)_data) + slot_size() * _capacity;
      intptr_t slot = (intptr_t)s;
      return slot >= start && slot < end;
    }

    /** Returns the slot at the given index. */
    inline Slot *slot_at_index(long index) {
      return (Slot *)(((uint8_t *)_data) + slot_size() * index);
    }

    /** Get a slot for a given hash and key, optionally locking it for concurrent access.
    */
    Slot *get_slot(int32_t hash, void *key, bool lock, bool check_key) {

      // Protects against concurrent resizing.
      if (lock) {
        pthread_rwlock_rdlock(&_global_lock);
      }

      int64_t first_offset = hash & (_capacity - 1);
      Slot *locked_slot = NULL;

      // Linear probing.
      for (long i = 0; i < _capacity; i++) {
        long index = (first_offset + 1) & (_capacity - 1);
        Slot *slot = slot_at_index(index);

        // Lock the slot to disable concurrent access on the LOCK_GRANULARITY buckets we probe.
        if (lock && (i == 0 || lockable_index(i))) {
          if (locked_slot != NULL) {
            locked_slot->release_access(); 
          }
          slot->acquire_access();
          locked_slot = slot;
        }

        if (slot->header.filled && check_key && slot->header.hash == hash && keys_eq(key, slot->key())) {
          assert(locked_slot);
          return slot;
        } else {
          assert(locked_slot);
          slot->header.hash = hash;
          return slot;
        }
      }

      if (locked_slot) {
        locked_slot->release_access();
      }
      return NULL;
    }

    /** Puts a value in a slot, marking it as filled and releasing the lock on it if one exists. */
    bool put_slot(Slot *slot, bool locked) {
      bool was_filled = slot->header.filled;
      slot->header.filled = 1;

      // If the slot was just updated, we don't have to worry about resizing etc.
      // Unlock the slot (if its locked) and return.
      if (was_filled) {
        if (locked) {
          Slot *locked_slot = lock_for_slot(slot);
          locked_slot->release_access();
          pthread_rwlock_unlock(&_global_lock);
        }
        return _full;
      }

      if (locked) {
        Slot *locked_slot = lock_for_slot(slot);
        locked_slot->release_access();
        __sync_fetch_and_add(&_size, 1);

        // If we need to resize, upgrade the read lock to a write lock.
        if (should_resize(_size)) {
          pthread_rwlock_unlock(&_global_lock);
          pthread_rwlock_wrlock(&_global_lock);

          if (should_resize(_size)) {
            resize();
          }
        }
        // Unlock the Read lock if we didn't resize, or the write lock if we did.
        pthread_rwlock_unlock(&_global_lock);
      } else {
        _size++;
        if (should_resize(_size)) {
          resize();
        } else if (should_resize(_size + 1) && _capacity * 2 * slot_size() > full_watermark) {
          _full = true;
        }
      }
      return _full;
    }

    /** Returns whether this dictionary should be resized when it reaches `new_size`. */
    inline bool should_resize(size_t new_size) {
      return new_size * 10 >= _capacity * RESIZE_THRES; 
    }

    /** Resizes the dictionary. */
    void resize() {
      const size_t sz = slot_size();
      InternalDict resized;
      resized._size = _size;
      resized._capacity = _capacity * 2;
      resized._data = weld_run_malloc(weld_rt_get_run_id(), resized._capacity * sz);
      memset(resized._data, 0, resized._capacity * sz);

      for (int i = 0; i < _capacity; i++) {
        Slot *old_slot = slot_at_index(i);
        if (old_slot->header.filled) {
          Slot *new_slot = resized.get_slot(old_slot->header.hash, NULL, false, false);
          assert(new_slot != NULL);
          memcpy(new_slot, old_slot, sz);
        }
      }

      // Free the old data.
      weld_run_free(weld_rt_get_run_id(), _data);

      _data = resized._data;
      _size = resized._size;
      _capacity = resized._capacity;
      _full = false;
    }
};

/** A multi-threaded dictionary used by the Weld runtime. */
class WeldDict {

  private:

    // The contiguous buffer of dictionaries. If there are N workers, there are
    // N + 1 dictionaries, where index i holds the dictionary for thread ID i,
    // and dictionary N+1 is the global dictionary.
    void *_dicts;

    int32_t workers;

    struct WeldDictFinalizeIterator {
      int32_t current_finalize_dict_index;
      int32_t current_finalize_slot_index;
    } finalize_iterator;

  public:

    bool finalized;

  public:

    /** Get the dictionary at the given index.
     *
     * @i the index.
     * @return the dictionary the index.
     */
    inline InternalDict *dict_at_index(int32_t i) {
      return (InternalDict *)weld_rt_get_merger_at_index(_dicts, sizeof(InternalDict), i);
    }

    /** Get the dictionary for the calling thread. */
    inline InternalDict *local() {
      return dict_at_index(weld_rt_thread_id());
    }

    /** Get the global dictionary. */
    inline InternalDict *global() {
      return dict_at_index(workers);
    }
  
    void init_welddict(int32_t key_size, int32_t value_size,
        KeyComparator keys_eq,
        int64_t capacity,
        int64_t full_watermark) {

      workers = weld_rt_get_nworkers();

        _dicts = weld_rt_new_merger(sizeof(InternalDict), workers + 1);
        for (int i = 0; i < workers + 1; i++) {
          InternalDict *dict = dict_at_index(i);
          dict->init_internal(key_size, value_size,  keys_eq, capacity, full_watermark);
        }

        finalized = (full_watermark == 0);
        memset(&finalize_iterator, 0, sizeof(finalize_iterator));
      }

    void free_welddict() {
      for (int i = 0; i < workers + 1; i++) {
        InternalDict *dict = dict_at_index(i);
        dict->free_internal();
      }
      weld_rt_free_merger(_dicts);
    }

    // Finalization.

    void finalize_begin() {
      assert(!finalized);

      finalized = true;
      memset(&finalize_iterator, 0, sizeof(finalize_iterator));
    }

    Slot *finalize_next() {
      assert(finalized);

      while (finalize_iterator.current_finalize_dict_index != workers) {
        InternalDict *dict = dict_at_index(finalize_iterator.current_finalize_dict_index);
        Slot *slot = dict->slot_at_index(finalize_iterator.current_finalize_slot_index);

        finalize_iterator.current_finalize_slot_index++; 
        // Finished this dictionary - move on to the next one.
        if (finalize_iterator.current_finalize_slot_index >= dict->capacity()) {
          finalize_iterator.current_finalize_dict_index++; 
          finalize_iterator.current_finalize_slot_index = 0; 
        }

        if (slot->header.filled) {
          return slot;
        }
      }
      return NULL;
    }

    void finalize_commit() {
      assert(finalized);

    }

    // Serialization. This currently requires that the dictionary be finalized.
};

// The dictionary API.

///////////////////////////////////////////////////////
//
//                  New and Free.
//
///////////////////////////////////////////////////////

extern "C" void *weld_rt_dict_new(int32_t key_size,
    KeyComparator keys_eq,
    int32_t val_size,
    int32_t to_array_true_val_size,
    int64_t max_local_bytes,
    int64_t capacity) {

  WeldDict *wd = (WeldDict *)weld_run_malloc(weld_rt_get_run_id(), sizeof(WeldDict));
  wd->init_welddict(key_size, val_size, keys_eq, capacity, max_local_bytes);
  return (void *)wd;
}

extern "C" void weld_rt_dict_free(void *d) {
  WeldDict *wd = (WeldDict *)d;
  wd->free_welddict();
  weld_run_free(weld_rt_get_run_id(), wd);
}

///////////////////////////////////////////////////////
//
//                Basic Get and Put
//
///////////////////////////////////////////////////////

extern "C" void *weld_rt_dict_lookup(void *d, int32_t hash, void *key) {
  WeldDict *wd = (WeldDict *)d;
  if (!wd->finalized) {
    InternalDict *dict = wd->local();
    Slot *slot = dict->get_slot(hash, key, false, true);

    // Use the slot in the local dictionary if (a) the dictionary is not full or (b) the slot is
    // already occupied and just needs to be updated with a new value for the key.
    if (!dict->full() || (slot != NULL && slot->header.filled)) {
      return (void *)slot;
    }
  }

  InternalDict *dict = wd->global();

  // If the dictionary is not finalized, lock it. Otherwise we are guaranteeing that
  // we will only do a read.
  bool lock = !wd->finalized;
  Slot *s = dict->get_slot(hash, key, lock, true);
  assert(s);

  return s;
}

extern "C" void weld_rt_dict_put(void *d, void *s) {
  WeldDict *wd = (WeldDict *)d;
  Slot *slot = (Slot *)s;
  InternalDict *dict = wd->local();

  if (dict->contains_slot(slot)) {
    dict->put_slot(slot, false);
    return;
  }

  dict = wd->global();
  assert(dict->contains_slot(slot));

  // TODO if we are doing a put, this should always be true?
  bool lock = !wd->finalized;
  dict->put_slot(slot, lock);
}


///////////////////////////////////////////////////////
//
//                    Finalization
//
///////////////////////////////////////////////////////


// Begin the finalization procedure.
extern "C" void weld_rt_dict_finalize_begin(void *d) {
  WeldDict *wd = (WeldDict *)d;
  wd->finalize_begin();
}

// Return the next slot to finalize.
extern "C" void *weld_rt_dict_finalize_next(void *d) {
  WeldDict *wd = (WeldDict *)d;
  return wd->finalize_next();
}
