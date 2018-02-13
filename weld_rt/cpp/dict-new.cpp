#include "runtime.h"
#include "dict.h"

#include <algorithm>
#include <assert.h>

// The number of slots to lock at once.
const int LOCK_GRANULARITY = 16;

// Resize threshold. Resizes when RESIZE_THRES * 10 percent of slots are full.
const int RESIZE_THRES = 7;

// Size of the global buffer, in # of slots.
const int GLOBAL_BATCH_SIZE = 128; 

// Specifies what kind of locking to use. Basically a boolean, but its here for additional
// type safety enforced by the compiler.
typedef enum {
  NO_LOCKING = 0,
  ACQUIRE = 1,
} LockMode;

// A single slot. The header is followed by sizeof(key_size) +
// sizeof(value_size) bytes. The header + the key and value define a full slot.
struct Slot {
public:
  // Slot metadata.
  struct SlotHeader {
    int32_t hash;
    uint8_t filled;
    uint8_t lockvar; // locks the slot with a CAS.
  } header;

  // Returns the key, which immediately follows the header.
  inline void *key() {
    return reinterpret_cast<uint8_t *>(this) + sizeof(header);
  }

  // Returns the value, which follows the header, and then `key_size` bytes.
  inline void *value(size_t key_size) {
    return reinterpret_cast<uint8_t *>(this) + sizeof(header) + key_size;
  }

  // Updates the slot using a merge function.
  void update(MergeFn merge,
      void *metadata,
      int32_t a_hash,
      void *a_key, int64_t key_size,
      void *a_val, int64_t val_size) {

    if (!header.filled) {
      memcpy(key(), a_key, key_size);
      header.hash = a_hash;
    }

    // Either use a merge function if one is provided, or replace the exsting key.
    if (merge) {
      merge(metadata, (int32_t)header.filled, value(key_size), a_val);
    } else {
      memcpy(value(key_size), a_val, val_size);
    }
  }

  // Lock a slot.
  inline void acquire_access() {
    while (!__sync_bool_compare_and_swap(&header.lockvar, 0, 1))
      ;
  }

  // Unlock a slot.
  inline void release_access() { header.lockvar = 0; }
};

// An internal dictionary which provides (optional) locked access to its slots.
class InternalDict {

public:
  const int64_t key_size;
  const int64_t value_size;

  // Measured in bytes, designating the size of the buffer. If the
  // full_watermark is reached, the dictionary is marked full.
  const int64_t full_watermark;

  // Compares the key.
  const KeyComparator keys_eq;

private:

  // An array of Slots.
  void *_data;
  volatile int64_t _size;
  volatile int64_t _capacity;
  bool _full;

  // Global lock for dictionaries accessed concurrently.
  pthread_rwlock_t _global_lock;

  /** Returns true if, with the configured locking granularity, the lock at the
   * given slot index should be used to lock a region of the hash table.
   *
   * As an example, if the LOCK_GRANULARITY is 4, only every 4th lock is
   * utilized. Slots at indices i where i % 4 != 0 use the slot at index (i / 4
   * * 4).
   *
   * */
  inline bool lockable_index(int64_t index) {
    return (index & (LOCK_GRANULARITY - 1)) == 0;
  }

  /** Given a slot, returns the Slot which holds the lock for the slot. If the
   * locking granularity is one slot, this will return the argument */
  inline Slot *lock_for_slot(Slot *slot) {
    // Compute the locked slot index, and then return the slot at the index
    int64_t index = ((intptr_t)slot - (intptr_t)_data) / slot_size();
    index &= ~(LOCK_GRANULARITY - 1);
    return slot_at_index(index);
  }

public:
  // Default constructor.
  InternalDict()
      : key_size(0), value_size(0), full_watermark(0), keys_eq(0), _size(0),
        _capacity(0) {}

  InternalDict(int64_t a_key_size, int64_t a_value_size,
               KeyComparator a_keys_eq, int64_t a_capacity,
               int64_t a_full_watermark)
      : key_size(a_key_size), value_size(a_value_size),
        full_watermark(a_full_watermark), keys_eq(a_keys_eq), _size(0),
        _capacity(a_capacity) {

    // Power of 2 check.
    assert(_capacity > 0 && (_capacity & (_capacity - 1)) == 0);
    size_t data_size = _capacity * slot_size();

    _data = weld_run_malloc(weld_rt_get_run_id(), data_size);
    memset(_data, 0, data_size);
    _full = full_watermark == 0;

    pthread_rwlock_init(&_global_lock, NULL);
  }

  ~InternalDict() {
    pthread_rwlock_destroy(&_global_lock);
    weld_run_free(weld_rt_get_run_id(), _data);
  }

  inline size_t slot_size() {
    return sizeof(Slot::SlotHeader) + key_size + value_size;
  }

  int64_t size() { return _size; }

  int64_t capacity() { return _capacity; }

  bool full() { return _full; }

  void *data() { return _data; }

  void read_lock() {
      pthread_rwlock_rdlock(&_global_lock);
  }

  void unlock() {
      pthread_rwlock_unlock(&_global_lock);
  }

  /** Returns true if this dictionary contains the slot, by checking whether the
   * slot address lies in the dictionary's allocated buffer. */
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

  /** Get a slot for a given hash and key, optionally locking it for concurrent
   * access on a slot.
   *
   * Prerequisites: If this is a concurrently accessed dictionary, call lock() first.
   *
   */
  Slot *get_slot(int32_t hash, void *key, LockMode mode, bool check_key) {

    int64_t first_offset = hash & (_capacity - 1);
    Slot *locked_slot = NULL;

    // Linear probing.
    for (long i = 0; i < _capacity; i++) {
      long index = (first_offset + 1) & (_capacity - 1);
      Slot *slot = slot_at_index(index);

      // Lock the slot to disable concurrent access on the LOCK_GRANULARITY
      // buckets we probe.
      if (mode == ACQUIRE && (i == 0 || lockable_index(i))) {
        if (locked_slot != NULL) {
          locked_slot->release_access();
        }
        slot->acquire_access();
        locked_slot = slot;
      }

      if (!slot->header.filled) {
        if (mode == ACQUIRE) {
          assert(locked_slot);
        }
        return slot;
      }

      // Slot is filled - check it.
      if (check_key && slot->header.hash == hash && keys_eq(key, slot->key())) {
        if (mode == ACQUIRE) {
          assert(locked_slot);
        }
        return slot;
      }
    }

    if (locked_slot) {
      locked_slot->release_access();
    }
    return NULL;
  }

  /** Puts a value in a slot, marking it as filled. The caller should have retrieved slot
   * using get_slot, and locked the dictionary using read_lock. */
  bool put_slot(Slot *slot, LockMode mode) {
    bool was_filled = slot->header.filled;

    // If the slot was just updated, we don't have to worry about resizing etc.
    // Unlock the slot (if its locked) and return.
    if (was_filled) {
      if (mode == ACQUIRE) {
        Slot *locked_slot = lock_for_slot(slot);
        locked_slot->release_access();
      }
      return _full;
    }

    // This slot is now filled.
    slot->header.filled = 1;

    if (mode == ACQUIRE) {
      Slot *locked_slot = lock_for_slot(slot);
      locked_slot->release_access();
      __sync_fetch_and_add(&_size, 1);

      // If we need to resize, upgrade the read lock to a write lock.
      if (should_resize(_size)) {

        unlock();
        pthread_rwlock_wrlock(&_global_lock);

        // In case someone resized before we got the lock.
        if (should_resize(_size)) {
          resize(_capacity * 2);
        }

        // Downgrade back to a read lock.
        unlock();
        read_lock();
      }
    } else {
      _size++;
      if (should_resize(_size)) {
        resize(_capacity * 2);
      } else if (should_resize(_size + 1) &&
                 _capacity * 2 * slot_size() > full_watermark) {
        // The local dictionary is full.
        _full = true;
      }
    }
    return _full;
  }

  /** Returns whether this dictionary should be resized when it reaches
   * `new_size`. */
  inline bool should_resize(size_t new_size) {
    return new_size * 10 >= _capacity * RESIZE_THRES;
  }

  /** Resizes the dictionary. */
  void resize(int64_t new_capacity) {
    const size_t sz = slot_size();
    InternalDict resized;
    resized._size = _size;
    resized._capacity = new_capacity;
    resized._data =
        weld_run_malloc(weld_rt_get_run_id(), resized._capacity * sz);
    memset(resized._data, 0, resized._capacity * sz);

    for (int i = 0; i < _capacity; i++) {
      Slot *old_slot = slot_at_index(i);
      if (old_slot->header.filled) {
        // Locking the slot here is not required, since the caller should acquire the write lock
        // on the full dictionary.
        Slot *new_slot =
            resized.get_slot(old_slot->header.hash, NULL, NO_LOCKING, false);
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

class GlobalBuffer {

  public:

    void *data;
    // The size, in slots.
    int64_t size;
    // Size of a single slot, in bytes.
    const int64_t slot_size;


  public:

    GlobalBuffer(int64_t a_slot_size): slot_size(a_slot_size) {
        data = weld_run_malloc(weld_rt_get_run_id(), GLOBAL_BATCH_SIZE * slot_size);
        memset(data, 0, GLOBAL_BATCH_SIZE * slot_size);
        size = 0;
    }

    ~GlobalBuffer() {
      weld_run_free(weld_rt_get_run_id(), data);
    }

    // Returns true if this buffer contains the slot, by checking whether the
    // slot address lies in the buffer's data.
    bool contains_slot(Slot *s) {
      intptr_t start = (intptr_t)data;
      intptr_t end = start + size * slot_size;
      intptr_t slot = (intptr_t)s;
      return slot >= start && slot < end;
    }

    /** Returns the slot at the given index. */
    inline Slot *slot_at_index(long index) {
      assert(index < GLOBAL_BATCH_SIZE);
      return (Slot *)(((uint8_t *)data) + slot_size * index);
    }

    /** Returns the next free slot. */
    inline Slot *next_slot() {
      assert(size < GLOBAL_BATCH_SIZE);
      return (Slot *)(((uint8_t *)data) + slot_size * size);
    }
};

struct SerializedWeldDictHeader {
  int32_t workers;
  int64_t key_size;
  int64_t value_size;
  int64_t size;
  int64_t capacity;
};

/** A multi-threaded dictionary used by the Weld runtime. */
class WeldDict {

private:
  // The contiguous buffer of dictionaries. If there are N workers, there are
  // N + 1 dictionaries, where index i holds the dictionary for thread ID i,
  // and dictionary N+1 is the global dictionary.
  void *_dicts;

  // Buffers for writes into the global dictionary.
  void *_global_buffers;
  int32_t _workers;
  void *_metadata;
  int32_t _packed_value_size;

  // Merge functions.
  MergeFn _merge_fn;
  MergeFn _finalize_merge_fn;

public:
  bool finalized;

public:

  /** Get the dictionary at the given index.
   *
   * @i the index.
   * @return the dictionary the index.
   */
  inline InternalDict *dict_at_index(int32_t i) {
    return (InternalDict *)weld_rt_get_merger_at_index(_dicts,
                                                       sizeof(InternalDict), i);
  }

  /** Get the thread-local write buffer at the given index.
   *
   * @i the index.
   * @return the buffer the index.
   */
  inline GlobalBuffer *buffer_at_index(int32_t i) {
    return (GlobalBuffer *)weld_rt_get_merger_at_index(_global_buffers,
                                                       sizeof(GlobalBuffer), i);
  }

  /** Get the write buffer for the calling thread. */
  inline GlobalBuffer *local_buffer() { return buffer_at_index(weld_rt_thread_id()); }

  /** Get the write buffer dictionary. */
  inline GlobalBuffer *global_buffer() { return buffer_at_index(_workers); }

  /** Get the dictionary for the calling thread. */
  inline InternalDict *local_dict() { return dict_at_index(weld_rt_thread_id()); }

  /** Get the global dictionary. */
  inline InternalDict *global_dict() { return dict_at_index(_workers); }

  WeldDict(int32_t key_size,
      KeyComparator keys_eq,
      MergeFn merge_fn,
      MergeFn finalize_merge_fn,
      void *metadata,
      int32_t value_size,
      int32_t packed_value_size, 
      int64_t full_watermark,
      int64_t capacity) {

    _workers = weld_rt_get_nworkers();
    _merge_fn = merge_fn;
    _finalize_merge_fn = finalize_merge_fn;
    _metadata = metadata;
    _packed_value_size = packed_value_size;
    _dicts = weld_rt_new_merger(sizeof(InternalDict), _workers + 1);
    _global_buffers = weld_rt_new_merger(sizeof(GlobalBuffer), _workers + 1);

    for (int i = 0; i < _workers + 1; i++) {
      InternalDict *dict = dict_at_index(i);
      new (dict)
          InternalDict(key_size, value_size, keys_eq, capacity, full_watermark);

      GlobalBuffer *glbuf = buffer_at_index(i);
      new (glbuf) GlobalBuffer(dict->slot_size());
    }

    finalized = (full_watermark == 0);
  }

  ~WeldDict() {
    for (int i = 0; i < _workers + 1; i++) {
      InternalDict *dict = dict_at_index(i);
      GlobalBuffer *glbuf = buffer_at_index(i);
      dict->~InternalDict();
      glbuf->~GlobalBuffer();
    }
    weld_rt_free_merger(_dicts);
    weld_rt_free_merger(_global_buffers);
  }


  Slot *lookup(int32_t hash, void *key) {
    assert(finalized);
    InternalDict *dict = global_dict();
    Slot *s = dict->get_slot(hash, key, NO_LOCKING, true);
    return s;
  }

  void merge(int32_t hash, void *key, void *value) {
    Slot *slot = lookup(hash, key);
    GlobalBuffer *glbuf = local_buffer();
    InternalDict *dict = local_dict();

    if (glbuf->contains_slot(slot)) {
      // The put occured in the write buffer.
      slot->update(NULL, NULL, hash, key, dict->key_size, value, dict->value_size);
      glbuf->size++;
      if (glbuf->size == GLOBAL_BATCH_SIZE) {
        drain_global_buffer(glbuf);
      }
    } else {
      // The put occurred in a dictionary (either global or local).
      slot->update(_merge_fn,
          _metadata,
          hash, key,
          dict->key_size,
          value,
          dict->value_size);

      if (dict->contains_slot(slot)) {
        dict->put_slot(slot, NO_LOCKING);
        return;
      }

      dict = global_dict();
      assert(dict->contains_slot(slot));
      dict->put_slot(slot, ACQUIRE);
    }
  }

  void finalize() {
    assert(!finalized);
    finalized = true;

    int64_t max_capacity = 0;
    for (int i = 0; i < _workers; i++) {
      InternalDict *dict = dict_at_index(i);
      if (dict->capacity() > max_capacity) {
        max_capacity = dict->capacity();
      }
    }

    // Prevents spurious resizes during the finalization phase.
    InternalDict *dict = global_dict();
    dict->resize(max_capacity);

    for (int i = 0; i < _workers; i++) {

      drain_global_buffer(buffer_at_index(i));

      InternalDict *ldict = dict_at_index(i);
      for (int j = 0; j < ldict->capacity(); j++) {
        Slot *slot = ldict->slot_at_index(j);
        if (slot->header.filled) {
          Slot *global_slot = dict->get_slot(slot->header.hash, slot->key(), NO_LOCKING, true);
          global_slot->update(_finalize_merge_fn,
              _metadata,
              global_slot->header.hash,
              global_slot->key(), dict->key_size,
              slot->value(dict->key_size), dict->value_size);
          dict->put_slot(global_slot, NO_LOCKING);
        }
      }
    }
  }

  /** Returns an array of { key, value } structs, potentially padded, given this
   * dictionary. The dictionary must be finalized.
   *
   * @param value_offset the offset of the value after the key. This accounts
   * for padding in the returned struct.
   * @param struct_size the total struct size. This handles padding after the
   * value.
   *
   * @return a buffer of {key, value} pairs.
   *
   * */
  void *new_kv_vector(int32_t value_offset, int32_t struct_size) {
    assert(finalized);
    InternalDict *dict = global_dict();

    int32_t key_padding_bytes = value_offset - dict->key_size;

    uint8_t *buf = (uint8_t *)weld_run_malloc(weld_rt_get_run_id(),
                                              dict->size() * struct_size);
    long offset = 0;

    for (long i = 0; i < dict->capacity(); i++) {
      Slot *slot = dict->slot_at_index(i);
      if (slot->header.filled) {
        uint8_t *offset_buf = buf + offset * struct_size;
        memcpy(offset_buf, slot->key(), dict->key_size);
        offset_buf += dict->key_size + key_padding_bytes;
        memcpy(offset_buf, slot->value(dict->key_size), _packed_value_size);
        offset++;
      }
    }
    assert(offset == dict->size());
    return (void *)buf;
  }

private:

  void drain_global_buffer(GlobalBuffer *glbuf) {
    InternalDict *dict = global_dict();

    dict->read_lock();

    for (int i = 0; i < glbuf->size; i++) {
      Slot *buf_slot = glbuf->slot_at_index(i);
      Slot *global_slot = dict->get_slot(buf_slot->header.hash, buf_slot->key(), ACQUIRE, true);

      global_slot->update(_merge_fn,
          _metadata,
          buf_slot->header.hash,
          global_slot->key(), dict->key_size,
          global_slot->value(dict->key_size), dict->value_size);

      dict->put_slot(global_slot, ACQUIRE);
    }

    dict->unlock();
    // The buffer is drained.
    glbuf->size = 0;
  }
};

extern "C" void *weld_rt_dict_new(int32_t key_size,
    KeyComparator keys_eq,
    MergeFn merge_fn,
    MergeFn finalize_merge_fn,
    void *metadata,
    int32_t val_size,
    int32_t packed_value_size,
    int64_t max_local_bytes,
    int64_t capacity) {

  WeldDict *wd =
      (WeldDict *)weld_run_malloc(weld_rt_get_run_id(), sizeof(WeldDict));

  new (wd) WeldDict(key_size,
      keys_eq,
      merge_fn,
      finalize_merge_fn,
      metadata,
      val_size,
      packed_value_size,
      max_local_bytes,
      capacity);

  return (void *)wd;
}

extern "C" void weld_rt_dict_free(void *d) {
  WeldDict *wd = (WeldDict *)d;
  wd->~WeldDict();
  weld_run_free(weld_rt_get_run_id(), wd);
}

extern "C" void *weld_rt_dict_lookup(void *d, int32_t hash, void *key) {
  WeldDict *wd = (WeldDict *)d;
  return (void *)wd->lookup(hash, key);
}

extern "C" void weld_rt_dict_merge(void *d, int32_t hash, void *key, void *value) {
  WeldDict *wd = (WeldDict *)d;
  wd->merge(hash, key, value);
}

// Begin the finalization procedure.
extern "C" void weld_rt_dict_finalize(void *d) {
  WeldDict *wd = (WeldDict *)d;
  wd->finalize();
}

extern "C" void *weld_rt_dict_to_array(void *d, int32_t value_offset,
                                            int32_t struct_size) {

  WeldDict *wd = (WeldDict *)d;
  return wd->new_kv_vector(value_offset, struct_size);
}

extern "C" int64_t weld_rt_dict_size(void *d) {
  WeldDict *wd = (WeldDict *)d;
  assert(wd->finalized);

  InternalDict *dict = wd->global_dict();
  return dict->size();
}
