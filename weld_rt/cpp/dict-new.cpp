/**
 *
 * The generic, bytes to bytes dictionary implementation for the Weld runtime.
 *
 * Note: See dict.h for the Weld-facing dictionary API. This implements that API.
 *
 * This is the generic bytes to bytes dictionary implementation for the Weld runtime. The dictionary
 * uses an adaptive algorithm that transitions from thread-local dictionaries to global dictionaries
 * based on the size of the thread-local dictionaries.
 *
 * Dictionaries follow the Weld builder pattern, i.e., "reads only after all writes are complete."
 *
 * The basic algorithm is as follows, assuming multi-threaded concurrent writes:
 *
 *
 * Writes
 * ___________________________________________________________________________________
 *
 * Writes occur on an un-contended thread-local dictionary unless the
 * thread-local dictionary is `full_watermark` bytes in size. If this is the
 * case, the writes go to a global shared dictionary.
 *
 * If a write occurs to a global dictionary, the update goes to a thread-local
 * buffer called GlobalBuffer to reduce the overhead of locking the dictionary.
 * Once the GlobalBuffer has GLOBAL_BATCH_SIZE items, the global dictionary is
 * locked via a pthread RW-lock. A read lock is held for inserts into a slot,
 * and a write lock is held if the dictionary must be resized.  The slots in
 * the global buffer are merged into the main dictionary. Individual slots are
 * protected at LOCK_GRANULARITY via a spin lock.
 *
 * 
 * Layout
 * ___________________________________________________________________________________
 *
 * The dictionary is laid out as a sequential buffer of Slots. A single slot has the following layout:
 *
 * ----------------------------------------------------------------------------------------------
 *                                                |                     |                       |
 *              SlotHeader bytes                  |  sizeof(key) bytes  |  sizeof(value) bytes  |
 *  (contains filled flag, hash, some metadata)   |                     |                       | .... (repeats)
 *                                                |                     |                       |
 * ------------------------------------------------ ---------------------------------------------
 *
 *
 * Finalization
 * ___________________________________________________________________________________
 *
 * When writes are complete, the `finalized` flag is set to true, and slots in the
 * local dictionary are merged into the global dictionary.
 *
 *
 * Conversion to Array of Keys/Values
 * ___________________________________________________________________________________
 *
 * The dictionary supports conversion to a value of key/value structs. Users specify a total size for the
 * KV-struct, and the padding after the key. This allows accounting for padding in C structs, since
 * keys and values are stored in a packed layout in the dictionary. 
 *
 */
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

// The growable vector type used for serialization.
struct GrowableVec {

public:
  struct {
    uint8_t *data;
    int64_t capacity;
  } vector;

  int64_t size;

public:

  // Size the vector to fit `bytes` data in the vector. Does nothing if the vector already
  // has the capacity to hold `bytes`.
  void resize_to_fit(int64_t bytes) {
    if (vector.capacity - size < bytes) {
      int64_t new_capacity;
      if (vector.capacity + bytes > vector.capacity * 2) {
        new_capacity = vector.capacity + bytes;
      } else {
        new_capacity = vector.capacity * 2;
      }
      vector.data = (uint8_t *)weld_run_realloc(weld_rt_get_run_id(), vector.data, new_capacity);
      vector.capacity = new_capacity;
    }
  }

};

/**Type alias for serialization function over pointers
 *
 * @param the growable vector where the serialized data is written
 * @param the value being serialized.
 *
 */
typedef void (*SerializeFn)(GrowableVec *, void *);

// Specifies what kind of locking to use. Basically a boolean, but its here for
// additional type safety enforced by the compiler.
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
    volatile uint8_t lockvar; // locks the slot with a CAS.
    uint16_t _pad;
  } header;

  // Returns the key, which immediately follows the header.
  inline void *key() {
    return reinterpret_cast<uint8_t *>(this) + sizeof(header);
  }

  // Returns the value, which follows the header, and then `key_size` bytes.
  inline void *value(size_t key_size) {
    return reinterpret_cast<uint8_t *>(this) + sizeof(header) + key_size;
  }

  /** Update the slot with a merge function.
   *
   * @param merge the merge function to use.
   * @param metadata data passed to the merge function.
   * @param a_hash the hash
   * @param a_key the key
   * @param a_key the key size
   * @param a_key the value
   * @param a_key the the value size
   *
   */
  void update(MergeFn merge,
      void *metadata,
      int32_t a_hash,
      void *a_key, int64_t key_size,
      void *a_val, int64_t val_size) {

    if (!header.filled) {
      memcpy(key(), a_key, key_size);
      header.hash = a_hash;
    }

    // Either use a merge function if one is provided, or replace the existing key.
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

// An internal dictionary which provides (optionally) locked access to its slots.
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

private:

  /** Given a slot, returns the Slot which holds the lock for the slot. If the
   * locking granularity is one slot, this will return the argument */
  inline Slot *lock_for_slot(Slot *slot) {
    // Compute the locked slot index, and then return the slot at the index
    int64_t index = ((intptr_t)slot - (intptr_t)_data) / slot_size();
    return slot_at_index(index & ~(LOCK_GRANULARITY - 1));
  }

public:

  // A default constructor that initializes all values with undefined memory
  // and allocates no heap memory.  Used to initialize the class as a C-struct.
  InternalDict(int64_t a_key_size, int64_t a_value_size)
      : key_size(a_key_size), value_size(a_value_size), full_watermark(0), keys_eq(0), _size(0),
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

  void free_internal_dict() {
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
   * @param hash the hash of the key
   * @param key the key data. Should be key_size bytes long.
   * @param mode the locking mode -- ACQUIRE for locked access, NO_LOCKING for no locking.
   * @param check_key a flag that, if true, causes this function to return the
   * first unfilled slot for the hash without comparing the keys at the lot.
   *
   * @return the slot.
   *
   * Prerequisites: If this is a concurrently accessed dictionary, call lock() first.
   *
   */
  Slot *get_slot(int32_t hash, void *key, LockMode mode, bool check_key) {

    int64_t first_offset = hash & (_capacity - 1);
    Slot *locked_slot = NULL;

    // Linear probing.
    for (long i = 0; i < _capacity; i++) {
      long index = (first_offset + i) & (_capacity - 1);
      Slot *slot = slot_at_index(index);

      // Lock the slot to disable concurrent access on the LOCK_GRANULARITY
      // buckets we probe.
      if (mode == ACQUIRE && (i == 0 || lockable_index(index))) {
        if (locked_slot != NULL) {
          locked_slot->release_access();
        }
        locked_slot = lock_for_slot(slot);
        locked_slot->acquire_access();
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
   * using get_slot, and locked the dictionary using read_lock. 
   *
   * @param slot a slot acquired via get_slot.
   * @param mode the locking mode. This should match the locking mode used to
   * get this slot using get_slot.
   *
   * Prerequisites: Get a slot using get_slot, lock the dictionary with
   * read_lock() if ACQUIRE is true.
   *
   * */
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
    assert(new_capacity > _capacity);

    const size_t sz = slot_size();
    InternalDict resized(key_size, value_size);
    resized._size = _size;
    resized._capacity = new_capacity;
    resized._data = weld_run_malloc(weld_rt_get_run_id(), resized._capacity * sz);
    memset(resized._data, 0, resized._capacity * sz);

    for (int i = 0; i < _capacity; i++) {
      Slot *old_slot = slot_at_index(i);
      if (old_slot->header.filled) {
        // Locking the slot here is not required, since the caller should
        // acquire the write lock on the full dictionary.
        Slot *new_slot =
            resized.get_slot(old_slot->header.hash, NULL, NO_LOCKING, false);
        assert(new_slot != NULL);
        memcpy(new_slot->key(), old_slot->key(), key_size);
        memcpy(new_slot->value(key_size), old_slot->value(key_size), value_size);
        new_slot->header.hash = old_slot->header.hash;
        new_slot->header.filled = 1;
      }
    }

    // Free the old data.
    weld_run_free(weld_rt_get_run_id(), _data);

    _data = resized._data;
    _size = resized._size;
    _capacity = resized._capacity;
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

    void free_global_buffer() {
      weld_run_free(weld_rt_get_run_id(), data);
    }

    // Returns true if this buffer contains the slot, by checking whether the
    // slot address lies in the buffer's data.
    bool contains_slot(Slot *s) {
      intptr_t start = (intptr_t)data;
      intptr_t end = (intptr_t)(((uint8_t *)data) + GLOBAL_BATCH_SIZE * slot_size);
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
      int64_t capacity,
      bool init_finalized) {

    _workers = weld_rt_get_nworkers();
    _merge_fn = merge_fn;
    _finalize_merge_fn = finalize_merge_fn;
    _metadata = metadata;
    _packed_value_size = packed_value_size;
    _dicts = weld_rt_new_merger(sizeof(InternalDict), _workers + 1);
    _global_buffers = weld_rt_new_merger(sizeof(GlobalBuffer), _workers);
    finalized = init_finalized;

    // All writes go directly to the global dictionary with no locking if we
    // run on one thread.
    if (_workers == 1 || init_finalized) {
      finalized = true;
      full_watermark = 0;
    }

    for (int i = 0; i < _workers + 1; i++) {
      InternalDict *dict = dict_at_index(i);
      new (dict)
        InternalDict(key_size, value_size, keys_eq, capacity, full_watermark);

      if (i != _workers) {
        GlobalBuffer *glbuf = buffer_at_index(i);
        new (glbuf) GlobalBuffer(dict->slot_size());
      }
    }
  }

  void free_weld_dict() {
    for (int i = 0; i < _workers + 1; i++) {
      InternalDict *dict = dict_at_index(i);
      GlobalBuffer *glbuf = buffer_at_index(i);
      dict->free_internal_dict();

      if (i != _workers) {
        glbuf->free_global_buffer();
      }
    }

    if (_metadata) {
      weld_run_free(weld_rt_get_run_id(), _metadata);
    }

    weld_rt_free_merger(_dicts);
    weld_rt_free_merger(_global_buffers);
  }


  Slot *lookup(int32_t hash, void *key) {
    // First check the local dictionary.
    if (!finalized) {
      InternalDict *dict = local_dict();
      Slot *slot = dict->get_slot(hash, key, NO_LOCKING, true);

      // Use the slot in the local dictionary if (a) the dictionary is not full or
      // (b) the slot is already occupied and just needs to be updated with a new
      // value for the key.
      if (!dict->full() || (slot != NULL && slot->header.filled)) {
        return slot;
      }
    }

    // Check the global dictionary - if its not finalized, provide a write slot
    // in the global buffer.
    if (!finalized) {
      GlobalBuffer *glbuf = local_buffer();
      Slot *slot = glbuf->next_slot();
      return slot;
    }

    // If the dictionary is finalized, provide a slot without locking.
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
      slot->header.filled = 0;
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
      // If finalized, we don't need to lock/unlock anything.
      dict->put_slot(slot, finalized ? NO_LOCKING : ACQUIRE);
    }
  }

  void finalize() {
    if (finalized) {
      return;
    }
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
    if (dict->capacity() < max_capacity) {
      dict->resize(max_capacity);
    }

    for (int i = 0; i < _workers; i++) {

      drain_global_buffer(buffer_at_index(i));

      InternalDict *ldict = dict_at_index(i);

      for (int j = 0; j < ldict->capacity(); j++) {
        Slot *slot = ldict->slot_at_index(j);
        if (slot->header.filled) {
          Slot *global_slot = dict->get_slot(slot->header.hash, slot->key(), NO_LOCKING, false);
          global_slot->update(_finalize_merge_fn,
              _metadata,
              slot->header.hash,
              slot->key(), dict->key_size,
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
    memset(buf, 0, struct_size*dict->size());
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

  /** Serializes a dictionary, flattening pointers if necessary. */
  void serialize(void *buffer, int32_t has_pointer, SerializeFn serialize_key_fn, SerializeFn serialize_value_fn) {
    assert(finalized);
    if (!has_pointer) {
      serialize_no_pointers(buffer);
    } else {
      serialize_with_pointers(buffer, serialize_key_fn, serialize_value_fn);
    }
  }

private:

  /** Serializes a dictionary, where the keys and values have no pointers. */
  void serialize_with_pointers(void *buffer, SerializeFn serialize_key_fn, SerializeFn serialize_value_fn) {
    assert(finalized);
    InternalDict *dict = global_dict();

    GrowableVec *gvec = (GrowableVec *)buffer;
    gvec->resize_to_fit(sizeof(int64_t));

    int64_t *as_i64_ptr = (int64_t *)(gvec->vector.data + gvec->size);
    *as_i64_ptr = dict->size();
    gvec->size += sizeof(int64_t);

    // Copy each key/value pair into the buffer.
    for (long i = 0; i < dict->capacity(); i++) {
      Slot *slot = dict->slot_at_index(i);
      if (slot->header.filled) {
        serialize_key_fn(gvec, slot->key());
        serialize_value_fn(gvec, slot->value(dict->key_size));
      }
    }
  }

  /** Serializes a dictionary, where the keys and values have no pointers. */
  void serialize_no_pointers(void *buffer) {
    assert(finalized);
    InternalDict *dict = global_dict();

    GrowableVec *gvec = (GrowableVec *)buffer;

    const int64_t bytes = sizeof(int64_t) + (dict->key_size + dict->value_size) * dict->size();
    gvec->resize_to_fit(bytes);

    uint8_t *offset = gvec->vector.data + gvec->size;

    int64_t *as_i64_ptr = (int64_t *)offset;
    *as_i64_ptr = dict->size();
    offset += 8;

    // Copy each key/value pair into the buffer.
    for (long i = 0; i < dict->capacity(); i++) {
      Slot *slot = dict->slot_at_index(i);
      if (slot->header.filled) {
        memcpy(offset, slot->key(), dict->key_size);
        offset += dict->key_size;
        memcpy(offset, slot->value(dict->key_size), dict->value_size);
        offset += dict->value_size;
      }
    }

    assert(offset - gvec->vector.data == bytes);
    gvec->size += bytes;
  }

  // Flush the global buffer into the global dictionary. Assumes concurrent access to the global dictionary.
  void drain_global_buffer(GlobalBuffer *glbuf) {
    InternalDict *dict = global_dict();

    dict->read_lock();
    for (int i = 0; i < glbuf->size; i++) {
      Slot *buf_slot = glbuf->slot_at_index(i);
      Slot *global_slot = dict->get_slot(buf_slot->header.hash, buf_slot->key(), ACQUIRE, true);
      global_slot->update(_merge_fn,
          _metadata,
          buf_slot->header.hash,
          buf_slot->key(), dict->key_size,
          buf_slot->value(dict->key_size), dict->value_size);

      dict->put_slot(global_slot, ACQUIRE);

      // Free the slot.
      buf_slot->header.filled = 0;
      buf_slot->header.hash = 0;
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
      capacity,
      false);

  return (void *)wd;
}

extern "C" void *weld_rt_dict_new_finalized(int32_t key_size,
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
      capacity,
      true);

  return (void *)wd;
}

extern "C" void weld_rt_dict_free(void *d) {
  WeldDict *wd = (WeldDict *)d;
  wd->free_weld_dict();
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

extern "C" void weld_rt_dict_serialize(void *d, void *buf, int32_t has_pointer, void *key_ser, void *val_ser) {
  WeldDict *wd = (WeldDict *)d;
  wd->serialize(buf, has_pointer, (SerializeFn)key_ser, (SerializeFn)val_ser);
}


