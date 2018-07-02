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

// Resize threshold. Resizes when RESIZE_THRES * 10 percent of slots are full.
const int RESIZE_THRES = 7;

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

// A single slot. The header is followed by sizeof(key_size) +
// sizeof(value_size) bytes. The header + the key and value define a full slot.
struct Slot {
public:
  // Slot metadata.
  struct SlotHeader {
    int32_t hash;
    uint8_t filled;
  } header;

  // Returns the key, which immediately follows the header.
  inline void *key() {
    return reinterpret_cast<uint8_t*>(this) + sizeof(header);
  }

  // Returns the value, which follows the header, and then `key_size` bytes.
  inline void *value(size_t key_size) {
    return reinterpret_cast<uint8_t*>(this) + sizeof(header) + key_size;
  }
};

// TODO(Alex): We need better plumbing of error conditions, especially OOM.
class WeldDict {
private:
  const int64_t _key_size;
  const int64_t _value_size;
  const KeyComparator _keys_eq;

  // An array of Slots.
  uint64_t *_data;
  int64_t _capacity;
  int64_t _size;

public:
  WeldDict(int64_t key_size, int64_t value_size, KeyComparator keys_eq, int64_t capacity)
      : _key_size(key_size), _value_size(value_size),
        _keys_eq(keys_eq), _data(nullptr), _capacity(capacity), _size(0) {
  }

  bool init() {
    // Power of 2 check.
    assert(_capacity > 0 && (_capacity & (_capacity - 1)) == 0);
    size_t data_size = _capacity * slot_size();
    _data = static_cast<uint64_t*>(weld_run_malloc(weld_rt_get_run_id(), data_size));
    if (_data == nullptr) return false;
    memset(_data, 0, data_size);
    return true;
  }

  void free_weld_dict() {
    weld_run_free(weld_rt_get_run_id(), _data);
  }

  inline int64_t slot_size() const {
    return sizeof(Slot::SlotHeader) + _key_size + _value_size;
  }

  int64_t size() const { return _size; }
  int64_t capacity() const { return _capacity; }

  // Returns the slot at the given index.
  inline Slot *slot_at_idx(int64_t slot_idx) const {
    return reinterpret_cast<Slot*>(_data + slot_size() * slot_idx);
  }


  inline bool init_slot(Slot *slot, int32_t hash, void *key, void *value) {
  	slot->header.filled = true;
  	memcpy(slot->key(), key, _key_size);
  	memcpy(slot->value(_key_size), value, _value_size);
  	++_size;
  	// Check if a resize is needed.
    if (_size * 10 >= _capacity * RESIZE_THRES) return resize(_capacity * 2);
    return true;
  }

  /**
   * Returns the slot for the given hash/key. If the dictionary does not have an entry
   * for the hash/key then its slot is initialized with the init_value and returned.
   * The dictionary may be resized as a result of initializing a new slot.
   * Returns NULL if no slot could be found or created.
   */
  Slot *upsert_slot(int32_t hash, void *key, void *init_value) {
    // Linear probing.
  	int64_t start_slot_idx = hash & (_capacity - 1);
    for (int64_t i = 0; i < _capacity; ++i) {
      int64_t slot_idx = (start_slot_idx + i) & (_capacity - 1);
      Slot *slot = slot_at_idx(slot_idx);
      if (!slot->header.filled) {
      	if (!init_slot(slot, hash, key, init_value)) return nullptr;
      	return slot;
      }

      // Slot is filled - check it.
      if (slot->header.hash == hash && _keys_eq(key, slot->key())) {
        return slot;
      }
    }
    return nullptr;
  }

  /**
   * Returns the first empty slot in linear-probe search order starting from
   * the slot corresponding to the given hash.
   */
  Slot *get_empty_slot(int32_t hash) {
  	int64_t start_slot_idx = hash & (_capacity - 1);
    for (int64_t i = 0; i < _capacity; ++i) {
      int64_t slot_idx = (start_slot_idx + i) & (_capacity - 1);
      Slot *slot = slot_at_idx(slot_idx);
      if (!slot->header.filled) return slot;
    }
    return nullptr;
  }

  /** Resizes the dictionary. */
  bool resize(int64_t new_capacity) {
    assert(new_capacity > _capacity);

    const int64_t sz = slot_size();
    WeldDict resized(_key_size, _value_size, _keys_eq, new_capacity);
    if (!resized.init()) return false;

    // Copy slots to resized dictionary.
    for (int i = 0; i < _capacity; i++) {
    	Slot *old_slot = slot_at_idx(i);
    	if (!old_slot->header.filled) continue;
    	Slot *new_slot = resized.get_empty_slot(old_slot->header.hash);
    	assert(new_slot != nullptr);
    	memcpy(new_slot, old_slot, sz);
    }

    // Free the old data.
    weld_run_free(weld_rt_get_run_id(), _data);

    _data = resized._data;
    _capacity = resized._capacity;
    return true;
  }

  /**
   * Returns an array of { key, value } structs, potentially padded, given this
   * dictionary. The dictionary must be finalized.
   *
   * @param value_offset the offset of the value after the key. This accounts
   * for padding in the returned struct.
   * @param struct_size the total struct size. This handles padding after the
   * value.
   *
   * @return a buffer of {key, value} pairs.
   */
  void *new_kv_vector(int32_t value_offset, int32_t struct_size) {
  	int64_t key_padding_bytes = value_offset - _key_size;
  	uint8_t *buf = reinterpret_cast<uint8_t*>(
  			weld_run_malloc(weld_rt_get_run_id(), _size * struct_size));
  	if (buf == nullptr) return nullptr;
  	memset(buf, 0, struct_size * _size);

  	int64_t offset = 0;
  	for (int64_t i = 0; i < _capacity; ++i) {
  		Slot *slot = slot_at_idx(i);
  		if (!slot->header.filled) continue;
  		uint8_t *offset_buf = buf + offset * struct_size;
  		memcpy(offset_buf, slot->key(), _key_size);
  		offset_buf += _key_size + key_padding_bytes;
  		// TODO(Alex): The value size used to be the _packed_value_size. Clean up the LLVM interop.
  		memcpy(offset_buf, slot->value(_key_size), _value_size);
  		offset++;
  	}
  	assert(offset == _size);
  	return buf;
  }

  /** Serializes a dictionary, flattening pointers if necessary. */
  void serialize(GrowableVec *gvec, int32_t has_pointer, SerializeFn serialize_key_fn, SerializeFn serialize_value_fn) {
  	if (!has_pointer) {
      serialize_no_pointers(gvec);
    } else {
      serialize_with_pointers(gvec, serialize_key_fn, serialize_value_fn);
    }
  }
private:

  /** Serializes a dictionary, where the keys and values have no pointers. */
  void serialize_with_pointers(GrowableVec *gvec, SerializeFn serialize_key_fn, SerializeFn serialize_value_fn) {
    gvec->resize_to_fit(sizeof(int64_t));

    int64_t *as_i64_ptr = (int64_t *)(gvec->vector.data + gvec->size);
    *as_i64_ptr = _size;
    gvec->size += sizeof(int64_t);

    // Copy each key/value pair into the buffer.
    for (long i = 0; i < _capacity; ++i) {
    	Slot *slot = slot_at_idx(i);
    	if (!slot->header.filled) continue;
    	serialize_key_fn(gvec, slot->key());
    	serialize_value_fn(gvec, slot->value(_key_size));
    }
  }

  /** Serializes a dictionary, where the keys and values have no pointers. */
  void serialize_no_pointers(GrowableVec *gvec) {
  	const int64_t bytes = sizeof(int64_t) + (_key_size + _value_size) * _size;
  	gvec->resize_to_fit(bytes);

  	uint8_t *offset = gvec->vector.data + gvec->size;
  	int64_t *as_i64_ptr = (int64_t *)offset;
  	*as_i64_ptr = _size;
  	offset += 8;

  	// Copy each key/value pair into the buffer.
  	for (long i = 0; i < _capacity; i++) {
  		Slot *slot = slot_at_idx(i);
  		if (!slot->header.filled) continue;
  		memcpy(offset, slot->key(), _key_size);
  		offset += _key_size;
  		memcpy(offset, slot->value(_key_size), _value_size);
  		offset += _value_size;
  	}

  	assert(offset - gvec->vector.data == bytes);
  	gvec->size += bytes;
  }
};

extern "C" void *weld_rt_dict_new(
		int32_t key_size,
		int32_t value_size,
    KeyComparator keys_eq,
    int64_t capacity) {
  WeldDict *wd = reinterpret_cast<WeldDict*>(
  		weld_run_malloc(weld_rt_get_run_id(), sizeof(WeldDict)));
  if (wd == nullptr) return nullptr;
  new (wd) WeldDict(key_size, value_size, keys_eq, capacity);
  if (!wd->init()) return nullptr;
  return wd;
}

extern "C" void weld_rt_dict_free(void *d) {
  WeldDict *wd = (WeldDict *)d;
  wd->free_weld_dict();
  weld_run_free(weld_rt_get_run_id(), wd);
}

extern "C" void *weld_rt_upsert_slot(void *d, int32_t hash, void *key, void *init_value) {
  WeldDict *wd = static_cast<WeldDict*>(d);
  return wd->upsert_slot(hash, key, init_value);
}

extern "C" int64_t weld_rt_dict_size(void *d) {
  WeldDict *wd = static_cast<WeldDict*>(d);
  return wd->size();
}

extern "C" void *weld_rt_dict_to_array(void *d, int32_t value_offset, int32_t struct_size) {
  WeldDict *wd = static_cast<WeldDict*>(d);
  return wd->new_kv_vector(value_offset, struct_size);
}

extern "C" void weld_rt_dict_serialize(void *d, void *buf, int32_t has_pointer, void *key_ser, void *val_ser) {
  WeldDict *wd = static_cast<WeldDict*>(d);
  wd->serialize(static_cast<GrowableVec*>(buf), has_pointer,
  		reinterpret_cast<SerializeFn>(key_ser), reinterpret_cast<SerializeFn>(val_ser));
}


