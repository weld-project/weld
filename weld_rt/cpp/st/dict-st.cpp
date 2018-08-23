/**
 *
 * The generic, bytes to bytes dictionary implementation for the Weld runtime.
 *
 * This implementation is single-threaded.
 *
 */
#include "strt.h"
#include "dict-st.h"
#include "vec.h"

#include <cstring>
#include <algorithm>
#include <assert.h>

#ifdef DEBUG
#define DBG(fmt, args...) fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, \
    __FILE__, __LINE__, __func__, ##args)
#else
#define DBG(fmt, args...)
#endif

// The initial capacity of a growable vector.
const int INITIAL_CAPACITY = 16;

// The canonical layout for a weld vector.
class WeldVec {
public:
  // Data Pointer
  void *data;
  // Size in the number of elements
  int64_t size;
};

class GrowableWeldVec {
  public:
    WeldVec wv;
    // Capacity in number of elements.
    int64_t capacity;

  public:
    void resize(WeldRunHandleRef run, int32_t value_size, int64_t new_capacity) {
      wv.data = weld_runst_realloc(run, wv.data, new_capacity * value_size);
      capacity = new_capacity;
    }

    void *at(int32_t val_size, int64_t i) {
      return (void *)((uint8_t *)wv.data + i * val_size);
    }
};


// Resize threshold. Resizes when RESIZE_THRES * 10 percent of slots are full.
const int RESIZE_THRES = 7;

/**Type alias for serialization function over pointers
 *
 * @param the growable vector where the serialized data is written
 * @param the value being serialized.
 *
 */
typedef void (*SerializeFn)(Vec<uint8_t> *, void *, WeldRunHandleRef);

// A single slot. The header is followed by sizeof(key_size) +
// sizeof(value_size) bytes. The header + the key and value define a full slot.
struct Slot {
public:
  // Slot metadata.
  struct SlotHeader {
    int32_t hash;
    uint32_t filled : 1;
    uint32_t pad    : 31;
  } header __attribute__((packed));

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
  uint8_t *_data;
  int64_t _capacity;
  int64_t _size;

  // Reference to the run that owns this dictionary.
  WeldRunHandleRef _run;

  // Size of a packed value. This can be used to trim bytes off the end of a value.
  // Currently only used for groupmerger.
  int32_t _packed_value_size;

public:
  WeldDict(WeldRunHandleRef run, int64_t key_size, int64_t value_size, KeyComparator keys_eq, int64_t capacity)
      : _key_size(key_size), _value_size(value_size),
        _keys_eq(keys_eq), _data(nullptr), _capacity(capacity), _size(0), _run(run), _packed_value_size(value_size) {
  }

  void set_packed_value_size(int64_t v) {
    _packed_value_size = v;
  }

  bool init() {
    // TODO(shoumik): This may be wasteful, but it prevents odd uninitialized
    // pointer errors later.
    if (_capacity == 0) {
      _capacity = 16;
    }
    // Power of 2 check.
    assert((_capacity & (_capacity - 1)) == 0);
    size_t data_size = _capacity * slot_size();
    _data = static_cast<uint8_t *>(weld_runst_malloc(_run, data_size));
    assert(_data);
    memset(_data, 0, data_size);
    return true;
  }

  void free_weld_dict() {
    weld_runst_free(_run, _data);
  }

#ifdef DEBUG
  inline int64_t dbg_print(void *p, int32_t sz) {
    if (sz >= sizeof(int64_t)) {
      return *((int64_t *)p);
    } else if (sz >= sizeof(int32_t)) {
      return static_cast<int64_t>(*((int32_t *)p));
    } else if (sz >= sizeof(int16_t)) {
      return static_cast<int64_t>(*((int16_t *)p));
    } else if (sz >= sizeof(int8_t)) {
      return static_cast<int64_t>(*((int8_t *)p));
    } else {
      return (int64_t)(p);
    }
  }
#endif

  /**
   * Returns the slot size in bytes.
   */
  inline int64_t slot_size() const {
    return sizeof(Slot::SlotHeader) + _key_size + _value_size;
  }

  inline int64_t key_size() const {
    return _key_size;
  }

  inline int64_t value_size() const {
    return _value_size;
  }

  /*
   * Returns the size of this dicitonary in *number of keys*.
   */
  int64_t size() const { return _size; }

  /*
   * Returns the capacity of this dicitonary in *number of keys*.
   */
  int64_t capacity() const { return _capacity; }

  // Returns the slot at the given index.
  inline Slot *slot_at_idx(int64_t slot_idx) const {
    return reinterpret_cast<Slot*>(_data + slot_size() * slot_idx);
  }


  // Initialize a new slot. Returns true if the initialization caused a resize.
  inline bool init_slot(Slot *slot, int32_t hash, void *key) {
    DBG("slot %p: hash=%d\n", slot, hash);
  	slot->header.filled = true;
    slot->header.hash = hash;
  	memcpy(slot->key(), key, _key_size);
  	++_size;
  	// Check if a resize is needed.
    if (_size * 10 >= _capacity * RESIZE_THRES) {
      resize(_capacity * 2);
      return true;
    }
    return false;
  }

  /**
   * Returns the slot for the given hash/key. If the dictionary does not have an entry
   * for the hash/key then its slot is initialized with the init_value and returned.
   * The dictionary may be resized as a result of initializing a new slot.
   * Returns NULL if no slot could be found or created.
   */
  Slot *upsert_slot(int32_t hash, void *key, void *init_value) {
    DBG("hash=%d, key=%lld, init_value=%lld\n",
        hash,
        dbg_print(key, _key_size),
        dbg_print(init_value, _value_size));
    bool filled;
    Slot *slot = get_slot(hash, key, &filled);
    if (!filled) {
      memcpy(slot->value(_key_size), init_value, _value_size);
    }
    return slot;
  }

  /**
   * Returns the slot for the given hash/key. If the dictionary does not have an entry
   * for the hash/key then its slot is initialized (with an uninitialized value).
   * The dictionary may be resized as a result of initializing a new slot.
   * Returns NULL if no slot could be found or created.
   *
   * If the slot was not filled and a new slot was allocated, *filled is set to
   * `true`. Otherwise, *filled is set to `false`.
   *
   * If an unfilled slot is returned, the value is guaranteed to be zeroed.
   */
  Slot *get_slot(int32_t hash, void *key, bool *filled) {
    // Linear probing.
    DBG("hash=%d, key=%lld\n",
        hash,
        dbg_print(key, _key_size));

  	int64_t start_slot_idx = hash & (_capacity - 1);
    for (int64_t i = 0; i < _capacity; ++i) {
      int64_t slot_idx = (start_slot_idx + i) & (_capacity - 1);
      Slot *slot = slot_at_idx(slot_idx);
      if (!slot->header.filled) {
        // The dictionary resized - we need to re-retrieve the slot!
        // Fortunately, it's guarnateed to be in the dictionary now.
        *filled = false;
      	if (init_slot(slot, hash, key)) {
          DBG("returned new slot for hash=%d, key=%lld\n", hash, dbg_print(key, _key_size));
          return get(hash, key);
        } else {
          DBG("returned new slot for hash=%d, key=%lld\n", hash, dbg_print(key, _key_size));
      	  return slot;
        }
      }

      /*
      if (_value_size > 8) {
        Vec<int32_t> *v = reinterpret_cast<Vec<int32_t>*>(slot->value(_key_size));
        DBG("(hack) vec len: %lld\n", v->capacity);
        for (int i = 0; i < v->capacity; i++) {
          DBG("elem %d: %d\n", i, v->data[i]);
        }
      }
      */

      // Slot is filled - check it.
      if (slot->header.hash == hash && _keys_eq(key, slot->key())) {
        *filled = true;
        DBG("returned existing slot for hash=%d, key=%lld, cur_value=%lld\n",
            hash, dbg_print(key, _key_size), dbg_print(slot->value(_key_size), _value_size));
        return slot;
      }
    }
    return nullptr;
  }

  /**
   * Returns a slot for a key.
   *
   * Raises a runtime error if the key is not in the dictionary.
   */
  Slot *get(int32_t hash, void *key) {
    // Linear probing.
    DBG("hash=%d, key=%lld\n", hash, dbg_print(key, _key_size));
  	int64_t start_slot_idx = hash & (_capacity - 1);
    for (int64_t i = 0; i < _capacity; ++i) {
      int64_t slot_idx = (start_slot_idx + i) & (_capacity - 1);
      Slot *slot = slot_at_idx(slot_idx);
      if (!slot->header.filled) {
        goto fail;
      }

      // Slot is filled - check it.
      if (slot->header.hash == hash && _keys_eq(key, slot->key())) {
        return slot;
      }
    }
fail:
    weld_runst_set_errno(_run, KeyNotFoundError);
    __builtin_unreachable();
  }

  /**
   * Returns whether the given key has an initialized slot.
   */
  int32_t keyexists(int32_t hash, void *key) {
    DBG("hash %d key %lld\n", hash, dbg_print(key, _key_size));
    // Linear probing.
  	int64_t start_slot_idx = hash & (_capacity - 1);
    for (int64_t i = 0; i < _capacity; ++i) {
      int64_t slot_idx = (start_slot_idx + i) & (_capacity - 1);
      DBG("slot_index %lld\n", slot_idx);
      Slot *slot = slot_at_idx(slot_idx);
      if (!slot->header.filled) {
        DBG("slot not found, returning 0\n");
        return 0;
      }

      // Slot is filled - check it.
      DBG("checking slot %p (arg=%d, slot=%d)\n", slot, hash, slot->header.hash);
      if (slot->header.hash == hash && _keys_eq(key, slot->key())) {
        DBG("found match for hash=%d, key=%lld\n", hash, dbg_print(key, _key_size));
        return 1;
      }
        DBG("found collision for hash=%d, key=%lld (conflict=%d, %lld) at slot\n",
            hash, dbg_print(key, _key_size),
            slot->header.hash, dbg_print(slot->key(), _key_size));
    }
    return 0;
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
  void resize(int64_t new_capacity) {
    assert(new_capacity > _capacity);
    DBG("old_capacity=%lld, new_capacity=%lld\n", _capacity, new_capacity);

    const int64_t sz = slot_size();
    WeldDict resized(_run, _key_size, _value_size, _keys_eq, new_capacity);
    assert(resized.init());

    // Copy slots to resized dictionary.
    for (int i = 0; i < _capacity; i++) {
    	Slot *old_slot = slot_at_idx(i);
    	if (!old_slot->header.filled) continue;
    	Slot *new_slot = resized.get_empty_slot(old_slot->header.hash);
    	assert(new_slot != nullptr);
    	memcpy(new_slot, old_slot, sz);
    }

    // Free the old data.
    weld_runst_free(_run, _data);

    _data = resized._data;
    _capacity = resized._capacity;
  }

  /**
   * Returns an array of { key, value } structs, potentially padded, given this
   * dictionary. The dictionary must be finalized.
   *
   * @param value_offset the offset of the value after the key. This accounts
   * for padding between the key and value in the returned struct.
   * @param struct_size the total struct size. This handles padding after the
   * value.
   *
   * @return a buffer of {key, value} pairs.
   */
  void new_kv_vector(int32_t value_offset, int32_t struct_size, void *out) {
    DBG("value_offset: %d struct_size: %d size: %lld\n", value_offset, struct_size, _size);
  	int64_t key_padding_bytes = value_offset - _key_size;
    DBG("padding: %lld\n", key_padding_bytes);
  	uint8_t *buf = reinterpret_cast<uint8_t*>(
  			weld_runst_malloc(_run, _size * struct_size));
    assert(buf);
  	memset(buf, 0, struct_size * _size);

    DBG("kv allocation size: %lld\n", struct_size * _size);

  	int64_t offset = 0;
  	for (int64_t i = 0; i < _capacity; ++i) {
  		Slot *slot = slot_at_idx(i);
  		if (!slot->header.filled) continue;
  		uint8_t *offset_buf = buf + offset * struct_size;
      DBG("copying (%lld -> %lld) to offset = %lld\n",
          dbg_print(slot->key(), _key_size),
          dbg_print(slot->value(_key_size), _value_size),
          (int64_t)(offset_buf - buf));
  		memcpy(offset_buf, slot->key(), _key_size);
  		offset_buf += _key_size + key_padding_bytes;
      memcpy(offset_buf, slot->value(_key_size), _packed_value_size);
  		offset++;
  	}
  	assert(offset == _size);

    struct retvec {
      void *pointer;
      int64_t len;
    };
    retvec *out_typed = (retvec *)out;
    out_typed->pointer = buf;
    out_typed->len = _size;
  }

  /** Serializes a dictionary, flattening pointers if necessary. */
  int64_t serialize(Vec<uint8_t> *vec,
      int32_t has_pointer,
      SerializeFn serialize_key_fn,
      SerializeFn serialize_value_fn) {
    DBG("has_pointer: %d\n", has_pointer);
  	if (has_pointer) {
      return serialize_with_pointers(vec, serialize_key_fn, serialize_value_fn);
    } else {
      return serialize_no_pointers(vec);
    }
  }
private:

  /** Serializes a dictionary, where the keys and values have pointers. */
  int64_t serialize_with_pointers(
      Vec<uint8_t> *vec,
      SerializeFn serialize_key_fn,
      SerializeFn serialize_value_fn) {

    // Offset where we write new data.
    int64_t base = vec->capacity;
    DBG("base: %lld\n", base);
    DBG("writing size %lld @ offset %lld\n", _size, vec->capacity);

    vec->extend(_run, sizeof(int64_t));
  	uint8_t *offset = vec->data + base;
    int64_t *as_i64_ptr = (int64_t *)(offset);
    *as_i64_ptr = _size;

    base += sizeof(int64_t);

    // Copy each key/value pair into the buffer.
    for (long i = 0; i < _capacity; ++i) {
    	Slot *slot = slot_at_idx(i);
    	if (!slot->header.filled) continue;
      DBG("writing key %lld @ offset %lld\n",
        dbg_print(slot->key(), _key_size), vec->capacity);
    	serialize_key_fn(vec, slot->key(), _run);
      DBG("writing value %lld @ offset %lld\n",
        dbg_print(slot->key(), _key_size), vec->capacity);
    	serialize_value_fn(vec, slot->value(_key_size), _run);
    }
    DBG("ser buf capacity: %lld\n", vec->capacity);
    return vec->capacity;
  }

  /** Serializes a dictionary, where the keys and values have no pointers. */
  int64_t serialize_no_pointers(Vec<uint8_t> *vec) {
    // Number of bytes we will write.
  	const int64_t bytes = sizeof(int64_t) + (_key_size + _packed_value_size) * _size;
    const int64_t old_capacity = vec->capacity;
  	vec->extend(_run, vec->capacity + bytes);

  	uint8_t *offset = vec->data + old_capacity;
  	int64_t *as_i64_ptr = (int64_t *)offset;
  	*as_i64_ptr = _size;
  	offset += 8;

  	// Copy each key/value pair into the buffer.
  	for (long i = 0; i < _capacity; i++) {
  		Slot *slot = slot_at_idx(i);
  		if (!slot->header.filled) continue;
  		memcpy(offset, slot->key(), _key_size);
  		offset += _key_size;
  		memcpy(offset, slot->value(_key_size), _packed_value_size);
  		offset += _packed_value_size;
  	}
  	assert(offset - vec->data == bytes);
    return vec->capacity;
  }
};

struct GroupMerger {
  // Dictionary that holds the K -> vec[V] mappings.
  WeldDict *dict;
  // Size of the value in bytes.
  int32_t value_size;
};


extern "C" void *weld_st_dict_new(
    WeldRunHandleRef run,
		int32_t key_size,
		int32_t value_size,
    KeyComparator keys_eq,
    int64_t capacity) {

  DBG("run: %p keysz: %d valsz: %d eq: %p cap: %lld\n",
      run, key_size, value_size, keys_eq, capacity);

  WeldDict *wd = reinterpret_cast<WeldDict*>(
  		weld_runst_malloc(run, sizeof(WeldDict)));
  assert(wd);
  new (wd) WeldDict(run, key_size, value_size, keys_eq, capacity);
  assert(wd->init());
  return wd;
}

extern "C" void weld_st_dict_free(WeldRunHandleRef run, void *d) {
  WeldDict *wd = (WeldDict *)d;
  wd->free_weld_dict();
  weld_runst_free(run, wd);
}

extern "C" void *weld_st_dict_upsert(
    WeldRunHandleRef run,
    void *d,
    void *key,
    int32_t hash,
    void *init_value) {
  WeldDict *wd = static_cast<WeldDict*>(d);
  return wd->upsert_slot(hash, key, init_value);
}

extern "C" void *weld_st_dict_get(
    WeldRunHandleRef run,
    void *d,
    void *key,
    int32_t hash) {
  WeldDict *wd = static_cast<WeldDict*>(d);
  return wd->get(hash, key);
}

extern "C" void *weld_st_dict_get_slot(
    WeldRunHandleRef run,
    void *d,
    void *key,
    int32_t hash) {
  bool filled;
  WeldDict *wd = static_cast<WeldDict*>(d);
  return wd->get_slot(hash, key, &filled);
}

extern "C" int32_t weld_st_dict_keyexists(
    WeldRunHandleRef run,
    void *d,
    void *key,
    int32_t hash) {
  WeldDict *wd = static_cast<WeldDict*>(d);
  return wd->keyexists(hash, key);
}

extern "C" int64_t weld_st_dict_size(WeldRunHandleRef run, void *d) {
  WeldDict *wd = static_cast<WeldDict*>(d);
  return wd->size();
}

extern "C" void weld_st_dict_tovec(
    WeldRunHandleRef run,
    void *d,
    int32_t value_offset,
    int32_t struct_size,
    void *out) {
  DBG("run: %p, dict: %p, offset: %d, struct_size: %d\n",
      run, d, value_offset, struct_size);
  WeldDict *wd = static_cast<WeldDict*>(d);
  wd->new_kv_vector(value_offset, struct_size, out);
}

extern "C" int64_t weld_st_dict_serialize(
    WeldRunHandleRef run,
    void *d,
    void *buf,
    int32_t has_pointer,
    void *key_ser,
    void *val_ser) {
  WeldDict *wd = static_cast<WeldDict*>(d);
  Vec<uint8_t> *serbuf = static_cast<Vec<uint8_t>*>(buf);
  return wd->serialize(
      serbuf,
      has_pointer,
  		reinterpret_cast<SerializeFn>(key_ser),
      reinterpret_cast<SerializeFn>(val_ser));
}

// -------------------------------------------------------------------------------

extern "C" void *weld_st_gb_new(
    WeldRunHandleRef run,
    int32_t key_size,
    int32_t val_size,
    KeyComparator keys_eq,
    int64_t capacity) {

  GroupMerger *gm = reinterpret_cast<GroupMerger*>(
  		weld_runst_malloc(run, sizeof(GroupMerger)));
  assert(gm);

  gm->dict = reinterpret_cast<WeldDict*>(weld_st_dict_new(
      run,
      key_size,
      sizeof(GrowableWeldVec),
      keys_eq,
      capacity));
  gm->dict->set_packed_value_size(sizeof(WeldVec));
  gm->value_size = val_size;
  return gm;
}

extern "C" void weld_st_gb_merge(
    WeldRunHandleRef run,
    void *p,
    void *key,
    int32_t hash,
    void *value) {

  GroupMerger *gm = (GroupMerger *)p;
  bool filled;
  Slot *s = gm->dict->get_slot(hash, key, &filled);

  GrowableWeldVec *vec = reinterpret_cast<GrowableWeldVec*>(s->value(gm->dict->key_size()));
  DBG("Accessed growable vec for key\n");

  if (filled && vec->capacity == vec->wv.size) {
    vec->resize(run, gm->value_size, vec->capacity * 2);
    DBG("Resizing vec from %lld to %lld elem capacity\n", vec->capacity, vec->capacity * 2);
  }

  if (!filled) {
    DBG("initializing vec\n");
    vec->capacity = INITIAL_CAPACITY;
    vec->wv.data = weld_runst_malloc(run, INITIAL_CAPACITY * gm->value_size);
    vec->wv.size = 0;
  }
  memcpy(vec->at(gm->value_size, vec->wv.size), value, gm->value_size);
  vec->wv.size++;
}

extern "C" void *weld_st_gb_result(
    WeldRunHandleRef run,
    void *p) {
  GroupMerger *gm = (GroupMerger *)p;
  WeldDict *result = gm->dict;
  weld_runst_free(run, gm);
  return result;
}
