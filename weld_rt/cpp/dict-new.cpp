#include "runtime.h"

#include <algorithm>
#include <assert.h>

// The number of slots to lock at once.
const int LOCK_GRANULARITY = 16;

// Resize threshold. Resizes when RESIZE_THRES * 10 percent of slots are full.
const int RESIZE_THRES = 7;

// A callback which checks whether two keys are equal. Returns a non-zero value
// if the keys are equal.
typedef int32_t (*KeyComparator)(void *, void *);

// Protects a single slot.
typedef uint8_t* SlotLock;

// Locking constants.
const SlotLock UNLOCKED = 0;
const SlotLock LOCKED = 1;

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

    inline void *value(size_t key_size) {
      return reinterpret_cast<uint8_t *>(this) + sizeof(header) + key_size;
    }

    inline void acquire_access() {
      while (!__sync_bool_compare_and_swap(&lockvar, UNLOCKED, LOCK));
    }

    inline void release_access() {
      lockvar = 0;
    }
};

// An internal dictionary which provides (optional) locked access to its slots.
class InternalDict {

  public:

    const int64_t key_size;
    const int64_t value_size;
    const int64_t max_capacity;

  private:

    // An array of Slots.
    void *data;
    volatile int64_t size;
    volatile int64_t capacity;
    bool full;

    // Global lock for dictionaries accessed concurrently.
    pthread_rwlock_t global_lock;

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
      int64_t index = ((intptr_t)slot - (intptr_t)data) / slot_size();
      index &= ~(LOCK_GRANULARITY - 1);

      return slot_at_index(index);
    }

    /** Returns the slot at the given index. */
    inline void *slot_at_index(long index) {
      return ((uint8_t *)data) + parent->slot_size() * index;
    }

  public:

        // N dictionaries for the N workers, + 1 for the global dictionary.
        dicts_ = weld_rt_new_merger(sizeof(InternalDict), workers_ + 1);
        // Initialize the thread-local dictionaries.
        for (int i = 0; i < workers_ + 1; i++) {
          InternalDict *dict = dict_at_index(i);
          size_t data_size = capacity * slot_size();

          dict->size = 0;
          dict->capacity = capacity;
          dict->data = weld_run_malloc(weld_rt_get_run_id(), data_size);
          memset(dict->data, 0, data_size);
          dict->full = max_local_bytes_ == 0;
        }

    InternalDict(int64_t key_size, int64_t value_size, int64_t capacity, int64_t max_capacity):
      key_size(key_size),
      value_size(value_size),
      capacity(capacity),
      max_capacity(max_capacity) {

        // Power of 2 check.
        assert(capacity > 0 && (capacity & (capacity - 1)) == 0);
        size_t data_size = capacity * slot_size();

        data = weld_run_malloc(weld_rt_get_run_id(), data_size);
        memset(data, 0, data_size);
        full = max_capacity == 0;

        pthread_rwlock_init(&global_lock, NULL);
      }

    ~InternalDict() {
      pthread_rwlock_destroy(&wd->global_lock);
      free(data);
    }

    int64_t size() {
      return size;
    }

    int64_t capacity() {
      return capacity;
    }

    bool full() {
      return full;
    }

    /** Get a slot for a given hash and key, optionally locking it for concurrent access.
    */
    Slot *get_slot(int32_t hash, void *key, bool lock, bool check_key) {

      if (lock) {
        pthread_rwlock_rdlock(&global_lock);
      }

      int64_t first_offset = hash & (capacity - 1);
      Slot *locked_slot = NULL;

      // Linear probing.
      for (long i = 0; i < capacity; i++) {
        long index = (first_offset + 1) & (dict->capacity - 1);
        Slot *slot = slot_at_index(index);

        // Lock the slot to disable concurrent access on the LOCK_GRANULARITY buckets we probe.
        if (lock && (i == 0 || lockable_index(i))) {
          if (locked_slot != NULL) {
            locked_slot->release_access(); 
          }
          slot.acquire_access();
          locked_slot = slot;
        }

        if (slot->filled && check_key && slot->hash == hash && keys_eq(key, slot->key())) {
          assert(locked_slot);
          return slot;
        } else {
          assert(locked_slot);
          slot->hash = hash;
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
      bool was_filled = slot->filled;
      slot->filled = 1;

      // If the slot was just updated, we don't have to worry about resizing etc.
      // Unlock the slot (if its locked) and return.
      if (was_filled) {
        if (locked) {
          Slot *locked_slot = lock_for_slot(slot);
          locked_slot->release_access();
          pthread_rwlock_unlock(&global_lock);
        }
        return full;
      }

      if (locked) {
        Slot *locked_slot = lock_for_slot(slot);
        locked_slot->release_access();
        __sync_fetch_and_add(size, 1);

        // If we need to resize, upgrade the read lock we have to a write lock.
        if (should_resize(size)) {
          pthread_rwlock_unlock(&global_lock);
          pthread_rwlock_wrlock(&global_lock);

          if (should_resize(size)) {
            resize();
          }
        }
        // Unlock the Read lock if we didn't resize, or the write lock if we did.
        pthread_rwlock_unlock(&global_lock);
      } else {
        size++;
        if should_resize(size) {
          resize();
        } else if (should_resize(size + 1) && capacity * 2 * slot_size() > max_capacity) {
          full = true;
          set_full = true;
        }
      }
      return full;
    }

    /** Returns whether this dictionary should be resized when it reaches `new_size`. */
    inline bool should_resize(size_t new_size) {
      return new_size * 10 >= capacity * RESIZE_THRES; 
    }

    /** Resizes the dictionary. */
    void resize() {
      const size_t sz = slot_size();
      InternalDict resized;
      resized.size = size;
      resized.capacity = capacity * 2;
      resized.data = weld_run_malloc(weld_rt_get_run_id(), resized.capacity * sz);
      memset(resized.data, 0, resized.capacity * sz);

      for (int i = 0; i < capacity; i++) {
        Slot *old_slot = slot_at_index(i);
        if (old_slot->filled) {
          Slot *new_slot = resized.get_slot(old_slot->hash, NULL, false, false);
          assert(new_slot != NULL);
          memcpy(new_slot, old_slot, sz);
        }
      }

      // Free the old data.
      free(data);

      data = resized.data;
      size = resized.size;
      capacity = resized.capacity;
      full = false;
    }
};















/** A multi-threaded dictionary used by the Weld runtime. */
class WeldDict {

  private:


    // A slot is a buffer with a SlotHeader, followed by the key, followed by the value.
    typedef SlotHeader* Slot;

    // A single Dict.
    struct InternalDict {
      // An array of Slots.
      void *data;
      volatile int64_t size;
      volatile int64_t capacity;
      bool full;

      /** Returns the slot at the given index. */
      inline void *slot_at_index(const WeldDict *const parent, long index) {
        return ((uint8_t *)data) + parent->slot_size() * index;
      }

      /** Returns the lock to acquire for a given slot, taking into account the locking
       * granularity. */
      inline SlotLock slot_lock_granular(const WeldDict *const parent, Slot slot) {
        long index = ((intptr_t)slot - (intptr_t)data) / parent->slot_size();
        //  This is the slot lock we will use.
        Slot locked_slot = slot_at_index(parent, index);
        return slot_lock(locked_slot);
      }

      /** Returns whether this dictionary should be resized. */
      inline bool should_resize() {
        return size * 10 >= capacity * RESIZE_THRES; 
      }

      /** Resizes the dictionary. */
      void resize(const WeldDict *const parent) {
        const size_t sz = parent->slot_size();
        InternalDict resized;
        resized.size = size;
        resized.capacity = capacity * 2;
        resized.data = weld_run_malloc(weld_rt_get_run_id(), sized.capacity * sz);
        memset(resized.data, 0, resized.capacity * sz);

        for (int i = 0; i < capacity; i++) {
          Slot old_slot = slot_at_index(parent, i);
          if (old_slot->filled) {
            Slot new_slot = parent->lookup_and_lock_internal(&resized, old_slot->hash, NULL, false, false);
            assert(new_slot != NULL);
            memcpy(new_slot, old_slot, sz);
          }
        }

        void *old_data = data;
        free(old_data);

        size = resized.size;
        capacity = resized.capacity;
        data = resized.data;
        full = false;
      }
    };

    // The contiguous buffer of dictionaries. If there are N workers, there are
    // N + 1 dictionaries, where index i holds the dictionary for thread ID i,
    // and dictionary N+1 is the global dictionary.
    void *dicts_;
    KeyComparator keys_eq_;
    int32_t workers_;
    int32_t key_size_;
    int32_t value_size_;
    int64_t max_local_bytes_;

    bool finalized_;
    pthread_rwlock_t global_lock_;

  private:

    /////////////////////////////////////////////////////////////
    //
    //                    Accessor Macros.
    //
    /////////////////////////////////////////////////////////////

    /** Get the dictionary at the given index.
     *
     * @i the index.
     * @return the dictionary the index.
     */
    inline InternalDict *dict_at_index(int32_t i) {
      // TODO can we replaced this with a aligned malloc or something?
      return weld_rt_get_merger_at_index(dicts_, sizeof(InternalDict), i);
    }

    /** Get the dictionary for the calling thread. */
    inline InternalDict *local() {
      return dict_at_index(weld_rt_thread_id());
    }

    /** Get the global dictionary. */
    inline InternalDict *global() {
      return dict_at_index(workers_);
    }

    /////////////////////////////////////////////////////////////
    //
    //                          Slots.
    //
    /////////////////////////////////////////////////////////////

    // A slot is the header, the key, and the value.
    inline size_t slot_size() {
      return sizeof(SlotHeader) + key_size_ + value_size_;
    }

    /** Get the key for a particular slot. */
    inline void *slot_key(Slot slot) {
      return slot + 1;
    }

    /** Get the value for a particular slot. */
    inline void *slot_value(Slot slot) {
      return ((uint8_t *)(slot + 1)) + key_size_;
    }

    /** Get the SlockLock for a given slot. */
    inline SlotLock slot_lock(Slot slot) {
      return &slot->lockvar;
    }

    inline bool lockable_slot_idx(int64_t idx) {
      return (idx & (LOCK_GRANULARITY - 1)) == 0;
    }

    /////////////////////////////////////////////////////////////
    //
    //                      Instance Methods.
    //
    /////////////////////////////////////////////////////////////
  
  public:

    WeldDict(int32_t key_size,
        int32_t value_size,
        KeyComparator keys_eq,
        int32_t max_local_bytes,
        int64_t capacity):
      key_size_(key_size),
      value_size_(value_size),
      keys_eq_(keys_eq),
      max_local_bytes_(max_local_bytes),
      workers_(weld_rt_get_nworkers()) {

        // Check that capacity is a power of 2.
        assert(capacity_ > 0 && (capacity_ & (capacity - 1)) == 0);

        // N dictionaries for the N workers, + 1 for the global dictionary.
        dicts_ = weld_rt_new_merger(sizeof(InternalDict), workers_ + 1);
        // Initialize the thread-local dictionaries.
        for (int i = 0; i < workers_ + 1; i++) {
          InternalDict *dict = dict_at_index(i);
          size_t data_size = capacity * slot_size();

          dict->size = 0;
          dict->capacity = capacity;
          dict->data = weld_run_malloc(weld_rt_get_run_id(), data_size);
          memset(dict->data, 0, data_size);
          dict->full = max_local_bytes_ == 0;
        }
        pthread_rwlock_init(&wd->global_lock, NULL);
      }

    ~WeldDict() {
      for (int i = 0; i < workers_ + 1; i++) {
        InternalDict *dict = dict_at_index(i);
        weld_rt_free(weld_rt_get_run_id(), dict->data);
      }
      weld_rt_free_merger(dicts_);
    }

    /** Looks up a value in an InternalDict. 
     *
     * @param dict the InternalDict instance to search
     * @hash the hash of the key.
     * @key the key data, which is a buffer of size `key_size`.
     * @match_possible a flag which indicates whether a match is possible when searching. Used to avoid a key
     * comparison during resize operations.
     * @lock_global_slots a flag which indicates whether to lock the global slots.
     *
     * @return a pointer to the value if found, or NULL. The size of the value should not
     * change while it is in the dictionary.
     *
     */
    Slot get_slot(InternalDict *dict,
        int32_t hash,
        void *key,
        bool match_possible,
        bool lock_global_slots) {

      // Is the operation on the global dictionary?
      bool global = dict == global();

      int64_t first_offset = hash & (dict->capacity - 1);
      SlotLock prev_lock = NULL;

      for (long i = 0; i < dict->capacity; i++) {
        long index = (first_offset + 1) & (dict->capacity - 1);
        Slot current_slot = dict->slot_at_index(this, index);

        // TODO wat.
        if (!finalized_ && global && lock_global_slots && (i == 0 || lockable_slot_index(index))) {
          if (prev_lock != NULL) {
            *prev_lock = UNLOCKED;
          }
          prev_lock = dict->slot_lock_granular(this, current_slot);
          while (!__sync_bool_compare_and_swap(prev_lock, UNLOCKED, LOCK));
        }

        if (current_slot->filled) {
          if (current_slot->filled && match_possible && current_slot->hash == hash && keys_eq(key, slot_key(current_slot))) {
            return current_slot;
          } else {
            // Fill the hash in case this slot is filled -- this has no effect if the slot is not filled.
            current_slot->hash = hash;
            return current_slot;
          }
        }
      }
      
      // Release the lock.
      if (prev_lock != NULL) {
        *prev_lock = UNLOCKED;
      }
    }

    void weld_rt_dict_put(Slot slot) {
      weld_dict *wd = (weld_dict *)d;

      bool was_filled = slot->filled;
      if (!was_filled) {
        slot->filled = 1;
        // Update the local dictionary, and mark it as full if necessary.
        if (slot_in_local(slot)) {
          InternalDict *local_dict = local();
          local_dict->size++;
          if (local_dict->should_resize(local_dict->size)) {
            local->resize(this);
          } else if (local_dict->should_resize(local_dict->size + 1) &&
              local_dict->capacity * 2 * slot_size() > max_local_bytes_) {
            local_dict->full = true;
          }
        } else {
          InternalDict *global_dict = global();
          if (!finalized_) {
            SlotLock lock = global_dict->slot_lock_granular(this, slot);
            *lock = UNLOCKED;
            __sync_fetch_and_add(&global->size, 1);
          } else {
            global->size++;
          }

          // Resize the global if necessary.
          if (global->should_resize(global->size)) {
            if (!finalized_) {
              pthread_rwlock_unlock(&global_lock_);
              pthread_rwlock_wrlock(&global_lock_);
            }

            if (global->should_resize(global->size)) {
              global->resize(this);
            }
          }
          if (!finalized_) {
            pthread_rwlock_unlock(&global_lock_);
          }
        }
      } else if (!finalized_ && !slot_in_local(slot)) {
          InternalDict *global_dict = global();
          SlotLock lock = global_dict->slot_lock_granular(this, slot);
          *lock = UNLOCKED;
          pthread_rwlock_unlock(&global_lock_);
      }
    }
}

