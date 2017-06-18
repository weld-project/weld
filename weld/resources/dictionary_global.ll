; Template for a dictionary and its helper functions. Uses linear probing.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - KEY: LLVM type of key (e.g. i32 or %MyStruct)
; - VALUE: LLVM type of value (e.g. i32 or %MyStruct)
; - KEY_PREFIX: prefix for helper functions of key (e.g. @i32 or @MyStruct)
; - KV_STRUCT: name of struct holding {KEY, VALUE} (should be generated outside)
; - KV_VEC: name of vector of KV_STRUCTs (should be generated outside)
; - KV_VEC_PREFIX: prefix for helper functions of KV_VEC
; - SLOT_STRUCT: struct of (i1, key, value) tuples
; - SLOT_VEC: name of vector of SLOT_STRUCTs (should be generated outside)
; - SLOT_VEC_PREFIX: prefix for helper functions of SLOT_VEC

%$NAME.entry = type { i32, i32, $SLOT_VEC }       ; is_locked, size of chain, (i1, key, value) chained list
%$NAME.slot = type %$NAME.entry*           ; handle to an entry in the API
%$NAME = type { %$NAME.entry*, i64, i64 }  ; entries, size, num_buckets

; Initialize and return a new dictionary with the given initial num_buckets.
; The num_buckets must be a power of 2.
define %$NAME @$NAME.new(i64 %num_buckets) {
  %entrySizePtr = getelementptr %$NAME.entry, %$NAME.entry* null, i32 1
  %entrySize = ptrtoint %$NAME.entry* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %num_buckets
  %runId = call i64 @get_runid()
  %bytes = call i8* @weld_rt_malloc(i64 %runId, i64 %allocSize)
  ; Memset all the bytes to 0 to set the isFilled fields to 0
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  %entries = bitcast i8* %bytes to %$NAME.entry*
  br label %entry

entry:
  %cond = icmp ult i64 0, %num_buckets
  br i1 %cond, label %body, label %done

body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %ptr = getelementptr %$NAME.entry, %$NAME.entry* %entries, i64 %i
  %chain_size_ptr = getelementptr %$NAME.entry, %$NAME.entry* %ptr, i64 0, i32 1
  store i32 16, i32* %chain_size_ptr
  %vec_ptr = getelementptr %$NAME.entry, %$NAME.entry* %ptr, i64 0, i32 2
  %vec = call $SLOT_VEC $SLOT_VEC_PREFIX.new(i64 16)
  call void $SLOT_VEC_PREFIX.zero($SLOT_VEC %vec)
  store $SLOT_VEC %vec, $SLOT_VEC* %vec_ptr
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %num_buckets
  br i1 %cond2, label %body, label %done

done:
  %1 = insertvalue %$NAME undef, %$NAME.entry* %entries, 0
  %2 = insertvalue %$NAME %1, i64 0, 1
  %3 = insertvalue %$NAME %2, i64 %num_buckets, 2
  ret %$NAME %3
}

; Free dictionary
define void @$NAME.free(%$NAME %dict) {
  %runId = call i64 @get_runid()
  %entries = extractvalue %$NAME %dict, 0
  %num_buckets = extractvalue %$NAME %dict, 2
  br label %entry

entry:
  %cond = icmp ult i64 0, %num_buckets
  br i1 %cond, label %body, label %done

body:
  %i = phi i64[ 0, %entry ], [ %i2, %body ]
  %ptr = getelementptr %$NAME.entry, %$NAME.entry* %entries, i64 %i
  %vec_ptr = getelementptr %$NAME.entry, %$NAME.entry* %ptr, i64 0, i32 2
  %vec_bytes = bitcast $SLOT_VEC* %vec_ptr to i8*
  call void @weld_rt_free(i64 %runId, i8* %vec_bytes)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %num_buckets
  br i1 %cond2, label %body, label %done

done:
  %bytes = bitcast %$NAME.entry* %entries to i8*
  call void @weld_rt_free(i64 %runId, i8* %bytes)
  ret void
}

; Clone a dictionary.
define %$NAME @$NAME.clone(%$NAME %dict) {
  %entries = extractvalue %$NAME %dict, 0
  %size = extractvalue %$NAME %dict, 1
  %num_buckets = extractvalue %$NAME %dict, 2
  %entrySizePtr = getelementptr %$NAME.entry, %$NAME.entry* null, i32 1
  %entrySize = ptrtoint %$NAME.entry* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %num_buckets
  %bytes = bitcast %$NAME.entry* %entries to i8*
  %dict2 = call %$NAME @$NAME.new(i64 %num_buckets)
  %entries2 = extractvalue %$NAME %dict2, 0
  %bytes2 = bitcast %$NAME.entry* %entries2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  %dict3 = insertvalue %$NAME %dict2, i64 %size, 1
  ret %$NAME %dict3
}

; Dummy hash function; this is needed for structs that use these dictionaries as fields,
; but it doesn't yet work! (Or technically it does, but is a very poor function.)
define i64 @$NAME.hash(%$NAME %dict) {
  ret i64 0
}

; Dummy comparison function; this is needed for structs that use these dictionaries as fields,
; but it doesn't yet work!
define i32 @$NAME.cmp(%$NAME %dict1, %$NAME %dict2) {
  ret i32 -1
}

; Get the size of a dictionary.
define i64 @$NAME.size(%$NAME %dict) {
  %size = extractvalue %$NAME %dict, 1
  ret i64 %size
}

; Try to lock the slot.
define i1 @$NAME.slot.try_lock(%$NAME.slot %slot) {
  %ptr = getelementptr %$NAME.entry, %$NAME.slot %slot, i64 0, i32 0
  %val_success = cmpxchg i32* %ptr, i32 0, i32 1 acq_rel monotonic
  %success = extractvalue { i32, i1 } %val_success, 1
  ret i1 %success
}

; Lock the slot.
define void @$NAME.slot.lock(%$NAME.slot %slot) {
  %success = call i1 @$NAME.slot.try_lock(%$NAME.slot %slot)
  br i1 %success, label %end, label %start

start:
  %success2 = call i1 @$NAME.slot.try_lock(%$NAME.slot %slot)
  br i1 %success2, label %end, label %start

end:
  ret void
}

; Unlock the slot.
define void @$NAME.slot.unlock(%$NAME.slot %slot) {
  %ptr = getelementptr %$NAME.entry, %$NAME.slot %slot, i64 0, i32 0
  cmpxchg i32* %ptr, i32 1, i32 0 acq_rel monotonic
  ret void
}

; Check whether a slot is filled.
define i1 @$NAME.slot.filled($SLOT_STRUCT* %slot) {
  %filledPtr = getelementptr $SLOT_STRUCT, $SLOT_STRUCT* %slot, i64 0, i32 0
  %filled = load i1, i1* %filledPtr
  ret i1 %filled
}

; Get the key for a slot (only valid if filled).
 define $KEY @$NAME.slot.key($SLOT_STRUCT* %slot) {
   %keyPtr = getelementptr $SLOT_STRUCT, $SLOT_STRUCT* %slot, i64 0, i32 1
   %key = load $KEY, $KEY* %keyPtr
   ret $KEY %key
 }

; Get the value for a slot (only valid if filled).
define $VALUE @$NAME.slot.value($SLOT_STRUCT* %slot) {
  %valuePtr = getelementptr $SLOT_STRUCT, $SLOT_STRUCT* %slot, i64 0, i32 2
  %value = load $VALUE, $VALUE* %valuePtr
  ret $VALUE %value
}

; Look up the given key, returning a slot for it. The slot functions may be
; used to tell whether the entry is filled, get its value, etc, and the put()
; function may be used to put a new value into the slot.
define $SLOT_STRUCT* @$NAME.lookup(%$NAME %dict, $KEY %key) {
entry:
  %entries = extractvalue %$NAME %dict, 0
  %num_buckets = extractvalue %$NAME %dict, 2
  %mask = sub i64 %num_buckets, 1
  %hash = call i64 $KEY_PREFIX.hash($KEY %key)
  %pos = and i64 %hash, %mask
  %slot = getelementptr %$NAME.entry, %$NAME.entry* %entries, i64 %pos
  %vec_ptr = getelementptr %$NAME.entry, %$NAME.entry* %slot, i64 0, i32 2
  %vec = load $SLOT_VEC, $SLOT_VEC* %vec_ptr
  %chain_size_ptr = getelementptr %$NAME.entry, %$NAME.entry* %slot, i64 0, i32 1
  %chain_size = load i32, i32* %chain_size_ptr
  br label %body

body:
  %i = phi i64 [ 0, %entry ], [ %i2, %true_cond ]
  %element = call $SLOT_STRUCT* $SLOT_VEC_PREFIX.at($SLOT_VEC %vec, i64 %i)
  %is_filled = call i1 @$NAME.slot.filled($SLOT_STRUCT* %element)
  br i1 %is_filled, label %true_cond, label %false_cond

true_cond:
  %element_key = call $KEY @$NAME.slot.key($SLOT_STRUCT* %element)
  %i2 = add i64 %i, 1
  %cmp = call i32 $KEY_PREFIX.cmp($KEY %key, $KEY %element_key)
  %keys_equal = icmp eq i32 %cmp, 0
  br i1 %keys_equal, label %false_cond, label %body

false_cond:
  ret $SLOT_STRUCT* %element
}

; Set the key and value at a given slot. The slot is assumed to have been
; returned by a lookup() on the same key provided here, and any old value for
; the key will be replaced. A new %$NAME is returned reusing the same storage.
define %$NAME @$NAME.put(%$NAME %dict, $SLOT_STRUCT* %slot, $KEY %key, $VALUE %value) {
start:
  %size = extractvalue %$NAME %dict, 1
  %filledPtr = getelementptr $SLOT_STRUCT, $SLOT_STRUCT* %slot, i64 0, i32 0
  %filled = load i1, i1* %filledPtr
  %keyPtr = getelementptr $SLOT_STRUCT, $SLOT_STRUCT* %slot, i64 0, i32 1
  %valuePtr = getelementptr $SLOT_STRUCT, $SLOT_STRUCT* %slot, i64 0, i32 2
  br i1 %filled, label %update, label %addNew

update:
  store i1 1, i1* %filledPtr
  store $KEY %key, $KEY* %keyPtr
  store $VALUE %value, $VALUE* %valuePtr
  ret %$NAME %dict

addNew:
  ; Add the entry into the empty slot
  store i1 1, i1* %filledPtr
  store $KEY %key, $KEY* %keyPtr
  store $VALUE %value, $VALUE* %valuePtr
  %incSize = add i64 %size, 1
  %dict2 = insertvalue %$NAME %dict, i64 %incSize, 1
  ret %$NAME %dict2
}

; Get the entries of a dictionary as a vector.
define $KV_VEC @$NAME.tovec(%$NAME %dict) {
entry:
  %entries = extractvalue %$NAME %dict, 0
  %size = extractvalue %$NAME %dict, 1
  %num_buckets = extractvalue %$NAME %dict, 2
  %vec = call $KV_VEC $KV_VEC_PREFIX.new(i64 %size)
  br label %body

body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body3 ], [ %i2, %body4 ]
  %j = phi i64 [ 0, %entry ], [ %j2, %body3 ], [ %j2, %body4 ]
  %i2 = add i64 %i, 1
  %comp = icmp uge i64 %i, %num_buckets
  br i1 %comp, label %done, label %body2

body2:
  %entPtr = getelementptr %$NAME.entry, %$NAME.entry* %entries, i64 %i
  %ent = load %$NAME.entry, %$NAME.entry* %entPtr
  %chain_vec = extractvalue %$NAME.entry %ent, 2
  %chain_size = extractvalue %$NAME.entry %ent, 1
  %chain_size_i64 = zext i32 %chain_size to i64
  br label %body3

body3:
  %k = phi i64 [ 0, %body2 ], [ %k2, %body5 ]
  %j2 = phi i64 [ %j, %body2 ], [ %j3, %body5 ]
  %k2 = add i64 %k, 1
  %comp2 = icmp ult i64 %k, %chain_size_i64
  br i1 %comp2, label %body4, label %body

body4:
  %slot_ptr = call $SLOT_STRUCT* $SLOT_VEC_PREFIX.at($SLOT_VEC %chain_vec, i64 %k)
  %filled = call i1 @$NAME.slot.filled($SLOT_STRUCT* %slot_ptr)
  br i1 %filled, label %body5, label %body

body5:
  %elemPtr = call $KV_STRUCT* $KV_VEC_PREFIX.at($KV_VEC %vec, i64 %j)
  %slot = load $SLOT_STRUCT, $SLOT_STRUCT* %slot_ptr
  %key = extractvalue $SLOT_STRUCT %slot, 1
  %value = extractvalue $SLOT_STRUCT %slot, 2
  %kv = insertvalue $KV_STRUCT undef, $KEY %key, 0
  %kv2 = insertvalue $KV_STRUCT %kv, $VALUE %value, 1
  store $KV_STRUCT %kv2, $KV_STRUCT* %elemPtr
  %j3 = add i64 %j, 1
  br label %body3

done:
  ret $KV_VEC %vec
}
