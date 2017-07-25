; Template for a dictionary and its helper functions. Uses linear probing.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - KEY: LLVM type of key (e.g. i32 or %MyStruct)
; - VALUE: LLVM type of value (e.g. i32 or %MyStruct)
; - KEY_PREFIX: prefix for helper functions of key (e.g. @i32 or @MyStruct)
; - KV_STRUCT: name of struct holding {{KEY, VALUE}} (should be generated outside)
; - KV_VEC: name of vector of KV_STRUCTs (should be generated outside)
; - KV_VEC_PREFIX: prefix for helper functions of KV_VEC

%{NAME}.entry = type {{ i1, {KEY}, {VALUE} }}        ; isFilled, key, value
%{NAME}.slot = type %{NAME}.entry*                ; handle to an entry in the API
%{NAME} = type {{ %{NAME}.entry*, i64, i64 }}  ; entries, size, capacity

; Initialize and return a new dictionary with the given initial capacity.
; The capacity must be a power of 2.
define %{NAME} @{NAME}.new(i64 %capacity) {{
  %entrySizePtr = getelementptr %{NAME}.entry, %{NAME}.entry* null, i32 1
  %entrySize = ptrtoint %{NAME}.entry* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %capacity
  %runId = call i64 @weld_rt_get_run_id()
  %bytes = call i8* @weld_run_malloc(i64 %runId, i64 %allocSize)
  ; Memset all the bytes to 0 to set the isFilled fields to 0
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  %entries = bitcast i8* %bytes to %{NAME}.entry*
  %1 = insertvalue %{NAME} undef, %{NAME}.entry* %entries, 0
  %2 = insertvalue %{NAME} %1, i64 0, 1
  %3 = insertvalue %{NAME} %2, i64 %capacity, 2
  ret %{NAME} %3
}}

; Free dictionary
define void @{NAME}.free(%{NAME} %dict) {{
  %runId = call i64 @weld_rt_get_run_id()
  %entries = extractvalue %{NAME} %dict, 0
  %bytes = bitcast %{NAME}.entry* %entries to i8*
  call void @weld_run_free(i64 %runId, i8* %bytes)
  ret void
}}

; Clone a dictionary.
define %{NAME} @{NAME}.clone(%{NAME} %dict) {{
  %entries = extractvalue %{NAME} %dict, 0
  %size = extractvalue %{NAME} %dict, 1
  %capacity = extractvalue %{NAME} %dict, 2
  %entrySizePtr = getelementptr %{NAME}.entry, %{NAME}.entry* null, i32 1
  %entrySize = ptrtoint %{NAME}.entry* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %capacity
  %bytes = bitcast %{NAME}.entry* %entries to i8*
  %dict2 = call %{NAME} @{NAME}.new(i64 %capacity)
  %entries2 = extractvalue %{NAME} %dict2, 0
  %bytes2 = bitcast %{NAME}.entry* %entries2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  %dict3 = insertvalue %{NAME} %dict2, i64 %size, 1
  ret %{NAME} %dict3
}}

; Dummy hash function; this is needed for structs that use these dictionaries as fields,
; but it doesn't yet work! (Or technically it does, but is a very poor function.)
define i32 @{NAME}.hash(%{NAME} %dict) {{
  ret i32 0
}}

; Dummy comparison function; this is needed for structs that use these dictionaries as fields,
; but it doesn't yet work!
define i32 @{NAME}.cmp(%{NAME} %dict1, %{NAME} %dict2) {{
  ret i32 -1
}}

; Dummy equality function (should call cmp when that is fully implemented)
define i1 @{NAME}.eq(%{NAME} %dict1, %{NAME} %dict2) {{
  ret i1 0
}}

; Get the size of a dictionary.
define i64 @{NAME}.size(%{NAME} %dict) {{
  %size = extractvalue %{NAME} %dict, 1
  ret i64 %size
}}

; Check whether a slot is filled.
define i1 @{NAME}.slot.filled(%{NAME}.slot %slot) {{
  %filledPtr = getelementptr %{NAME}.entry, %{NAME}.slot %slot, i64 0, i32 0
  %filled = load i1, i1* %filledPtr
  ret i1 %filled
}}

; Get the key for a slot (only valid if filled).
define {KEY} @{NAME}.slot.key(%{NAME}.slot %slot) {{
  %keyPtr = getelementptr %{NAME}.entry, %{NAME}.slot %slot, i64 0, i32 1
  %key = load {KEY}, {KEY}* %keyPtr
  ret {KEY} %key
}}

; Get the value for a slot (only valid if filled).
define {VALUE} @{NAME}.slot.value(%{NAME}.slot %slot) {{
  %valuePtr = getelementptr %{NAME}.entry, %{NAME}.slot %slot, i64 0, i32 2
  %value = load {VALUE}, {VALUE}* %valuePtr
  ret {VALUE} %value
}}

; Look up the given key, returning a slot for it. The slot functions may be
; used to tell whether the entry is filled, get its value, etc, and the put()
; function may be used to put a new value into the slot.
define %{NAME}.slot @{NAME}.lookup(%{NAME} %dict, {KEY} %key) {{
entry:
  %entries = extractvalue %{NAME} %dict, 0
  %capacity = extractvalue %{NAME} %dict, 2
  %mask = sub i64 %capacity, 1
  %raw_hash = call i32 {KEY_PREFIX}.hash({KEY} %key)
  %finalized_hash = call i32 @hash_finalize(i32 %raw_hash)
  %hash = zext i32 %raw_hash to i64
  br label %body

body:
  %h = phi i64 [ %hash, %entry ], [ %h2, %body2 ]
  %pos = and i64 %h, %mask
  %ptr = getelementptr %{NAME}.entry, %{NAME}.entry* %entries, i64 %pos
  %filledPtr = getelementptr %{NAME}.entry, %{NAME}.entry* %ptr, i64 0, i32 0
  %filled = load i1, i1* %filledPtr
  %filled32 = zext i1 %filled to i32
  br i1 %filled, label %body2, label %done

body2:
  %keyPtr = getelementptr %{NAME}.entry, %{NAME}.entry* %ptr, i64 0, i32 1
  %elemKey = load {KEY}, {KEY}* %keyPtr
  %eq = call i1 {KEY_PREFIX}.eq({KEY} %key, {KEY} %elemKey)
  %h2 = add i64 %h, 1
  br i1 %eq, label %done, label %body

done:
  ret %{NAME}.slot %ptr
}}

; Set the key and value at a given slot. The slot is assumed to have been
; returned by a lookup() on the same key provided here, and any old value for
; the key will be replaced. A new %{NAME} is returned reusing the same storage.
define %{NAME} @{NAME}.put(%{NAME} %dict, %{NAME}.slot %slot, {KEY} %key, {VALUE} %value) {{
start:
  %entries = extractvalue %{NAME} %dict, 0
  %size = extractvalue %{NAME} %dict, 1
  %capacity = extractvalue %{NAME} %dict, 2
  %filledPtr = getelementptr %{NAME}.entry, %{NAME}.entry* %slot, i64 0, i32 0
  %filled = load i1, i1* %filledPtr
  %keyPtr = getelementptr %{NAME}.entry, %{NAME}.entry* %slot, i64 0, i32 1
  %valuePtr = getelementptr %{NAME}.entry, %{NAME}.entry* %slot, i64 0, i32 2
  br i1 %filled, label %update, label %addNew

update:
  store i1 1, i1* %filledPtr
  store {KEY} %key, {KEY}* %keyPtr
  store {VALUE} %value, {VALUE}* %valuePtr
  ret %{NAME} %dict

addNew:
  ; Add the entry into the empty slot
  store i1 1, i1* %filledPtr
  store {KEY} %key, {KEY}* %keyPtr
  store {VALUE} %value, {VALUE}* %valuePtr
  %incSize = add i64 %size, 1
  ; Check whether table is at least 70% full; this means 10 * size >= 7 * capacity
  %v1 = mul i64 %size, 10
  %v2 = mul i64 %capacity, 7
  %full = icmp sge i64 %v1, %v2
  br i1 %full, label %onFull, label %returnCurrent

returnCurrent:
  %dict2 = insertvalue %{NAME} %dict, i64 %incSize, 1
  ret %{NAME} %dict2

onFull:
  %newCapacity = mul i64 %capacity, 2
  %newDict = call %{NAME} @{NAME}.new(i64 %newCapacity)
  ; Loop over old elements and insert them into newDict
  br label %body

body:
  %i = phi i64 [ 0, %onFull ], [ %i2, %body2 ], [ %i2, %moveEntry ]
  %newDict2 = phi %{NAME} [ %newDict, %onFull ], [ %newDict2, %body2 ], [ %newDict3, %moveEntry ]
  %comp = icmp eq i64 %i, %capacity
  br i1 %comp, label %done, label %body2

body2:
  %i2 = add i64 %i, 1
  %entryPtr = getelementptr %{NAME}.entry, %{NAME}.entry* %entries, i64 %i
  %entry = load %{NAME}.entry, %{NAME}.entry* %entryPtr
  %entryFilled = extractvalue %{NAME}.entry %entry, 0
  br i1 %entryFilled, label %moveEntry, label %body

moveEntry:
  %entryKey = extractvalue %{NAME}.entry %entry, 1
  %entryValue = extractvalue %{NAME}.entry %entry, 2
  %newSlot = call %{NAME}.slot @{NAME}.lookup(%{NAME} %newDict, {KEY} %entryKey)
  %newDict3 = call %{NAME} @{NAME}.put(%{NAME} %newDict2, %{NAME}.slot %newSlot, {KEY} %entryKey, {VALUE} %entryValue)
  br label %body

done:
  ret %{NAME} %newDict2
}}

; Get the entries of a dictionary as a vector.
define {KV_VEC} @{NAME}.tovec(%{NAME} %dict) {{
entry:
  %entries = extractvalue %{NAME} %dict, 0
  %size = extractvalue %{NAME} %dict, 1
  %capacity = extractvalue %{NAME} %dict, 2
  %vec = call {KV_VEC} {KV_VEC_PREFIX}.new(i64 %size)
  br label %body

body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body2 ], [ %i2, %body3 ]
  %j = phi i64 [ 0, %entry ], [ %j, %body2 ], [ %j2, %body3 ]
  %i2 = add i64 %i, 1
  %comp = icmp uge i64 %i, %capacity
  br i1 %comp, label %done, label %body2

body2:
  %entPtr = getelementptr %{NAME}.entry, %{NAME}.entry* %entries, i64 %i
  %ent = load %{NAME}.entry, %{NAME}.entry* %entPtr
  %filled = extractvalue %{NAME}.entry %ent, 0
  br i1 %filled, label %body3, label %body

body3:
  %elemPtr = call {KV_STRUCT}* {KV_VEC_PREFIX}.at({KV_VEC} %vec, i64 %j)
  %k = extractvalue %{NAME}.entry %ent, 1
  %v = extractvalue %{NAME}.entry %ent, 2
  %kv = insertvalue {KV_STRUCT} undef, {KEY} %k, 0
  %kv2 = insertvalue {KV_STRUCT} %kv, {VALUE} %v, 1
  store {KV_STRUCT} %kv2, {KV_STRUCT}* %elemPtr
  %j2 = add i64 %j, 1
  br label %body

done:
  ret {KV_VEC} %vec
}}
