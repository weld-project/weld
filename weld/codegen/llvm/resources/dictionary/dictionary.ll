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

; hash, isFilled, lockVar, padding, key, value 
; (packed so that C code can easily store it in a byte array without considering padding)
%{NAME}.entry = type <{{ i32, i8, i8, i16, {KEY}, {VALUE} }}>
%{NAME}.slot = type %{NAME}.entry*                ; handle to an entry in the API
%{NAME} = type i8*  ; entries, size, capacity

; Initialize and return a new dictionary with the given initial capacity.
; The capacity must be a power of 2.
define %{NAME} @{NAME}.new(i64 %capacity, i64 %maxLocalBytes, void (i8*, i32, i8*, i8*)* %mergeNewVal,
  void (i8*, i32, i8*, i8*)* %mergeValsFinalize, i8* %metadata) {{
  %keySizePtr = getelementptr {KEY}, {KEY}* null, i32 1
  %keySize = ptrtoint {KEY}* %keySizePtr to i32
  %valSizePtr = getelementptr {VALUE}, {VALUE}* null, i32 1
  %valSize = ptrtoint {VALUE}* %valSizePtr to i32
  %dict = call i8* @weld_rt_dict_new(i32 %keySize, i32 (i8*, i8*)* {KEY_PREFIX}.eq_on_pointers,
    void (i8*, i32, i8*, i8*)* %mergeNewVal, void (i8*, i32, i8*, i8*)* %mergeValsFinalize,
    i8* %metadata, i32 %valSize, i32 %valSize, i64 %maxLocalBytes, i64 %capacity)
  ret %{NAME} %dict
}}

; Initialize a new finalized dictionary with the given capacity.
; The capacity must be a power of 2.
define %{NAME} @{NAME}.newFinalized(i64 %capacity) {{
  %keySizePtr = getelementptr {KEY}, {KEY}* null, i32 1
  %keySize = ptrtoint {KEY}* %keySizePtr to i32
  %valSizePtr = getelementptr {VALUE}, {VALUE}* null, i32 1
  %valSize = ptrtoint {VALUE}* %valSizePtr to i32
  %dict = call i8* @weld_rt_dict_new_finalized(i32 %keySize, i32 (i8*, i8*)* {KEY_PREFIX}.eq_on_pointers,
    void (i8*, i32, i8*, i8*)* null, void (i8*, i32, i8*, i8*)* null,
    i8* null, i32 %valSize, i32 %valSize, i64 0, i64 %capacity)
  ret %{NAME} %dict
}}

; Free dictionary
define void @{NAME}.free(%{NAME} %dict) {{
  call void @weld_rt_dict_free(i8* %dict)
  ret void
}}

; Get the size of a dictionary.
define i64 @{NAME}.size(%{NAME} %dict) {{
  %size = call i64 @weld_rt_dict_size(i8* %dict)
  ret i64 %size
}}

; Check whether a slot is filled.
define i1 @{NAME}.slot.filled(%{NAME}.slot %slot) {{
  %filledPtr = getelementptr %{NAME}.entry, %{NAME}.slot %slot, i64 0, i32 1
  %filled_i8 = load i8, i8* %filledPtr
  %filled = trunc i8 %filled_i8 to i1
  ret i1 %filled
}}

; Get a pointer to the hash in the slot.
define i32* @{NAME}.slot.hashPtr(%{NAME}.slot %slot) alwaysinline {{
  %hashPtr = getelementptr %{NAME}.entry, %{NAME}.slot %slot, i64 0, i32 0
  ret i32* %hashPtr
}}

; Get a pointer to the filled flag in the slot
define i8* @{NAME}.slot.filledPtr(%{NAME}.slot %slot) alwaysinline {{
  %filledPtr = getelementptr %{NAME}.entry, %{NAME}.slot %slot, i64 0, i32 1
  ret i8* %filledPtr
}}

; Get a pointer to the key in the slot.
define {KEY}* @{NAME}.slot.keyPtr(%{NAME}.slot %slot) alwaysinline {{
  %keyPtr = getelementptr %{NAME}.entry, %{NAME}.slot %slot, i64 0, i32 4
  ret {KEY}* %keyPtr
}}

; Get a pointer to the value in the slot.
define {VALUE}* @{NAME}.slot.valuePtr(%{NAME}.slot %slot) alwaysinline {{
  %valuePtr = getelementptr %{NAME}.entry, %{NAME}.slot %slot, i64 0, i32 5
  ret {VALUE}* %valuePtr
}}

; Get the key for a slot (only valid if filled).
define {KEY} @{NAME}.slot.key(%{NAME}.slot %slot) {{
  %keyPtr = call {KEY}* @{NAME}.slot.keyPtr(%{NAME}.slot %slot)
  %key = load {KEY}, {KEY}* %keyPtr
  ret {KEY} %key
}}

; Get the value for a slot (only valid if filled).
define {VALUE} @{NAME}.slot.value(%{NAME}.slot %slot) {{
  %valuePtr = call {VALUE}* @{NAME}.slot.valuePtr(%{NAME}.slot %slot)
  %value = load {VALUE}, {VALUE}* %valuePtr
  ret {VALUE} %value
}}

; Look up the given key, returning a slot for it. The slot functions may be
; used to tell whether the entry is filled, get its value, etc. This function
; gives *read only* access to the slot.
define %{NAME}.slot @{NAME}.lookup(%{NAME} %dict, {KEY} %key) {{
  %keyPtr = alloca {KEY}
  store {KEY} %key, {KEY}* %keyPtr
  %rawHash = call i32 {KEY_PREFIX}.hash({KEY} %key)
  %finalizedHash = call i32 @hash_finalize(i32 %rawHash)
  %keyPtrRaw = bitcast {KEY}* %keyPtr to i8*
  %slotRaw = call i8* @weld_rt_dict_lookup(i8* %dict, i32 %finalizedHash, i8* %keyPtrRaw)
  %slot = bitcast i8* %slotRaw to %{NAME}.slot
  ret %{NAME}.slot %slot
}}

; Get the entries of a dictionary as a vector.
define {KV_VEC} @{NAME}.tovec(%{NAME} %dict) {{
  %valOffsetPtr = getelementptr {KV_STRUCT}, {KV_STRUCT}* null, i32 0, i32 1
  %valOffset = ptrtoint {VALUE}* %valOffsetPtr to i32
  %structSizePtr = getelementptr {KV_STRUCT}, {KV_STRUCT}* null, i32 1
  %structSize = ptrtoint {KV_STRUCT}* %structSizePtr to i32
  %arrRaw = call i8* @weld_rt_dict_to_array(i8* %dict, i32 %valOffset, i32 %structSize)
  %arr = bitcast i8* %arrRaw to {KV_STRUCT}*
  %size = call i64 @weld_rt_dict_size(i8* %dict)
  %1 = insertvalue {KV_VEC} undef, {KV_STRUCT}* %arr, 0
  %2 = insertvalue {KV_VEC} %1, i64 %size, 1
  ret {KV_VEC} %2
}}
