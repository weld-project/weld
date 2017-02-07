; Template for a dictmerger and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - KEY: LLVM type of key (e.g. i32 or %MyStruct)
; - VALUE: LLVM type of value (e.g. i32 or %MyStruct)
; - KV_STRUCT: name of struct holding {KEY, VALUE} (should be generated outside)
; - OP: binary commutative merge operation (example: add or fadd)

%$NAME.bld = type %$NAME* ; the dictmerger is a pointer to the corresponding dictionary

; Initialize and return a new dictionary with the given initial capacity.
; The capacity must be a power of 2.
define %$NAME.bld @$NAME.bld.new(i64 %capacity) {
  %entrySizePtr = getelementptr %$NAME.entry* null, i32 1
  %entrySize = ptrtoint %$NAME.entry* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %capacity
  %runId = call i64 @get_runid()
  %bytes = call i8* @weld_rt_malloc(i64 %runId, i64 %allocSize)
  ; Memset all the bytes to 0 to set the isFilled fields to 0
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  %entries = bitcast i8* %bytes to %$NAME.entry*
  %1 = insertvalue %$NAME undef, %$NAME.entry* %entries, 0
  %2 = insertvalue %$NAME %1, i64 0, 1
  %3 = insertvalue %$NAME %2, i64 %capacity, 2
  %bldSizePtr = getelementptr %$NAME.bld null, i32 1
  %bldSize = ptrtoint %$NAME.bld %bldSizePtr to i64
  %4 = call i8* @weld_rt_malloc(i64 %runId, i64 %bldSize)
  %5 = bitcast i8* %4 to %$NAME*
  store %$NAME %3, %$NAME.bld %5
  ret %$NAME.bld %5
}

; Append a value into a builder, growing its space if needed.
define %$NAME.bld @$NAME.bld.merge(%$NAME.bld %bldPtr, %$KV_STRUCT %keyValue) {
entry:
  %bld = load %$NAME.bld %bldPtr
  %key = extractvalue %$KV_STRUCT %keyValue, 0
  %value = extractvalue %$KV_STRUCT %keyValue, 1
  %slot = call %$NAME.slot @$NAME.lookup(%$NAME %bld, $KEY %key)
  %filled = call i1 @$NAME.slot.filled(%$NAME.slot %slot)
  br i1 %filled, label %onFilled, label %onEmpty

onFilled:
  %oldValue = call $VALUE @$NAME.slot.value(%$NAME.slot %slot)
  %newValue = $OP $VALUE %oldValue, %value  ; TODO: Fix this when making Op more generic
  %res1 = call %$NAME @$NAME.put(%$NAME %bld, %$NAME.slot %slot, $KEY %key, $VALUE %newValue)
  br label %done

onEmpty:
  %res2 = call %$NAME @$NAME.put(%$NAME %bld, %$NAME.slot %slot, $KEY %key, $VALUE %value)
  br label %done

done:
  %res = phi %$NAME [ %res1, %onFilled ], [ %res2, %onEmpty ]
  store %$NAME %res, %$NAME.bld %bldPtr
  ret %$NAME.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %$NAME @$NAME.bld.result(%$NAME.bld %bldPtr) {
  ; TODO: Fix this
  %bld = load %$NAME.bld %bldPtr
  %toFree = bitcast %$NAME.bld %bldPtr to i8*
  %runId = call i64 @get_runid()
  call void @weld_rt_free(i64 %runId, i8* %toFree)
  ret %$NAME %bld
}

; Dummy hash function; this is needed for structs that use these dictmergers as fields.
define i64 @$NAME.bld.hash(%$NAME.bld %bld) {
  ret i64 0
}

; Dummy comparison function; this is needed for structs that use these dictmergers as fields.
define i32 @$NAME.bld.cmp(%$NAME.bld %bld1, %$NAME.bld %bld2) {
  ret i32 -1
}
