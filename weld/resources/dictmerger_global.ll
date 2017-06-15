; Template for a dictmerger and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - KEY: LLVM type of key (e.g. i32 or %MyStruct)
; - VALUE: LLVM type of value (e.g. i32 or %MyStruct)
; - KV_STRUCT: name of struct holding {KEY, VALUE} (should be generated outside)
; - KV_VEC: name of vector of KV_STRUCTs (should be generated outside)
; - KV_VEC_PREFIX: prefix for helper functions of KV_VEC
; - OP: binary commutative merge operation (example: add or fadd)

%$NAME.bld = type %$NAME* ; the dictmerger is a pointer to the corresponding dictionary

; Initialize and return a new dictionary with the given initial capacity.
; The capacity must be a power of 2.
define %$NAME.bld @$NAME.bld.new(i64 %capacity) {
  %structSizePtr = getelementptr %$NAME, %$NAME* null, i32 1
  %structSize = ptrtoint %$NAME* %structSizePtr to i64
  %rawPtr = call i8* @new_merger(i64 %structSize, i32 1)
  %newDict = call %$NAME @$NAME.new(i64 %capacity)
  %bldPtr = bitcast i8* %rawPtr to %$NAME*
  store %$NAME %newDict, %$NAME* %bldPtr, align 1
  ret %$NAME.bld %bldPtr
}

; Append a value into a builder, growing its space if needed.
define %$NAME.bld @$NAME.bld.merge(%$NAME.bld %bldPtr, %$KV_STRUCT %keyValue, i32 %workerId) {
  ; TODO: Make the updates below atomic.
entry:
  %bld = load %$NAME, %$NAME* %bldPtr
  %key = extractvalue %$KV_STRUCT %keyValue, 0
  %value = extractvalue %$KV_STRUCT %keyValue, 1
  %slot = call %$NAME.slot @$NAME.lookup(%$NAME %bld, $KEY %key)
  call void @$NAME.slot.lock(%$NAME.slot %slot)
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
  call void @$NAME.slot.unlock(%$NAME.slot %slot)
  %res = phi %$NAME [ %res1, %onFilled ], [ %res2, %onEmpty ]
  store %$NAME %res, %$NAME* %bldPtr
  ret %$NAME.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %$NAME @$NAME.bld.result(%$NAME.bld %bldPtr) {
  %bld = load %$NAME, %$NAME* %bldPtr
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
