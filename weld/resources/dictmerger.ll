; Template for a dictmerger and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - KEY: LLVM type of key (e.g. i32 or %MyStruct)
; - VALUE: LLVM type of value (e.g. i32 or %MyStruct)
; - KV_STRUCT: name of struct holding {{KEY, VALUE}} (should be generated outside)
;
; In addition, the function {NAME}.bld.merge_op({VALUE}, {VALUE}) is expected to be
; defined, implementing the operation needed to merge two values.

%{NAME}.bld = type %{NAME}

; Initialize and return a new dictionary with the given initial capacity.
; The capacity must be a power of 2.
define %{NAME}.bld @{NAME}.bld.new(i64 %capacity) {{
  %bld = call %{NAME} @{NAME}.new(i64 %capacity)
  ret %{NAME}.bld %bld
}}

; Append a value into a builder, growing its space if needed.
define %{NAME}.bld @{NAME}.bld.merge(%{NAME}.bld %bld, %{KV_STRUCT} %keyValue) {{
entry:
  %key = extractvalue %{KV_STRUCT} %keyValue, 0
  %value = extractvalue %{KV_STRUCT} %keyValue, 1
  %slot = call %{NAME}.slot @{NAME}.lookup(%{NAME} %bld, {KEY} %key)
  %filled = call i1 @{NAME}.slot.filled(%{NAME}.slot %slot)
  br i1 %filled, label %onFilled, label %onEmpty

onFilled:
  %oldValue = call {VALUE} @{NAME}.slot.value(%{NAME}.slot %slot)
  %newValue = call {VALUE} @{NAME}.bld.merge_op({VALUE} %oldValue, {VALUE} %value)
  call %{NAME} @{NAME}.put(%{NAME} %bld, %{NAME}.slot %slot, {KEY} %key, {VALUE} %newValue)
  br label %done

onEmpty:
  call %{NAME} @{NAME}.put(%{NAME} %bld, %{NAME}.slot %slot, {KEY} %key, {VALUE} %value)
  br label %done

done:
  ret %{NAME}.bld %bld
}}

; Complete building a dictionary
define %{NAME} @{NAME}.bld.result(%{NAME}.bld %bld) {{
start:
  br label %entry
entry:
  %nextSlotRaw = call i8* @weld_rt_dict_finalize_next_local_slot(i8* %bld)
  %nextSlotLong = ptrtoint i8* %nextSlotRaw to i64
  %isNull = icmp eq i64 %nextSlotLong, 0
  br i1 %isNull, label %done, label %body
body:
  %nextSlot = bitcast i8* %nextSlotRaw to %{NAME}.slot
  %key = call {KEY} @{NAME}.slot.key(%{NAME}.slot %nextSlot)
  %localValue = call {VALUE} @{NAME}.slot.value(%{NAME}.slot %nextSlot)
  %globalSlotRaw = call i8* @weld_rt_dict_finalize_global_slot_for_local(i8* %bld, i8* %nextSlotRaw)
  %globalSlot = bitcast i8* %globalSlotRaw to %{NAME}.slot
  %filled = call i1 @{NAME}.slot.filled(%{NAME}.slot %globalSlot)
  br i1 %filled, label %onFilled, label %onEmpty
onFilled:
  %globalValue = call {VALUE} @{NAME}.slot.value(%{NAME}.slot %globalSlot)
  %newValue = call {VALUE} @{NAME}.bld.merge_op({VALUE} %localValue, {VALUE} %globalValue)
  call %{NAME} @{NAME}.put(%{NAME} %bld, %{NAME}.slot %globalSlot, {KEY} %key, {VALUE} %newValue)
  br label %entry
onEmpty:
  call %{NAME} @{NAME}.put(%{NAME} %bld, %{NAME}.slot %globalSlot, {KEY} %key, {VALUE} %localValue)
  br label %entry
done:
  ret %{NAME} %bld
}}
