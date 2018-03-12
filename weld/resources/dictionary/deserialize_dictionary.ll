; Template for a dictionary and its helper functions. Uses linear probing.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - KEY: LLVM type of key (e.g. i32 or %MyStruct)
; - VALUE: LLVM type of value (e.g. i32 or %MyStruct)
; - KEY_PREFIX: prefix for helper functions of key (e.g. @i32 or @MyStruct)
; - VALUE_PREFIX: prefix for helper functions of value (e.g. @i32 or @MyStruct)
; - BUFNAME
; - BUF_PREFIX

define i64 @{NAME}.deserialize({BUFNAME} %buf, i64 %offset, %{NAME}* %resPtr) {{
  %keyPtr = alloca {KEY}
  ; First, extract the number of key-value pairs in the dictionary.
  %sizePtrRaw = call i8* {BUF_PREFIX}.at({BUFNAME} %buf, i64 %offset)
  %sizePtr = bitcast i8* %sizePtrRaw to i64*
  %size = load i64, i64* %sizePtr
  %capacity = call i64 @i64.nextPower2(i64 %size) ; Requires next power of 2.
  %newDict = call %{NAME} @{NAME}.newFinalized(i64 %capacity, i64 %capacity)
  ; Increment offset to point to the data.
  %dataOffset = add i64 %offset, 8
  ; Main loop - deserialize each key/value pair.
  br label %entry
entry:
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %keyOffset = phi i64 [ %dataOffset, %entry ], [ %nextOffset, %body ]
  %valOffset = call {KEY_PREFIX}.deserialize({BUFNAME} %buf, i64 %keyOffset, {KEY}* %keyPtr)
  ; Lookup the slot - this is similar to lookup, but we need the hash so we compute it directly here.
  %key = load {KEY}, {KEY}* %keyPtr
  %rawHash = call i32 {KEY_PREFIX}.hash({KEY} %key)
  %finalizedHash = call i32 @hash_finalize(i32 %rawHash)
  %keyPtrRaw = bitcast {KEY}* %keyPtr to i8*
  %slotRaw = call i8* @weld_rt_dict_lookup(i8* %dict, i32 %finalizedHash, i8* %keyPtrRaw)
  %slot = bitcast i8* %slotRaw to %{NAME}.slot

  ; Fill the slot. For the value, just deserialize directly into the slot pointer.
  %slotFilledPtr = call i8* @{NAME}.slot.filledPtr(%{NAME}.slot %slot)
  store i8 1, i8* %slotFilledPtr
  %slotHashPtr = call i32* @{NAME}.slot.hashPtr(%{NAME}.slot %slot)
  store i32 %hash, i32* %slotHashPtr
  %slotKeyPtr = call {KEY}* @{NAME}.slot.keyPtr(%{KEY}.slot %slot)
  store {KEY} %key, {KEY}* %slotKeyPtr
  %slotValPtr = call {VALUE}* @{NAME}.slot.valuePtr(%{NAME}.slot %slot)
  %nextOffset = call {VALUE_PREFIX}.deserialize({BUFNAME} %buf, i64 %valOffset, {VALUE}* %slotValPtr)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
done:
  %finalOffset = phi i64 [ %dataOffset, %entry ], [ %nextOffset, %body ]
  ret i64 %finalOffset
}}
