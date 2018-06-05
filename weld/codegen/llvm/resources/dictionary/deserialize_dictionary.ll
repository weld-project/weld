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
  %valPtr = alloca {VALUE}
  ; First, extract the number of key-value pairs in the dictionary.
  %sizePtrRaw = call i8* {BUF_PREFIX}.at({BUFNAME} %buf, i64 %offset)
  %sizePtr = bitcast i8* %sizePtrRaw to i64*
  %size = load i64, i64* %sizePtr
  %capacity = call i64 @i64.nextPower2(i64 %size) ; Requires next power of 2.
  %newDict = call %{NAME} @{NAME}.newFinalized(i64 %capacity)
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
  %valOffset = call i64 {KEY_PREFIX}.deserialize({BUFNAME} %buf, i64 %keyOffset, {KEY}* %keyPtr)
  %nextOffset = call i64 {VALUE_PREFIX}.deserialize({BUFNAME} %buf, i64 %valOffset, {VALUE}* %valPtr)
  %key = load {KEY}, {KEY}* %keyPtr
  %rawHash = call i32 {KEY_PREFIX}.hash({KEY} %key)
  %finalizedHash = call i32 @hash_finalize(i32 %rawHash)
  %keyPtrRaw = bitcast {KEY}* %keyPtr to i8*
  %valPtrRaw = bitcast {VALUE}* %valPtr to i8*
  call void @weld_rt_dict_merge(i8* %newDict, i32 %finalizedHash, i8* %keyPtrRaw, i8* %valPtrRaw)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
done:
  %finalOffset = phi i64 [ %dataOffset, %entry ], [ %nextOffset, %body ]
  store %{NAME} %newDict, %{NAME}* %resPtr
  ret i64 %finalOffset
}}
