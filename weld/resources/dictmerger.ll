; Template for a dictmerger and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - KEY: LLVM type of key (e.g. i32 or %MyStruct)
; - KEY_PREFIX
; - VALUE: LLVM type of value (e.g. i32 or %MyStruct)
; - KV_STRUCT: name of struct holding {{KEY, VALUE}} (should be generated outside)
;
; In addition, the function {NAME}.bld.merge_op({VALUE}, {VALUE}) is expected to be
; defined, implementing the operation needed to merge two values.

%{NAME}.bld = type %{NAME}

define void @{NAME}.bld.merge_op_on_pointers(i8* %metadata, i32 %isFilled, i8* %dst, i8* %value) {{
entry:
  %dstTypedPtr = bitcast i8* %dst to {VALUE}*
  %valueTypedPtr = bitcast i8* %value to {VALUE}*
  %dstTyped = load {VALUE}, {VALUE}* %dstTypedPtr
  %valueTyped = load {VALUE}, {VALUE}* %valueTypedPtr
  %isFilledBool = trunc i32 %isFilled to i1
  br i1 %isFilledBool, label %filled, label %unfilled
filled:
  %newValue = call {VALUE} @{NAME}.bld.merge_op({VALUE} %dstTyped, {VALUE} %valueTyped)
  store {VALUE} %newValue, {VALUE}* %dstTypedPtr
  br label %end
unfilled:
  store {VALUE} %valueTyped, {VALUE}* %dstTypedPtr
  br label %end
end:
  ret void
}}

; Initialize and return a new dictionary with the given initial capacity.
; The capacity must be a power of 2.
define %{NAME}.bld @{NAME}.bld.new(i64 %capacity, i64 %maxLocalBytes) {{
  %bld = call %{NAME} @{NAME}.new(i64 %capacity, i64 %maxLocalBytes,
    void (i8*, i32, i8*, i8*)* @{NAME}.bld.merge_op_on_pointers,
    void (i8*, i32, i8*, i8*)* @{NAME}.bld.merge_op_on_pointers, i8* null)
  ret %{NAME}.bld %bld
}}

; Append a value into a builder, growing its space if needed.
define %{NAME}.bld @{NAME}.bld.merge(%{NAME}.bld %bld, %{KV_STRUCT} %keyValue) {{
  %keyPtr = alloca {KEY}
  %valPtr = alloca {VALUE}
  %key = extractvalue %{KV_STRUCT} %keyValue, 0
  %value = extractvalue %{KV_STRUCT} %keyValue, 1
  store {KEY} %key, {KEY}* %keyPtr
  store {VALUE} %value, {VALUE}* %valPtr
  %rawHash = call i32 {KEY_PREFIX}.hash({KEY} %key)
  %finalizedHash = call i32 @hash_finalize(i32 %rawHash)
  %keyPtrRaw = bitcast {KEY}* %keyPtr to i8*
  %valPtrRaw = bitcast {VALUE}* %valPtr to i8*
  call void @weld_rt_dict_merge(i8* %bld, i32 %finalizedHash, i8* %keyPtrRaw, i8* %valPtrRaw)
  ret %{NAME}.bld %bld
}}

; Complete building a dictionary
define %{NAME} @{NAME}.bld.result(%{NAME}.bld %bld) {{
  call void @weld_rt_dict_finalize(i8* %bld)
  ret %{NAME} %bld
}}
