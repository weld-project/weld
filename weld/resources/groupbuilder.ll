; Template for a groupbuilder and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - KEY: LLVM type of key (e.g. i32 or %MyStruct)
; - KEY_PREFIX
; - VALUE: LLVM type of value (e.g. i32 or %MyStruct)
; - KV_STRUCT: LLVM type of key,value struct

%{NAME}.gbld = type i8*

; Initialize and return a new groupbuilder with the given initial capacity.
; The capacity must be a power of 2.
define %{NAME}.gbld @{NAME}.gbld.new(i64 %capacity, i64 %maxLocalBytes) {{
  %keySizePtr = getelementptr {KEY}, {KEY}* null, i32 1
  %keySize = ptrtoint {KEY}* %keySizePtr to i32
  %valSizePtr = getelementptr {VALUE}, {VALUE}* null, i32 1
  %valSize = ptrtoint {VALUE}* %valSizePtr to i32
  %gbld = call i8* @weld_rt_gb_new(i32 %keySize, i32 (i8*, i8*)* {KEY_PREFIX}.eq_on_pointers,
    i32 %valSize, i64 %maxLocalBytes, i64 %capacity)
  ret %{NAME}.gbld %gbld
}}

; Append a value into a groupbuilder, growing its space if needed.
define %{NAME}.gbld @{NAME}.gbld.merge(%{NAME}.gbld %gbld, %{KV_STRUCT} %keyValue) {{
  %keyPtr = alloca {KEY}
  %valuePtr = alloca {VALUE}
  %key = extractvalue %{KV_STRUCT} %keyValue, 0
  %value = extractvalue %{KV_STRUCT} %keyValue, 1
  store {KEY} %key, {KEY}* %keyPtr
  store {VALUE} %value, {VALUE}* %valuePtr
  %keyPtrRaw = bitcast {KEY}* %keyPtr to i8*
  %valuePtrRaw = bitcast {VALUE}* %valuePtr to i8*
  %rawHash = call i32 {KEY_PREFIX}.hash({KEY} %key)
  %finalizedHash = call i32 @hash_finalize(i32 %rawHash)
  call void @weld_rt_gb_merge(%{NAME}.gbld %gbld, i8* %keyPtrRaw, i32 %finalizedHash, i8* %valuePtrRaw)
  ret %{NAME}.gbld %gbld
}}

; Complete building a groupbuilder
define %{NAME} @{NAME}.gbld.result(%{NAME}.gbld %gbld) {{
  %result = call i8* @weld_rt_gb_result(i8* %gbld)
  ret %{NAME} %result
}}