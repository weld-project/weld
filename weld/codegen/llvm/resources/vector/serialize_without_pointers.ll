; Template for serialization of a vector where the elements have a pointer.
;
; Parameters:
; - BUFNAME: name of vec[i8] type that this vector serializes into.
; - NAME: name of generated vector type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)

define %{BUFNAME}.growable @{NAME}.serialize(%{BUFNAME}.growable %buf, %{NAME} %vec) {{

  %elSizeTmp = call i32 @{NAME}.elSize()
  %elSize = zext i32 %elSizeTmp to i64
  %vecSize = call i64 @{NAME}.size(%{NAME} %vec)
  %dataSize = mul i64 %elSize, %vecSize

  ; Add 8 bytes since we also store the length.
  %totalSize = add i64 %dataSize, 8
  %buf2 = call %{BUFNAME}.growable @{BUFNAME}.growable.resize_to_fit(%{BUFNAME}.growable %buf, i64 %totalSize)
  %ptr = call i8* @{BUFNAME}.growable.last(%{BUFNAME}.growable %buf2)

  ; Store the vector length.
  %lenPtr = bitcast i8* %ptr to i64*
  store i64 %vecSize, i64* %lenPtr

  ; Increment the pointer and store the vector data.
  %dataStorePtr = getelementptr i8, i8* %ptr, i64 8
  %vecPtr = call {ELEM}* @{NAME}.at(%{NAME} %vec, i64 0)
  %vecPtrRaw = bitcast {ELEM}* %vecPtr to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dataStorePtr, i8* %vecPtrRaw, i64 %dataSize, i32 8, i1 0)
  %result = call %{BUFNAME}.growable @{BUFNAME}.growable.extend(%{BUFNAME}.growable %buf2, i64 %totalSize)
  ret %{BUFNAME}.growable %result
}}
