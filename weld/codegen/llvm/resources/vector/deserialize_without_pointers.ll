; Template for deserialization of a vector where the elements do not have a pointer.
;
; Parameters:
; - BUFNAME: name of vec[i8]
; - BUF_PREFIX
; - NAME: name of generated vector type, without % or @ prefix
; - ELEM

define i64 @{NAME}.deserialize({BUFNAME} %buf, i64 %offset, %{NAME}* %resPtr) {{
  %elSizeTmp = call i32 @{NAME}.elSize()
  %elSize = zext i32 %elSizeTmp to i64
  %sizePtrRaw = call i8* {BUF_PREFIX}.at({BUFNAME} %buf, i64 %offset)
  %sizePtr = bitcast i8* %sizePtrRaw to i64*
  %size = load i64, i64* %sizePtr
  
  ; Pointer to the data = sizePtrRaw + sizeof(i64).
  %dataPtrRaw = getelementptr i8, i8* %sizePtrRaw, i64 8
  %newVec = call %{NAME} @{NAME}.new(i64 %size)
  %dataPtrNewVec = call {ELEM}* @{NAME}.at(%{NAME} %newVec, i64 0)
  %dataPtrNewVecRaw = bitcast {ELEM}* %dataPtrNewVec to i8*
  %copySize = mul i64 %elSize, %size
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dataPtrNewVecRaw, i8* %dataPtrRaw, i64 %copySize, i32 8, i1 0)
  store %{NAME} %newVec, %{NAME}* %resPtr
  
  ; Compute and return the new offset in the serialize buffer.
  %1 = add i64 %offset, 8
  %2 = add i64 %1, %copySize 
  ret i64 %2
}}
