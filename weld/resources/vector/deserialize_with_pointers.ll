; Template for deserialization of a vector where the elements do not have a pointer.
;
; Parameters:
; - BUFNAME: name of vec[i8]
; - BUF_PREFIX
; - NAME: name of generated vector type, without % or @ prefix
; - ELEM
; - ELEM_PREFIX

define i64 @{NAME}.deserialize({BUFNAME} %buf, i64 %offset, %{NAME}* %resPtr) {{
  %sizePtrRaw = call i8* {BUF_PREFIX}.at({BUFNAME} %buf, i64 %offset)
  %sizePtr = bitcast i8* %sizePtrRaw to i64*
  %size = load i64, i64* %sizePtr
  %newVec = call %{NAME} @{NAME}.new(i64 %size)
  %dataOffset = add i64 %offset, 8
  br label %entry
entry:
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %nextOffset = phi i64 [ %dataOffset, %entry ], [ %dataOffset2, %body ]
  %dataPtrNewVec = call {ELEM}* @{NAME}.at(%{NAME} %newVec, i64 %i)
  %dataOffset2 = call i64 {ELEM_PREFIX}.deserialize({BUFNAME} %buf, i64 %nextOffset, {ELEM}* %dataPtrNewVec)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
done:
  %finalOffset = phi i64 [ %dataOffset, %entry ], [ %dataOffset2, %body ]
  store %{NAME} %newVec, %{NAME}* %resPtr
  ret i64 %finalOffset
}}
