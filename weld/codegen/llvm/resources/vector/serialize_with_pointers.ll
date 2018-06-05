; Template for serialization of a vector where the elements have a pointer.
;
; Parameters:
; - BUFNAME: name of vec[i8] type that this vector serializes into.
; - NAME: name of generated vector type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_SERIALIZE: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)


define %{BUFNAME}.growable @{NAME}.serialize(%{BUFNAME}.growable %buf, %{NAME} %vec) {{
  %elements = extractvalue %{NAME} %vec, 0 
  %size = extractvalue %{NAME} %vec, 1 

  %buf2 = call %{BUFNAME}.growable @{BUFNAME}.growable.resize_to_fit(%{BUFNAME}.growable %buf, i64 8)
  %ptr = call i8* @{BUFNAME}.growable.last(%{BUFNAME}.growable %buf2)
  %lenPtr = bitcast i8* %ptr to i64*
  store i64 %size, i64* %lenPtr
  %buf3 = call %{BUFNAME}.growable @{BUFNAME}.growable.extend(%{BUFNAME}.growable %buf2, i64 8)
  br label %entry

entry:
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %buf4 = phi %{BUFNAME}.growable [ %buf3, %entry ], [ %buf5, %body ]
  %elemPtr = call {ELEM}* @{NAME}.at(%{NAME} %vec, i64 %i)
  %elem = load {ELEM}, {ELEM}* %elemPtr
  %buf5 = call %{BUFNAME}.growable {ELEM_SERIALIZE}(%{BUFNAME}.growable %buf4, {ELEM} %elem)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
done:
  %buf6 = phi %{BUFNAME}.growable [ %buf3, %entry ], [ %buf5, %body ]
  ret %{BUFNAME}.growable %buf6
}}
