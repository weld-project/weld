; Sorts a copy of the vector and returns it.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - FUNC: the key function (must return scalar)
; - KEY: type returned by FUNC (must be scalar)
define %{NAME} @{NAME}.{FUNC}.sort(%{NAME} %vec) {{
  %res = call %{NAME} @{NAME}.clone(%{NAME} %vec)
  %elements = extractvalue %{NAME} %res, 0
  %size = extractvalue %{NAME} %res, 1
  %elementsRaw = bitcast {ELEM}* %elements to i8*
  %elemSizePtr = getelementptr {ELEM}, {ELEM}* null, i32 1
  %elemSize = ptrtoint {ELEM}* %elemSizePtr to i64
  call void @qsort(i8* %elementsRaw, i64 %size, i64 %elemSize, i32 (i8*, i8*)* @{NAME}.{FUNC}.helper)
  ret %{NAME} %res
}}

define i32 @{NAME}.{FUNC}.helper(i8* %p1, i8* %p2) {{
  %kv1 = bitcast i8* %p1 to {ELEM}*
  %kv2 = bitcast i8* %p2 to {ELEM}*
  %ev1 = call {KEY} @{FUNC}({ELEM}* %kv1)
  %ev2 = call {KEY} @{FUNC}({ELEM}* %kv2)
  %res = call i32 @{KEY}.cmp({KEY} %ev1, {KEY} %ev2)
  ret i32 %res
}}
