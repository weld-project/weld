; Sorts a copy of the vector and returns it.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)

define %{NAME} @{NAME}.sort(%{NAME} %vec) {{
  %res = call %{NAME} @{NAME}.clone(%{NAME} %vec)
  %elements = extractvalue %{NAME} %res, 0
  %size = extractvalue %{NAME} %res, 1
  %elementsRaw = bitcast {ELEM}* %elements to i8*
  %elemSizePtr = getelementptr {ELEM}, {ELEM}* null, i32 1
  %elemSize = ptrtoint {ELEM}* %elemSizePtr to i64
  call void @qsort(i8* %elementsRaw, i64 %size, i64 %elemSize, i32 (i8*, i8*)* @{NAME}.helper)
  ret %{NAME} %res
}}

define i32 @{NAME}.helper(i8* %p1, i8* %p2) {{
  %kv1 = bitcast i8* %p1 to {ELEM}*
  %kv2 = bitcast i8* %p2 to {ELEM}*
  %kPtr1 = getelementptr {ELEM}, {ELEM}* %kv1, i64 0
  %kPtr2 = getelementptr {ELEM}, {ELEM}* %kv2, i64 0
  %k1 = load {ELEM}, {ELEM}* %kPtr1
  %k2 = load {ELEM}, {ELEM}* %kPtr2
  %res = call i32 @{ELEM}.cmp({ELEM} %k1, {ELEM} %k2)
  ret i32 %res
}}