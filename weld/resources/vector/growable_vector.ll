; Template for a growable single-threaded vector. This template
; assumes that a vector of the given type has already been initialized.

; Parameters:
; - NAME: name to give generated vector type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)

%{NAME}.growable = type {{ %{NAME}, i64 }}   ; vector (data, capacity), size

; Initialize and return a new growable vector with the given capacity and size 0.
define %{NAME}.growable @{NAME}.growable.new(i64 %capacity) {{
  %vector = call %{NAME} @{NAME}.new(i64 %capacity)
  %1 = insertvalue %{NAME}.growable undef, %{NAME} %vector, 0
  %2 = insertvalue %{NAME}.growable %1, i64 0, 1
  ret %{NAME}.growable %2
}}

; Converts the growable vector to a regular vector.
define %{NAME} @{NAME}.growable.tovec(%{NAME}.growable %gvec) alwaysinline {{
  %vec = extractvalue %{NAME}.growable %gvec, 0
  %size = extractvalue %{NAME}.growable %gvec, 1
  %vec2 = insertvalue %{NAME} %vec, i64 %size, 1
  ret %{NAME} %vec2
}}

; Get a pointer to the index'th element.
define {ELEM}* @{NAME}.growable.at(%{NAME}.growable %gvec, i64 %index) alwaysinline {{
  %vec = extractvalue %{NAME}.growable %gvec, 0
  %elements = extractvalue %{NAME} %vec, 0
  %ptr = getelementptr {ELEM}, {ELEM}* %elements, i64 %index
  ret {ELEM}* %ptr
}}

; Get a pointer to the first unfilled slot.
define {ELEM}* @{NAME}.growable.last(%{NAME}.growable %gvec) alwaysinline {{
  %vec = extractvalue %{NAME}.growable %gvec, 0
  %size = extractvalue %{NAME}.growable %gvec, 1
  %elements = extractvalue %{NAME} %vec, 0
  %ptr = getelementptr {ELEM}, {ELEM}* %elements, i64 %size
  ret {ELEM}* %ptr
}}

; Increase the size by `sz`.
define %{NAME}.growable @{NAME}.growable.extend(%{NAME}.growable %gvec, i64 %sz) alwaysinline {{
  %prev = extractvalue %{NAME}.growable %gvec, 1 
  %newSize = add i64 %prev, %sz
  %ret = insertvalue %{NAME}.growable %gvec, i64 %newSize, 1 
  ret %{NAME}.growable %ret
}}

; Resize the vector to fit `num_elements` elements. Does nothing if the vector already has enough capacity.
define %{NAME}.growable @{NAME}.growable.resize_to_fit(%{NAME}.growable %gvec, i64 %num_elements) {{
  %vec = extractvalue %{NAME}.growable %gvec, 0
  %size = extractvalue %{NAME}.growable %gvec, 1
  %capacity = extractvalue %{NAME} %vec, 1
  %space = sub i64 %capacity, %size
  br label %entry

entry:
  %cond = icmp ult i64 %space, %num_elements
  br i1 %cond, label %resize, label %done

resize:
  %newSize = add i64 %num_elements, %capacity
  %doubled = mul i64 %capacity, 2
  %cond2 = icmp uge i64 %doubled, %newSize
  ; New capacity is either capacity + elements we want to add
  ; or double the capacity: whichever is bigger.
  %newCapacity = select i1 %cond2, i64 %doubled, i64 %newSize
  %elemSizePtr = getelementptr {ELEM}, {ELEM}* null, i32 1
  %elemSize = ptrtoint {ELEM}* %elemSizePtr to i64
  %bytes = extractvalue %{NAME} %vec, 0
  %bytesRaw = bitcast {ELEM}* %bytes to i8*
  %allocSize = mul i64 %elemSize, %newCapacity
  %runId = call i64 @weld_rt_get_run_id()
  %newBytes = call i8* @weld_run_realloc(i64 %runId, i8* %bytesRaw, i64 %allocSize)
  %newBuf = bitcast i8* %newBytes to {ELEM}*

  %1 = insertvalue %{NAME} undef, {ELEM}* %newBuf, 0
  %2 = insertvalue %{NAME} %1, i64 %newCapacity, 1
  %3 = insertvalue %{NAME}.growable undef, %{NAME} %2, 0
  %4 = insertvalue %{NAME}.growable %3, i64 %size, 1
  br label %done

done:
  %ret = phi %{NAME}.growable [ %gvec, %entry ], [ %4, %resize]
  ret %{NAME}.growable %ret
}}
