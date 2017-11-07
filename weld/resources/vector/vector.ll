; Template for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)
; - VECSIZE: Size of vectors.

%{NAME} = type {{ {ELEM}*, i64 }}           ; elements, size
%{NAME}.bld = type i8*

; VecMerger
%{NAME}.vm.bld = type %{NAME}*

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define %{NAME}.vm.bld @{NAME}.vm.bld.getPtrIndexed(%{NAME}.vm.bld %bldPtr, i32 %i) alwaysinline {{
  %mergerPtr = getelementptr %{NAME}, %{NAME}* null, i32 1
  %mergerSize = ptrtoint %{NAME}* %mergerPtr to i64
  %asPtr = bitcast %{NAME}.vm.bld %bldPtr to i8*
  %rawPtr = call i8* @weld_rt_get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %{NAME}.vm.bld
  ret %{NAME}.vm.bld %ptr
}}

; Initialize and return a new vecmerger with the given initial vector.
define %{NAME}.vm.bld @{NAME}.vm.bld.new(%{NAME} %vec) {{
  %nworkers = call i32 @weld_rt_get_nworkers()
  %structSizePtr = getelementptr %{NAME}, %{NAME}* null, i32 1
  %structSize = ptrtoint %{NAME}* %structSizePtr to i64

  %bldPtr = call i8* @weld_rt_new_merger(i64 %structSize, i32 %nworkers)
  %typedPtr = bitcast i8* %bldPtr to %{NAME}.vm.bld

  ; Copy the initial value into the first vector
  %first = call %{NAME}.vm.bld @{NAME}.vm.bld.getPtrIndexed(%{NAME}.vm.bld %typedPtr, i32 0)
  %cloned = call %{NAME} @{NAME}.clone(%{NAME} %vec)
  %capacity = call i64 @{NAME}.size(%{NAME} %vec)
  store %{NAME} %cloned, %{NAME}.vm.bld %first
  br label %entry

entry:
  %cond = icmp ult i32 1, %nworkers
  br i1 %cond, label %body, label %done

body:
  %i = phi i32 [ 1, %entry ], [ %i2, %body ]
  %vecPtr = call %{NAME}* @{NAME}.vm.bld.getPtrIndexed(%{NAME}.vm.bld %typedPtr, i32 %i)
  %newVec = call %{NAME} @{NAME}.new(i64 %capacity)
  call void @{NAME}.zero(%{NAME} %newVec)
  store %{NAME} %newVec, %{NAME}* %vecPtr
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done

done:
  ret %{NAME}.vm.bld %typedPtr
}}

; Returns a pointer to the value an element should be merged into.
; The caller should perform the merge operation on the contents of this pointer
; and then store the resulting value back.
define i8* @{NAME}.vm.bld.merge_ptr(%{NAME}.vm.bld %bldPtr, i64 %index, i32 %workerId) {{
  %bldPtrLocal = call %{NAME}* @{NAME}.vm.bld.getPtrIndexed(%{NAME}.vm.bld %bldPtr, i32 %workerId)
  %vec = load %{NAME}, %{NAME}* %bldPtrLocal
  %elem = call {ELEM}* @{NAME}.at(%{NAME} %vec, i64 %index)
  %elemPtrRaw = bitcast {ELEM}* %elem to i8*
  ret i8* %elemPtrRaw
}}

; Initialize and return a new vector with the given size.
define %{NAME} @{NAME}.new(i64 %size) {{
  %elemSizePtr = getelementptr {ELEM}, {ELEM}* null, i32 1
  %elemSize = ptrtoint {ELEM}* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %runId = call i64 @weld_rt_get_run_id()
  %bytes = call i8* @weld_run_malloc(i64 %runId, i64 %allocSize)
  %elements = bitcast i8* %bytes to {ELEM}*
  %1 = insertvalue %{NAME} undef, {ELEM}* %elements, 0
  %2 = insertvalue %{NAME} %1, i64 %size, 1
  ret %{NAME} %2
}}

; Zeroes a vector's underlying buffer.
define void @{NAME}.zero(%{NAME} %v) {{
  %elements = extractvalue %{NAME} %v, 0
  %size = extractvalue %{NAME} %v, 1
  %bytes = bitcast {ELEM}* %elements to i8*

  %elemSizePtr = getelementptr {ELEM}, {ELEM}* null, i32 1
  %elemSize = ptrtoint {ELEM}* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  ret void
}}

; Clone a vector.
define %{NAME} @{NAME}.clone(%{NAME} %vec) {{
  %elements = extractvalue %{NAME} %vec, 0
  %size = extractvalue %{NAME} %vec, 1
  %entrySizePtr = getelementptr {ELEM}, {ELEM}* null, i32 1
  %entrySize = ptrtoint {ELEM}* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %size
  %bytes = bitcast {ELEM}* %elements to i8*
  %vec2 = call %{NAME} @{NAME}.new(i64 %size)
  %elements2 = extractvalue %{NAME} %vec2, 0
  %bytes2 = bitcast {ELEM}* %elements2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  ret %{NAME} %vec2
}}

; Get a new vec object that starts at the index'th element of the existing vector, and has size size.
; If the specified size is greater than the remaining size, then the remaining size is used.
define %{NAME} @{NAME}.slice(%{NAME} %vec, i64 %index, i64 %size) {{
  ; Check if size greater than remaining size
  %currSize = extractvalue %{NAME} %vec, 1
  %remSize = sub i64 %currSize, %index
  %sgtr = icmp ugt i64 %size, %remSize
  %finSize = select i1 %sgtr, i64 %remSize, i64 %size

  %elements = extractvalue %{NAME} %vec, 0
  %newElements = getelementptr {ELEM}, {ELEM}* %elements, i64 %index
  %1 = insertvalue %{NAME} undef, {ELEM}* %newElements, 0
  %2 = insertvalue %{NAME} %1, i64 %finSize, 1

  ret %{NAME} %2
}}

; Initialize and return a new builder, with the given initial capacity.
define %{NAME}.bld @{NAME}.bld.new(i64 %capacity, %work_t* %cur.work, i32 %fixedSize) {{
  %elemSizePtr = getelementptr {ELEM}, {ELEM}* null, i32 1
  %elemSize = ptrtoint {ELEM}* %elemSizePtr to i64
  %newVb = call i8* @weld_rt_new_vb(i64 %elemSize, i64 %capacity, i32 %fixedSize)
  call void @{NAME}.bld.newPiece(%{NAME}.bld %newVb, %work_t* %cur.work)
  ret %{NAME}.bld %newVb
}}

define void @{NAME}.bld.newPiece(%{NAME}.bld %bldPtr, %work_t* %cur.work) {{
  call void @weld_rt_new_vb_piece(i8* %bldPtr, %work_t* %cur.work)
  ret void
}}

; Append a value into a builder, growing its space if needed.
define %{NAME}.bld @{NAME}.bld.merge(%{NAME}.bld %bldPtr, {ELEM} %value, i32 %myId) {{
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %bytesPtr = getelementptr inbounds %vb.vp, %vb.vp* %curPiecePtr, i32 0, i32 0
  %sizePtr = getelementptr inbounds %vb.vp, %vb.vp* %curPiecePtr, i32 0, i32 1
  %capacityPtr = getelementptr inbounds %vb.vp, %vb.vp* %curPiecePtr, i32 0, i32 2
  %size = load i64, i64* %sizePtr
  %capacity = load i64, i64* %capacityPtr
  %full = icmp eq i64 %size, %capacity
  br i1 %full, label %onFull, label %finish

onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr {ELEM}, {ELEM}* null, i32 1
  %elemSize = ptrtoint {ELEM}* %elemSizePtr to i64
  %bytes = load i8*, i8** %bytesPtr
  %allocSize = mul i64 %elemSize, %newCapacity
  %runId = call i64 @weld_rt_get_run_id()
  %newBytes = call i8* @weld_run_realloc(i64 %runId, i8* %bytes, i64 %allocSize)
  store i8* %newBytes, i8** %bytesPtr
  store i64 %newCapacity, i64* %capacityPtr
  br label %finish

finish:
  %bytes1 = load i8*, i8** %bytesPtr
  %elements = bitcast i8* %bytes1 to {ELEM}*
  %insertPtr = getelementptr {ELEM}, {ELEM}* %elements, i64 %size
  store {ELEM} %value, {ELEM}* %insertPtr
  %newSize = add i64 %size, 1
  store i64 %newSize, i64* %sizePtr
  ret %{NAME}.bld %bldPtr
}}

; Complete building a vector, trimming any extra space left while growing it.
define %{NAME} @{NAME}.bld.result(%{NAME}.bld %bldPtr) {{
  %out = call %vb.out @weld_rt_result_vb(i8* %bldPtr)
  %bytes = extractvalue %vb.out %out, 0
  %size = extractvalue %vb.out %out, 1
  %elems = bitcast i8* %bytes to {ELEM}*
  %1 = insertvalue %{NAME} undef, {ELEM}* %elems, 0
  %2 = insertvalue %{NAME} %1, i64 %size, 1
  ret %{NAME} %2
}}

; Get the length of a vector.
define i64 @{NAME}.size(%{NAME} %vec) alwaysinline {{
  %size = extractvalue %{NAME} %vec, 1
  ret i64 %size
}}

; Get a pointer to the index'th element.
define {ELEM}* @{NAME}.at(%{NAME} %vec, i64 %index) alwaysinline {{
  %elements = extractvalue %{NAME} %vec, 0
  %ptr = getelementptr {ELEM}, {ELEM}* %elements, i64 %index
  ret {ELEM}* %ptr
}}


; Get the length of a VecBuilder.
define i64 @{NAME}.bld.size(%{NAME}.bld nocapture %bldPtr, i32 %myId) readonly nounwind norecurse {{
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %sizePtr = getelementptr inbounds %vb.vp, %vb.vp* %curPiecePtr, i32 0, i32 1
  %size = load i64, i64* %sizePtr
  ret i64 %size
}}

; Get a pointer to the index'th element of a VecBuilder.
define {ELEM}* @{NAME}.bld.at(%{NAME}.bld nocapture %bldPtr, i64 %index, i32 %myId) readonly nounwind norecurse {{
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %bytesPtr = getelementptr inbounds %vb.vp, %vb.vp* %curPiecePtr, i32 0, i32 0
  %bytes = load i8*, i8** %bytesPtr
  %elements = bitcast i8* %bytes to {ELEM}*
  %ptr = getelementptr {ELEM}, {ELEM}* %elements, i64 %index
  ret {ELEM}* %ptr
}}
