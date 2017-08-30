; Vector extensions for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - VECSIZE: Size of vectors.

; Get a pointer to the index'th element, fetching a vector
define <{VECSIZE} x {ELEM}>* @{NAME}.vat(%{NAME} %vec, i64 %index) {{
  %elements = extractvalue %{NAME} %vec, 0
  %ptr = getelementptr {ELEM}, {ELEM}* %elements, i64 %index
  %retPtr = bitcast {ELEM}* %ptr to <{VECSIZE} x {ELEM}>*
  ret <{VECSIZE} x {ELEM}>* %retPtr
}}

; Append a value into a builder, growing its space if needed.
define %{NAME}.bld @{NAME}.bld.vmerge(%{NAME}.bld %bldPtr, <{VECSIZE} x {ELEM}> %value, i32 %myId) {{
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %adjusted = add i64 %size, {VECSIZE}
  %full = icmp sgt i64 %adjusted, %capacity
  br i1 %full, label %onFull, label %finish

onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr {ELEM}, {ELEM}* null, i32 1
  %elemSize = ptrtoint {ELEM}* %elemSizePtr to i64
  %bytes = extractvalue %vb.vp %curPiece, 0
  %allocSize = mul i64 %elemSize, %newCapacity
  %runId = call i64 @weld_rt_get_run_id()
  %newBytes = call i8* @weld_run_realloc(i64 %runId, i8* %bytes, i64 %allocSize)
  %curPiece1 = insertvalue %vb.vp %curPiece, i8* %newBytes, 0
  %curPiece2 = insertvalue %vb.vp %curPiece1, i64 %newCapacity, 2
  br label %finish

finish:
  %curPiece3 = phi %vb.vp [ %curPiece, %entry ], [ %curPiece2, %onFull ]
  %bytes1 = extractvalue %vb.vp %curPiece3, 0
  %elements = bitcast i8* %bytes1 to {ELEM}*
  %insertPtr = getelementptr {ELEM}, {ELEM}* %elements, i64 %size
  %vecInsertPtr = bitcast {ELEM}* %insertPtr to <{VECSIZE} x {ELEM}>*
  store <{VECSIZE} x {ELEM}> %value, <{VECSIZE} x {ELEM}>* %vecInsertPtr, align 1
  %newSize = add i64 %size, {VECSIZE}
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %{NAME}.bld %bldPtr
}}
