; Template for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)

%$NAME = type { $ELEM*, i64 }           ; elements, size
%$NAME.bld = type i8*

; Initialize and return a new vector with the given size.
define %$NAME @$NAME.new(i64 %size) {
  %elemSizePtr = getelementptr $ELEM* null, i32 1
  %elemSize = ptrtoint $ELEM* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %runId = call i64 @get_runid()
  %bytes = call i8* @weld_rt_malloc(i64 %runId, i64 %allocSize)
  %elements = bitcast i8* %bytes to $ELEM*
  %1 = insertvalue %$NAME undef, $ELEM* %elements, 0
  %2 = insertvalue %$NAME %1, i64 %size, 1
  ret %$NAME %2
}

; Clone a vector.
define %$NAME @$NAME.clone(%$NAME %vec) {
  %elements = extractvalue %$NAME %vec, 0
  %size = extractvalue %$NAME %vec, 1
  %entrySizePtr = getelementptr $ELEM* null, i32 1
  %entrySize = ptrtoint $ELEM* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %size
  %bytes = bitcast $ELEM* %elements to i8*
  %vec2 = call %$NAME @$NAME.new(i64 %size)
  %elements2 = extractvalue %$NAME %vec2, 0
  %bytes2 = bitcast $ELEM* %elements2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  ret %$NAME %vec2
}

; Initialize and return a new builder, with the given initial capacity.
define %$NAME.bld @$NAME.bld.new(i64 %capacity, %work_t* %cur.work) {
  %elemSizePtr = getelementptr $ELEM* null, i32 1
  %elemSize = ptrtoint $ELEM* %elemSizePtr to i64
  %newVb = call i8* @new_vb(i64 %elemSize, i64 %capacity)
  call void @$NAME.bld.newPiece(%$NAME.bld %newVb, %work_t* %cur.work)
  ret %$NAME.bld %newVb
}

define void @$NAME.bld.newPiece(%$NAME.bld %bldPtr, %work_t* %cur.work) {
  call void @new_piece(i8* %bldPtr, %work_t* %cur.work)
  ret void
}

; Append a value into a builder, growing its space if needed.
define %$NAME.bld @$NAME.bld.merge(%$NAME.bld %bldPtr, $ELEM %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @cur_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %full = icmp eq i64 %size, %capacity
  br i1 %full, label %onFull, label %finish

onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr $ELEM* null, i32 1
  %elemSize = ptrtoint $ELEM* %elemSizePtr to i64
  %bytes = extractvalue %vb.vp %curPiece, 0
  %allocSize = mul i64 %elemSize, %newCapacity
  %runId = call i64 @get_runid()
  %newBytes = call i8* @weld_rt_realloc(i64 %runId, i8* %bytes, i64 %allocSize)
  %curPiece1 = insertvalue %vb.vp %curPiece, i8* %newBytes, 0
  %curPiece2 = insertvalue %vb.vp %curPiece1, i64 %newCapacity, 2
  br label %finish

finish:
  %curPiece3 = phi %vb.vp [ %curPiece, %entry ], [ %curPiece2, %onFull ]
  %bytes1 = extractvalue %vb.vp %curPiece3, 0
  %elements = bitcast i8* %bytes1 to $ELEM*
  %insertPtr = getelementptr $ELEM* %elements, i64 %size
  store $ELEM %value, $ELEM* %insertPtr
  %newSize = add i64 %size, 1
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %$NAME.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %$NAME @$NAME.bld.result(%$NAME.bld %bldPtr) {
  %out = call %vb.out @result_vb(i8* %bldPtr)
  %bytes = extractvalue %vb.out %out, 0
  %size = extractvalue %vb.out %out, 1
  %elems = bitcast i8* %bytes to $ELEM*
  %1 = insertvalue %$NAME undef, $ELEM* %elems, 0
  %2 = insertvalue %$NAME %1, i64 %size, 1
  ret %$NAME %2
}

; Get the length of a vector.
define i64 @$NAME.size(%$NAME %vec) {
  %size = extractvalue %$NAME %vec, 1
  ret i64 %size
}

; Get a pointer to the index'th element.
define $ELEM* @$NAME.at(%$NAME %vec, i64 %index) {
  %elements = extractvalue %$NAME %vec, 0
  %ptr = getelementptr $ELEM* %elements, i64 %index
  ret $ELEM* %ptr
}


; Get the length of a VecBuilder.
define i64 @$NAME.bld.size(%$NAME.bld %bldPtr, i32 %myId) {
  %curPiecePtr = call %vb.vp* @cur_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  ret i64 %size
}

; Get a pointer to the index'th element of a VecBuilder.
define $ELEM* @$NAME.bld.at(%$NAME.bld %bldPtr, i64 %index, i32 %myId) {
  %curPiecePtr = call %vb.vp* @cur_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp* %curPiecePtr
  %bytes = extractvalue %vb.vp %curPiece, 0
  %elements = bitcast i8* %bytes to $ELEM*
  %ptr = getelementptr $ELEM* %elements, i64 %index
  ret $ELEM* %ptr
}

; Compute the hash code of a vector.
; TODO: We should hash more bytes at a time if elements are non-pointer types.
define i64 @$NAME.hash(%$NAME %vec) {
entry:
  %elements = extractvalue %$NAME %vec, 0
  %size = extractvalue %$NAME %vec, 1
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done

body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %prevHash = phi i64 [ 0, %entry ], [ %newHash, %body ]
  %ptr = getelementptr $ELEM* %elements, i64 %i
  %elem = load $ELEM* %ptr
  %elemHash = call i64 $ELEM_PREFIX.hash($ELEM %elem)
  %newHash = call i64 @hash_combine(i64 %prevHash, i64 %elemHash)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done

done:
  %res = phi i64 [ 0, %entry ], [ %newHash, %body ]
  ret i64 %res
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i64 @$NAME.bld.hash(%$NAME.bld %bld) {
  ret i64 0
}

; Compare two vectors lexicographically.
define i32 @$NAME.cmp(%$NAME %a, %$NAME %b) {
entry:
  %elemsA = extractvalue %$NAME %a, 0
  %elemsB = extractvalue %$NAME %b, 0
  %sizeA = extractvalue %$NAME %a, 1
  %sizeB = extractvalue %$NAME %b, 1
  %cond1 = icmp ult i64 %sizeA, %sizeB
  %minSize = select i1 %cond1, i64 %sizeA, i64 %sizeB
  %cond = icmp ult i64 0, %minSize
  br i1 %cond, label %body, label %done

body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body2 ]
  %ptrA = getelementptr $ELEM* %elemsA, i64 %i
  %ptrB = getelementptr $ELEM* %elemsB, i64 %i
  %elemA = load $ELEM* %ptrA
  %elemB = load $ELEM* %ptrB
  %cmp = call i32 $ELEM_PREFIX.cmp($ELEM %elemA, $ELEM %elemB)
  %ne = icmp ne i32 %cmp, 0
  br i1 %ne, label %return, label %body2

return:
  ret i32 %cmp

body2:
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %minSize
  br i1 %cond2, label %body, label %done

done:
  %res = call i32 @i64.cmp(i64 %sizeA, i64 %sizeB)
  ret i32 %res
}

; Dummy comparison function; this is needed for structs that use these vecbuilders as fields.
define i32 @$NAME.bld.cmp(%$NAME.bld %bld1, %$NAME.bld %bld2) {
  ret i32 -1
}
