; Template for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)

%$NAME = type { $ELEM*, i64 }           ; elements, size
%$NAME.bld.inner = type { $ELEM*, i64, i64 }  ; elements, size, capacity
%$NAME.bld = type %$NAME.bld.inner*

; Initialize and return a new vector with the given size.
define %$NAME @$NAME.new(i64 %size) {
  %elemSizePtr = getelementptr $ELEM* null, i32 1
  %elemSize = ptrtoint $ELEM* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %bytes = call i8* @malloc(i64 %allocSize)
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
define %$NAME.bld @$NAME.bld.new(i64 %capacity) {
  %elemSizePtr = getelementptr $ELEM* null, i32 1
  %elemSize = ptrtoint $ELEM* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %capacity
  %bytes = call i8* @malloc(i64 %allocSize)
  %elements = bitcast i8* %bytes to $ELEM*
  %1 = insertvalue %$NAME.bld.inner undef, $ELEM* %elements, 0
  %2 = insertvalue %$NAME.bld.inner %1, i64 0, 1
  %3 = insertvalue %$NAME.bld.inner %2, i64 %capacity, 2
  %bldSizePtr = getelementptr %$NAME.bld.inner* null, i32 1
  %bldSize = ptrtoint %$NAME.bld.inner* %bldSizePtr to i64
  %4 = call i8* @malloc(i64 %bldSize)
  %5 = bitcast i8* %4 to %$NAME.bld.inner*
  store %$NAME.bld.inner %3, %$NAME.bld.inner* %5
  ret %$NAME.bld %5
}

; Append a value into a builder, growing its space if needed.
define %$NAME.bld @$NAME.bld.merge(%$NAME.bld %bldPtr, $ELEM %value) {
entry:
  %bld = load %$NAME.bld.inner* %bldPtr
  %size = extractvalue %$NAME.bld.inner %bld, 1
  %capacity = extractvalue %$NAME.bld.inner %bld, 2
  %full = icmp eq i64 %size, %capacity
  br i1 %full, label %onFull, label %finish

onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr $ELEM* null, i32 1
  %elemSize = ptrtoint $ELEM* %elemSizePtr to i64
  %elements = extractvalue %$NAME.bld.inner %bld, 0
  %bytes = bitcast $ELEM* %elements to i8*
  %allocSize = mul i64 %elemSize, %newCapacity
  %newBytes = call i8* @realloc(i8* %bytes, i64 %allocSize)
  %newElements = bitcast i8* %newBytes to $ELEM*
  %bld1 = insertvalue %$NAME.bld.inner %bld, $ELEM* %newElements, 0
  %bld2 = insertvalue %$NAME.bld.inner %bld1, i64 %newCapacity, 2
  br label %finish

finish:
  %bld3 = phi %$NAME.bld.inner [ %bld, %entry ], [ %bld2, %onFull ]
  %elements3 = extractvalue %$NAME.bld.inner %bld3, 0
  %insertPtr = getelementptr $ELEM* %elements3, i64 %size
  store $ELEM %value, $ELEM* %insertPtr
  %newSize = add i64 %size, 1
  %bld4 = insertvalue %$NAME.bld.inner %bld3, i64 %newSize, 1
  store %$NAME.bld.inner %bld4, %$NAME.bld.inner* %bldPtr
  ret %$NAME.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %$NAME @$NAME.bld.result(%$NAME.bld %bldPtr) {
  %bld = load %$NAME.bld.inner* %bldPtr
  %elements = extractvalue %$NAME.bld.inner %bld, 0
  %size = extractvalue %$NAME.bld.inner %bld, 1
  %bytes = bitcast $ELEM* %elements to i8*
  %elemSizePtr = getelementptr $ELEM* null, i32 1
  %elemSize = ptrtoint $ELEM* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %newBytes = call i8* @realloc(i8* %bytes, i64 %allocSize)
  %newElements = bitcast i8* %newBytes to $ELEM*
  %toFree = bitcast %$NAME.bld.inner* %bldPtr to i8*
  call void @free(i8* %toFree)
  %1 = insertvalue %$NAME undef, $ELEM* %newElements, 0
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
define i64 @$NAME.bld.size(%$NAME.bld %bldPtr) {
  %bld = load %$NAME.bld.inner* %bldPtr
  %size = extractvalue %$NAME.bld.inner %bld, 1
  ret i64 %size
}

; Get a pointer to the index'th element of a VecBuilder.
define $ELEM* @$NAME.bld.at(%$NAME.bld %bldPtr, i64 %index) {
  %bld = load %$NAME.bld.inner* %bldPtr
  %elements = extractvalue %$NAME.bld.inner %bld, 0
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
