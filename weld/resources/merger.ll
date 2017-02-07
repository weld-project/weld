; Template for the merger builder.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or f32).
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @f32)
; - OP: binary commutatitive merge operation (e.g. add or fadd)

%$NAME.bld.inner = type { $ELEM }
%$NAME.bld = type %$NAME.bld.inner*

; Initialize and return a new merger.
define %$NAME.bld @$NAME.bld.new() {
  %1 = insertvalue %$NAME.bld.inner undef, $ELEM 0, 0
  %bldSizePtr = getelementptr %$NAME.bld.inner* null, i32 1
  %bldSize = ptrtoint %$NAME.bld.inner* %bldSizePtr to i64
  %runId = call i64 @get_runid()
  %2 = call i8* @weld_rt_malloc(i64 %runId, i64 %bldSize)
  %3 = bitcast i8* %2 to %$NAME.bld.inner*
  store %$NAME.bld.inner %1, %$NAME.bld.inner* %3
  ret %$NAME.bld %3
}

; Merge a value into a builder.
define %$NAME.bld @$NAME.bld.merge(%$NAME.bld %bldPtr, $ELEM %value) {
  %bld = load %$NAME.bld.inner* %bldPtr
  %v = extractvalue %$NAME.bld.inner %bld, 0
  %1 = $OP $ELEM %v, %value
  %2 = insertvalue %$NAME.bld.inner %bld, $ELEM %1, 0
  store %$NAME.bld.inner %2, %$NAME.bld.inner* %bldPtr
  ret %$NAME.bld %bldPtr
}

; Merge a value into a builder.
define $ELEM @$NAME.bld.result(%$NAME.bld %bldPtr) {
  %bld = load %$NAME.bld.inner* %bldPtr
  %v = extractvalue %$NAME.bld.inner %bld, 0
  %toFree = bitcast %$NAME.bld.inner* %bldPtr to i8*
  %runId = call i64 @get_runid()
  call void @weld_rt_free(i64 %runId, i8* %toFree)
  ret $ELEM %v
}

; Dummy hash function; this is needed for structs that use these mergers as fields.
define i64 @$NAME.bld.hash(%$NAME.bld %bld) {
  ret i64 0
}

; Dummy comparison function; this is needed for structs that use these mergers as fields.
define i32 @$NAME.bld.cmp(%$NAME.bld %bld1, %$NAME.bld %bld2) {
  ret i32 -1
}
