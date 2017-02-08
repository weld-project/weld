; Template for the merger builder.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or f32).
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @f32)
; - OP: binary commutatitive merge operation (e.g. add or fadd)

%$NAME.bld.inner = type { $ELEM }
%$NAME.bld = type %$NAME.bld.inner*

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define %$NAME.bld @$NAME.bld.getPtrIndexed(%$NAME.bld %bldPtr, i32 %i) alwaysinline {
  %mergerPtr = getelementptr %$NAME.bld null, i32 1
  %mergerSize = ptrtoint %$NAME.bld %mergerPtr to i64
  %asPtr = bitcast %$NAME.bld %bldPtr to i8*
  %rawPtr = call i8* @get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %$NAME.bld
  ret %$NAME.bld %ptr
}

; Initialize and return a new merger.
define %$NAME.bld @$NAME.bld.new() {
  %bldSizePtr = getelementptr %$NAME.bld.inner* null, i32 1
  %bldSize = ptrtoint %$NAME.bld.inner* %bldSizePtr to i64
  %runId = call i64 @get_runid()
  %nworkers = call i32 @get_nworkers()
  %bldPtr = call i8* @new_merger(i64 %bldSize, i64 %runId, i32 %nworkers)
  ; TODO(shoumik): For now, mergers can only be scalars. We may need to do some
  ; kind of initialization here like in the dictmerger if we allow more complex
  ; merger types.
  %bldPtrTyped = bitcast i8* %bldPtr to %$NAME.bld
  ret %$NAME.bld %bldPtrTyped
}

; Merge a value into a builder.
define %$NAME.bld @$NAME.bld.merge(%$NAME.bld %bldPtr, $ELEM %value, i32 %workerId) {
  %bldPtrLocal = call %$NAME.bld @$NAME.bld.getPtrIndexed(%$NAME.bld %bldPtr, i32 %workerId)
  %bld = load %$NAME.bld.inner* %bldPtrLocal
  %v = extractvalue %$NAME.bld.inner %bld, 0
  %1 = $OP $ELEM %v, %value
  %2 = insertvalue %$NAME.bld.inner %bld, $ELEM %1, 0
  store %$NAME.bld.inner %2, %$NAME.bld.inner* %bldPtrLocal
  ret %$NAME.bld %bldPtr
}

; Merge a value into a builder.
define $ELEM @$NAME.bld.result(%$NAME.bld %bldArgPtr) {
  %bldPtrFirst = call %$NAME.bld @$NAME.bld.getPtrIndexed(%$NAME.bld %bldArgPtr, i32 0)
  %bldFirst = load %$NAME.bld.inner* %bldPtrFirst
  %first = extractvalue %$NAME.bld.inner %bldFirst, 0
  %nworkers = call i32 @get_nworkers()
  br label %entry
entry:
  %cond = icmp ult i32 1, %nworkers
  br i1 %cond, label %body, label %done
body:
  %i = phi i32 [ 1, %entry ], [ %i2, %body ]
  %cur = phi $ELEM [ %first, %entry ], [ %updated, %body ]
  %bldPtr = call %$NAME.bld @$NAME.bld.getPtrIndexed(%$NAME.bld %bldArgPtr, i32 %i)
  %bld = load %$NAME.bld.inner* %bldPtr
  %val = extractvalue %$NAME.bld.inner %bld, 0
  %updated = $OP $ELEM %val, %cur
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done
done:
  %final = phi $ELEM [ %first, %entry ], [ %updated, %body ]
  %runId = call i64 @get_runid()
  %asPtr = bitcast %$NAME.bld %bldArgPtr to i8*
  call void @free_merger(i8* %asPtr, i64 %runId)
  ret $ELEM %final
}

; Dummy hash function; this is needed for structs that use these mergers as fields.
define i64 @$NAME.bld.hash(%$NAME.bld %bld) {
  ret i64 0
}

; Dummy comparison function; this is needed for structs that use these mergers as fields.
define i32 @$NAME.bld.cmp(%$NAME.bld %bld1, %$NAME.bld %bld2) {
  ret i32 -1
}
