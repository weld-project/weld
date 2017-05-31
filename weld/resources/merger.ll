; Template for the merger builder.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or f32).
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @f32)

%$NAME.bld = type { $ELEM* }

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define $ELEM* @$NAME.bld.getPtrIndexed(%$NAME.bld %bldInp, i32 %i) alwaysinline {
  %bldPtr = extractvalue %$NAME.bld %bldInp, 0
  %mergerPtr = getelementptr $ELEM, $ELEM* null, i32 1
  %mergerSize = ptrtoint $ELEM* %mergerPtr to i64
  %asPtr = bitcast $ELEM* %bldPtr to i8*
  %rawPtr = call i8* @get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to $ELEM*
  ret $ELEM* %ptr
}

; Returns a raw pointer to the builder which can be passed to the builder functions.
define i8* @$NAME.bld.getRawPtr(%$NAME.bld %bldInp) alwaysinline {
  %ptr = call $ELEM* @$NAME.bld.getPtrIndexed(%$NAME.bld %bldInp, i32 0)
  %ret = bitcast $ELEM* %ptr to i8*
  ret i8* %ret
}

; Initialize and return a new merger.
define %$NAME.bld @$NAME.bld.new() {
  %bldSizePtr = getelementptr $ELEM, $ELEM* null, i32 1
  %bldSize = ptrtoint $ELEM* %bldSizePtr to i64
  %nworkers = call i32 @get_nworkers()
  %bldPtr = call i8* @new_merger(i64 %bldSize, i32 %nworkers)
  ; TODO(shoumik): For now, mergers can only be scalars. We may need to do some
  ; kind of initialization here like in the dictmerger if we allow more complex
  ; merger types.
  %bldPtrTyped = bitcast i8* %bldPtr to $ELEM*
  %ret = insertvalue %$NAME.bld undef, $ELEM* %bldPtrTyped, 0
  ret %$NAME.bld %ret
}

; Returns a pointer to the value an element should be merged into.
; The caller should perform the merge operation on the contents of this pointer
; and then store the resulting value back.
define i8* @$NAME.bld.merge_ptr(%$NAME.bld %bldPtr, i32 %workerId) {
  %bldPtrLocal = call $ELEM* @$NAME.bld.getPtrIndexed(%$NAME.bld %bldPtr, i32 %workerId)
  %bldPtrRaw = bitcast $ELEM* %bldPtrLocal to i8*
  ret i8* %bldPtrRaw
}

; Dummy hash function; this is needed for structs that use these mergers as fields.
define i64 @$NAME.bld.hash(%$NAME.bld %bld) {
  ret i64 0
}

; Dummy comparison function; this is needed for structs that use these mergers as fields.
define i32 @$NAME.bld.cmp(%$NAME.bld %bld1, %$NAME.bld %bld2) {
  ret i32 -1
}
