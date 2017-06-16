; Template for the merger builder.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or f32).
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @f32)
; - VECSIZE: Size of vector.

%$NAME.bld.inner = type { $ELEM, <$VECSIZE x $ELEM> }
%$NAME.bld = type %$NAME.bld.inner*

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define %$NAME.bld @$NAME.bld.getPtrIndexed(%$NAME.bld %bldPtr, i32 %i) alwaysinline {
  %mergerPtr = getelementptr %$NAME.bld.inner, %$NAME.bld.inner* null, i32 1
  %mergerSize = ptrtoint %$NAME.bld.inner* %mergerPtr to i64
  %asPtr = bitcast %$NAME.bld %bldPtr to i8*
  %rawPtr = call i8* @get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %$NAME.bld
  ret %$NAME.bld %ptr
}

; Initialize and return a new merger.
define %$NAME.bld @$NAME.bld.new() {
  %bldSizePtr = getelementptr %$NAME.bld.inner, %$NAME.bld.inner* null, i32 1
  %bldSize = ptrtoint %$NAME.bld.inner* %bldSizePtr to i64
  %nworkers = call i32 @get_nworkers()
  %bldPtr = call i8* @new_merger(i64 %bldSize, i32 %nworkers)
  ; TODO(shoumik): For now, mergers can only be scalars. We may need to do some
  ; kind of initialization here like in the dictmerger if we allow more complex
  ; merger types.
  %bldPtrTyped = bitcast i8* %bldPtr to %$NAME.bld
  ret %$NAME.bld %bldPtrTyped
}

; Returns a pointer to a scalar value that an element can be merged into. %bldPtr is
; a value retrieved via getPtrIndexed.
define $ELEM* @$NAME.bld.scalarMergePtr(%$NAME.bld %bldPtr) {
  %bldScalarPtr = getelementptr %$NAME.bld.inner, %$NAME.bld %bldPtr, i32 0, i32 0
  ret $ELEM* %bldScalarPtr
}
; Returns a pointer to a vector value that an element can be merged into. %bldPtr is
; a value retrieved via getPtrIndexed.
define <$VECSIZE x $ELEM>* @$NAME.bld.vectorMergePtr(%$NAME.bld %bldPtr) {
  %bldScalarPtr = getelementptr %$NAME.bld.inner, %$NAME.bld %bldPtr, i32 0, i32 1
  ret <$VECSIZE x $ELEM>* %bldScalarPtr
}

; Dummy hash function; this is needed for structs that use these mergers as fields.
define i64 @$NAME.bld.hash(%$NAME.bld %bld) {
  ret i64 0
}

; Dummy comparison function; this is needed for structs that use these mergers as fields.
define i32 @$NAME.bld.cmp(%$NAME.bld %bld1, %$NAME.bld %bld2) {
  ret i32 -1
}
