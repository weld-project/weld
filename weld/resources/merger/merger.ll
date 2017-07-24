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
  %rawPtr = call i8* @weld_rt_get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %$NAME.bld
  ret %$NAME.bld %ptr
}

; Initialize and return a new merger.
define %$NAME.bld @$NAME.bld.new() {
  %bldSizePtr = getelementptr %$NAME.bld.inner, %$NAME.bld.inner* null, i32 1
  %bldSize = ptrtoint %$NAME.bld.inner* %bldSizePtr to i64
  %nworkers = call i32 @weld_rt_get_nworkers()
  %bldPtr = call i8* @weld_rt_new_merger(i64 %bldSize, i32 %nworkers)
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
  %bldVectorPtr = getelementptr %$NAME.bld.inner, %$NAME.bld %bldPtr, i32 0, i32 1
  ret <$VECSIZE x $ELEM>* %bldVectorPtr
}

; Clear the vector by assigning each vector element to a user defined identity value.
define void @$NAME.bld.clearVector(%$NAME.bld %bldPtr, $ELEM %identity) {
    %vectorPtr = call <$VECSIZE x $ELEM>* @$NAME.bld.vectorMergePtr(%$NAME.bld %bldPtr)
    %vector = load <$VECSIZE x $ELEM>, <$VECSIZE x $ELEM>* %vectorPtr
    br label %entry
entry:
  %cond = icmp ult i32 0, $VECSIZE
  br i1 %cond, label %body, label %done
body:
  %i = phi i32 [ 0, %entry ], [ %i2, %body ]
  %vector2 = phi <$VECSIZE x $ELEM> [ %vector, %entry], [ %vector3, %body ]
  %vector3 = insertelement <$VECSIZE x $ELEM> %vector2, $ELEM %identity, i32 %i
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, $VECSIZE
  br i1 %cond2, label %body, label %done
done:
  %vector4 = phi <$VECSIZE x $ELEM> [ %vector, %entry ], [ %vector3, %body ]
  store <$VECSIZE x $ELEM> %vector4, <$VECSIZE x $ELEM>* %vectorPtr
  ret void
}

; Dummy hash function; this is needed for structs that use these mergers as fields.
define i32 @$NAME.bld.hash(%$NAME.bld %bld) {
  ret i32 0
}

; Dummy comparison function; this is needed for structs that use these mergers as fields.
define i32 @$NAME.bld.cmp(%$NAME.bld %bld1, %$NAME.bld %bld2) {
  ret i32 -1
}

; Dummy equality function; this is needed for structs that use these mergers as fields.
define i1 @$NAME.bld.eq(%$NAME.bld %bld1, %$NAME.bld %bld2) {
  ret i1 0
}
