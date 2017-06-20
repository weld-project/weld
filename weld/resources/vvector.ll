; Vector extensions for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)
; - VECSIZE: Size of vectors.

; Get a pointer to the index'th element, fetching a vector
define <$VECSIZE x $ELEM>* @$NAME.vat(%$NAME %vec, i64 %index) {
  %elements = extractvalue %$NAME %vec, 0
  %ptr = getelementptr $ELEM, $ELEM* %elements, i64 %index
  %retPtr = bitcast $ELEM* %ptr to <$VECSIZE x $ELEM>*
  ret <$VECSIZE x $ELEM>* %retPtr
}
