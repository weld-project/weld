; Template for the merger builder.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or f32).
; - VECSIZE: Size of vector.

%{NAME}.bld.piece = type {{ {ELEM}, <{VECSIZE} x {ELEM}> }}
%{NAME}.bld.piecePtr = type %{NAME}.bld.piece*
%{NAME}.bld = type {{ %{NAME}.bld.piece*, i8*, i1 }} ; local version, global version, hasGlobal

; Returns a pointer to merger piece for index i (generally, i is the thread ID).
define %{NAME}.bld.piecePtr @{NAME}.bld.getPtrIndexed(%{NAME}.bld* %bldPtr, i32 %i) alwaysinline {{
  %bld = load %{NAME}.bld, %{NAME}.bld* %bldPtr
  %mergerPtr = getelementptr %{NAME}.bld.piece, %{NAME}.bld.piece* null, i32 1
  %mergerSize = ptrtoint %{NAME}.bld.piece* %mergerPtr to i64
  %asPtr = extractvalue %{NAME}.bld %bld, 1
  %rawPtr = call i8* @weld_rt_get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %{NAME}.bld.piecePtr
  ret %{NAME}.bld.piecePtr %ptr
}}

; Clear the merger piece by assigning the scalar element and each vector element to
; a user defined identity value.
define %{NAME}.bld.piece @{NAME}.bld.clearPieceInternal({ELEM} %identity) {{
  %1 = insertvalue %{NAME}.bld.piece undef, {ELEM} %identity, 0
  %vector = extractvalue %{NAME}.bld.piece %1, 1
  br label %entry
entry:
  %cond = icmp ult i32 0, {VECSIZE}
  br i1 %cond, label %body, label %done
body:
  %i = phi i32 [ 0, %entry ], [ %i2, %body ]
  %vector2 = phi <{VECSIZE} x {ELEM}> [ %vector, %entry], [ %vector3, %body ]
  %vector3 = insertelement <{VECSIZE} x {ELEM}> %vector2, {ELEM} %identity, i32 %i
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, {VECSIZE}
  br i1 %cond2, label %body, label %done
done:
  %vector4 = phi <{VECSIZE} x {ELEM}> [ %vector, %entry ], [ %vector3, %body ]
  %2 = insertvalue %{NAME}.bld.piece %1, <{VECSIZE} x {ELEM}> %vector4, 1
  ret %{NAME}.bld.piece %2
}}

; Clear the merger piece by assigning the scalar element and each vector element to
; a user defined identity value.
define void @{NAME}.bld.clearPiece(%{NAME}.bld.piecePtr %piecePtr, {ELEM} %identity) {{
  %result = call %{NAME}.bld.piece @{NAME}.bld.clearPieceInternal({ELEM} %identity)
  store %{NAME}.bld.piece %result, %{NAME}.bld.piecePtr %piecePtr
  ret void
}}

define void @{NAME}.bld.clearStackPiece(%{NAME}.bld* %bldPtr, {ELEM} %identity) {{
  %stackPiecePtr = getelementptr %{NAME}.bld, %{NAME}.bld* %bldPtr, i32 0, i32 0
  %stackPiece = load %{NAME}.bld.piecePtr, %{NAME}.bld.piecePtr* %stackPiecePtr
  call void @{NAME}.bld.clearPiece(%{NAME}.bld.piecePtr %stackPiece, {ELEM} %identity)
  ret void
}}

define void @{NAME}.bld.insertStackPiece(%{NAME}.bld* %bldPtr, %{NAME}.bld.piecePtr %stackPiece) {{
  %stackPiecePtr = getelementptr %{NAME}.bld, %{NAME}.bld* %bldPtr, i32 0, i32 0
  store %{NAME}.bld.piecePtr %stackPiece, %{NAME}.bld.piecePtr* %stackPiecePtr
  ret void
}}

define void @{NAME}.bld.initGlobalIfNeeded(%{NAME}.bld* %bldPtr, {ELEM} %identity) {{
  %bld = load %{NAME}.bld, %{NAME}.bld* %bldPtr
  %isGlobal = extractvalue %{NAME}.bld %bld, 2
  br i1 %isGlobal, label %done, label %cont
cont:
  %bldSizePtr = getelementptr %{NAME}.bld.piece, %{NAME}.bld.piece* null, i32 1
  %bldSize = ptrtoint %{NAME}.bld.piece* %bldSizePtr to i64
  %nworkers = call i32 @weld_rt_get_nworkers()
  %bldGlobal = call i8* @weld_rt_new_merger(i64 %bldSize, i32 %nworkers)
  %1 = insertvalue %{NAME}.bld %bld, i8* %bldGlobal, 1
  %2 = insertvalue %{NAME}.bld %1, i1 1, 2
  store %{NAME}.bld %2, %{NAME}.bld* %bldPtr
  br label %body
body:
  %i = phi i32 [ 0, %cont ], [ %i2, %body ]
  %curPiecePtr = call %{NAME}.bld.piecePtr @{NAME}.bld.getPtrIndexed(%{NAME}.bld* %bldPtr, i32 %i)
  call void @{NAME}.bld.clearPiece(%{NAME}.bld.piecePtr %curPiecePtr, {ELEM} %identity)
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done
done:
  ret void
}}

; Returns a pointer to a scalar value that an element can be merged into.
define {ELEM}* @{NAME}.bld.scalarMergePtrForPiece(%{NAME}.bld.piecePtr %piecePtr) {{
  %bldScalarPtr = getelementptr %{NAME}.bld.piece, %{NAME}.bld.piecePtr %piecePtr, i32 0, i32 0
  ret {ELEM}* %bldScalarPtr
}}

; Returns a pointer to a vector value that an element can be merged into.
define <{VECSIZE} x {ELEM}>* @{NAME}.bld.vectorMergePtrForPiece(%{NAME}.bld.piecePtr %piecePtr) {{
  %bldVectorPtr = getelementptr %{NAME}.bld.piece, %{NAME}.bld.piecePtr %piecePtr, i32 0, i32 1
  ret <{VECSIZE} x {ELEM}>* %bldVectorPtr
}}

; Initialize and return a new merger.
define %{NAME}.bld @{NAME}.bld.new({ELEM} %identity, {ELEM} %init, %{NAME}.bld.piecePtr %stackPiece) {{
  ; TODO(shoumik): For now, mergers can only be scalars. We may need to do some
  ; kind of initialization here like in the dictmerger if we allow more complex
  ; merger types.
  %piece = call %{NAME}.bld.piece @{NAME}.bld.clearPieceInternal({ELEM} %identity)
  store %{NAME}.bld.piece %piece, %{NAME}.bld.piecePtr %stackPiece
  %scalarPtr = call {ELEM}* @{NAME}.bld.scalarMergePtrForPiece(%{NAME}.bld.piecePtr %stackPiece)
  store {ELEM} %init, {ELEM}* %scalarPtr
  %1 = insertvalue %{NAME}.bld undef, %{NAME}.bld.piecePtr %stackPiece, 0
  %2 = insertvalue %{NAME}.bld %1, i1 0, 2
  ret %{NAME}.bld %2
}}

; Returns a pointer to a scalar value that an element can be merged into.
define {ELEM}* @{NAME}.bld.scalarMergePtrForStackPiece(%{NAME}.bld* %bldPtr) {{
  %stackPiecePtr = getelementptr %{NAME}.bld, %{NAME}.bld* %bldPtr, i32 0, i32 0
  %stackPiece = load %{NAME}.bld.piecePtr, %{NAME}.bld.piecePtr* %stackPiecePtr
  %bldScalarPtr = getelementptr %{NAME}.bld.piece, %{NAME}.bld.piecePtr %stackPiece, i32 0, i32 0
  ret {ELEM}* %bldScalarPtr
}}

; Returns a pointer to a vector value that an element can be merged into.
define <{VECSIZE} x {ELEM}>* @{NAME}.bld.vectorMergePtrForStackPiece(%{NAME}.bld* %bldPtr) {{
  %stackPiecePtr = getelementptr %{NAME}.bld, %{NAME}.bld* %bldPtr, i32 0, i32 0
  %stackPiece = load %{NAME}.bld.piecePtr, %{NAME}.bld.piecePtr* %stackPiecePtr
  %bldVectorPtr = getelementptr %{NAME}.bld.piece, %{NAME}.bld.piecePtr %stackPiece, i32 0, i32 1
  ret <{VECSIZE} x {ELEM}>* %bldVectorPtr
}}

define i1 @{NAME}.bld.isGlobal(%{NAME}.bld* %bldPtr) {{
  %bld = load %{NAME}.bld, %{NAME}.bld* %bldPtr
  %isGlobal = extractvalue %{NAME}.bld %bld, 2
  ret i1 %isGlobal
}}

define i8* @{NAME}.bld.getGlobal(%{NAME}.bld* %bldPtr) {{
  %bld = load %{NAME}.bld, %{NAME}.bld* %bldPtr
  %global = extractvalue %{NAME}.bld %bld, 1
  ret i8* %global
}}
