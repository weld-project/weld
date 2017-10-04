; PRELUDE:

; Common prelude to add at the start of generated LLVM modules.

; Unsigned data types -- we use these in the generated code for clarity and to make
; template substitution work nicely when calling type-specific functions
%u8 = type i8;
%u16 = type i16;
%u32 = type i32;
%u64 = type i64;

; LLVM intrinsic functions
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)
declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i32, i1)
declare float @llvm.exp.f32(float)
declare double @llvm.exp.f64(double)

declare float @llvm.log.f32(float)
declare double @llvm.log.f64(double)

declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>)
declare <4 x float> @llvm.log.v4f32(<4 x float>)
declare <8 x float> @llvm.log.v8f32(<8 x float>)
declare <4 x float> @llvm.exp.v4f32(<4 x float>)
declare <8 x float> @llvm.exp.v8f32(<8 x float>)

declare <2 x double> @llvm.sqrt.v2f64(<2 x double>)
declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)
declare <2 x double> @llvm.log.v2f64(<2 x double>)
declare <4 x double> @llvm.log.v4f64(<4 x double>)
declare <2 x double> @llvm.exp.v2f64(<2 x double>)
declare <4 x double> @llvm.exp.v4f64(<4 x double>)

declare float @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)

declare i64 @llvm.ctlz.i64(i64, i1)

; std library functions
declare i8* @malloc(i64)
declare void @qsort(i8*, i64, i64, i32 (i8*, i8*)*)
declare i32 @memcmp(i8*, i8*, i64)
declare float @erff(float)
declare double @erf(double)

; Weld runtime functions

declare i64     @weld_run_begin(void (%work_t*)*, i8*, i64, i32)
declare i8*     @weld_run_get_result(i64)

declare i8*     @weld_run_malloc(i64, i64)
declare i8*     @weld_run_realloc(i64, i8*, i64)
declare void    @weld_run_free(i64, i8*)

declare void    @weld_run_set_errno(i64, i64)
declare i64     @weld_run_get_errno(i64)

declare i32     @weld_rt_thread_id()
declare void    @weld_rt_abort_thread()
declare i32     @weld_rt_get_nworkers()
declare i64     @weld_rt_get_run_id()

declare void    @weld_rt_start_loop(%work_t*, i8*, i8*, void (%work_t*)*, void (%work_t*)*, i64, i64, i32)
declare void    @weld_rt_set_result(i8*)

declare i8*     @weld_rt_new_vb(i64, i64, i32)
declare void    @weld_rt_new_vb_piece(i8*, %work_t*)
declare %vb.vp* @weld_rt_cur_vb_piece(i8*, i32)
declare %vb.out @weld_rt_result_vb(i8*)

declare i8*     @weld_rt_new_merger(i64, i32)
declare i8*     @weld_rt_get_merger_at_index(i8*, i64, i32)
declare void    @weld_rt_free_merger(i8*)

; Parallel runtime structures
; work_t struct in runtime.h
%work_t = type { i8*, i64, i64, i64, i32, i64*, i64*, i32, i64, void (%work_t*)*, %work_t*, i32, i32, i32 }
; vec_piece struct in runtime.h
%vb.vp = type { i8*, i64, i64, i64*, i64*, i32 }
; vec_output struct in runtime.h
%vb.out = type { i8*, i64 }

; Input argument (input data pointer, nworkers, mem_limit)
%input_arg_t = type { i64, i32, i64 }
; Return type (output data pointer, run ID, errno)
%output_arg_t = type { i64, i64, i64 }

; Hash functions

; Combines two hash values using the method in Effective Java
define i32 @hash_combine(i32 %start, i32 %value) alwaysinline {
  ; return 31 * start + value
  %1 = mul i32 %start, 31
  %2 = add i32 %1, %value
  ret i32 %2
}

; Mixes the bits in a hash code, similar to Java's HashMap
define i32 @hash_finalize(i32 %hash) {
  ; h ^= (h >>> 20) ^ (h >>> 12);
  ; return h ^ (h >>> 7) ^ (h >>> 4);
  %1 = lshr i32 %hash, 20
  %2 = lshr i32 %hash, 12
  %3 = xor i32 %hash, %1
  %h2 = xor i32 %3, %2
  %4 = lshr i32 %h2, 7
  %5 = lshr i32 %h2, 4
  %6 = xor i32 %h2, %4
  %res = xor i32 %6, %5
  ret i32 %res
}

define i32 @i64.hash(i64 %arg) {
  ; return (i32) ((arg >>> 32) ^ arg)
  %1 = lshr i64 %arg, 32
  %2 = xor i64 %arg, %1
  %3 = trunc i64 %2 to i32
  ret i32 %3
}

define i32 @i32.hash(i32 %arg) {
  ret i32 %arg
}

define i32 @i16.hash(i16 %arg) {
  %1 = zext i16 %arg to i32
  ret i32 %1
}

define i32 @i8.hash(i8 %arg) {
  %1 = zext i8 %arg to i32
  ret i32 %1
}

define i32 @i1.hash(i1 %arg) {
  %1 = zext i1 %arg to i32
  ret i32 %1
}

define i32 @u64.hash(%u64 %arg) {
  %1 = call i32 @i64.hash(i64 %arg)
  ret i32 %1
}

define i32 @u32.hash(%u32 %arg) {
  %1 = call i32 @i32.hash(i32 %arg)
  ret i32 %1
}

define i32 @u16.hash(%u16 %arg) {
  %1 = call i32 @i16.hash(i16 %arg)
  ret i32 %1
}

define i32 @u8.hash(%u8 %arg) {
  %1 = call i32 @i8.hash(i8 %arg)
  ret i32 %1
}

define i32 @float.hash(float %arg) {
  %1 = bitcast float %arg to i32
  ret i32 %1
}

define i32 @double.hash(double %arg) {
  %1 = bitcast double %arg to i64
  %2 = call i32 @i64.hash(i64 %1)
  ret i32 %2
}

; Comparison functions

define i32 @i64.cmp(i64 %a, i64 %b) {
  %1 = icmp eq i64 %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = icmp slt i64 %a, %b
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

define i32 @i32.cmp(i32 %a, i32 %b) {
  %1 = icmp eq i32 %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = icmp slt i32 %a, %b
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

define i32 @i16.cmp(i16 %a, i16 %b) {
  %1 = icmp eq i16 %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = icmp slt i16 %a, %b
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

define i32 @i8.cmp(i8 %a, i8 %b) {
  %1 = icmp eq i8 %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = icmp slt i8 %a, %b
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

define i32 @i1.cmp(i1 %a, i1 %b) {
  %1 = icmp eq i1 %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = select i1 %b, i32 -1, i32 1
  ret i32 %2
}

define i32 @u64.cmp(%u64 %a, %u64 %b) {
  %1 = icmp eq i64 %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = icmp ult i64 %a, %b
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

define i32 @u32.cmp(%u32 %a, %u32 %b) {
  %1 = icmp eq i32 %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = icmp ult i32 %a, %b
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

define i32 @u16.cmp(%u16 %a, %u16 %b) {
  %1 = icmp eq i16 %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = icmp ult i16 %a, %b
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

define i32 @u8.cmp(%u8 %a, %u8 %b) {
  %1 = icmp eq i8 %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = icmp ult i8 %a, %b
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

define i32 @float.cmp(float %a, float %b) {
  %1 = fcmp oeq float %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = fcmp olt float %a, %b
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

define i32 @double.cmp(double %a, double %b) {
  %1 = fcmp oeq double %a, %b
  br i1 %1, label %eq, label %ne
eq:
  ret i32 0
ne:
  %2 = fcmp olt double %a, %b
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

; Equality functions

define i1 @i64.eq(i64 %a, i64 %b) {
  %1 = icmp eq i64 %a, %b
  ret i1 %1
}

define i1 @i32.eq(i32 %a, i32 %b) {
  %1 = icmp eq i32 %a, %b
  ret i1 %1
}

define i1 @i16.eq(i16 %a, i16 %b) {
  %1 = icmp eq i16 %a, %b
  ret i1 %1
}

define i1 @i8.eq(i8 %a, i8 %b) {
  %1 = icmp eq i8 %a, %b
  ret i1 %1
}

define i1 @i1.eq(i1 %a, i1 %b) {
  %1 = icmp eq i1 %a, %b
  ret i1 %1
}

define i1 @u64.eq(%u64 %a, %u64 %b) {
  %1 = icmp eq i64 %a, %b
  ret i1 %1
}

define i1 @u32.eq(%u32 %a, %u32 %b) {
  %1 = icmp eq i32 %a, %b
  ret i1 %1
}

define i1 @u16.eq(%u16 %a, %u16 %b) {
  %1 = icmp eq i16 %a, %b
  ret i1 %1
}

define i1 @u8.eq(%u8 %a, %u8 %b) {
  %1 = icmp eq i8 %a, %b
  ret i1 %1
}

define i1 @float.eq(float %a, float %b) {
  %1 = fcmp oeq float %a, %b
  ret i1 %1
}

define i1 @double.eq(double %a, double %b) {
  %1 = fcmp oeq double %a, %b
  ret i1 %1
}

; Template for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)
; - VECSIZE: Size of vectors.

%v0 = type { %u8*, i64 }           ; elements, size
%v0.bld = type i8*

; VecMerger
%v0.vm.bld = type %v0*

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define %v0.vm.bld @v0.vm.bld.getPtrIndexed(%v0.vm.bld %bldPtr, i32 %i) alwaysinline {
  %mergerPtr = getelementptr %v0, %v0* null, i32 1
  %mergerSize = ptrtoint %v0* %mergerPtr to i64
  %asPtr = bitcast %v0.vm.bld %bldPtr to i8*
  %rawPtr = call i8* @weld_rt_get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %v0.vm.bld
  ret %v0.vm.bld %ptr
}

; Initialize and return a new vecmerger with the given initial vector.
define %v0.vm.bld @v0.vm.bld.new(%v0 %vec) {
  %nworkers = call i32 @weld_rt_get_nworkers()
  %structSizePtr = getelementptr %v0, %v0* null, i32 1
  %structSize = ptrtoint %v0* %structSizePtr to i64
  
  %bldPtr = call i8* @weld_rt_new_merger(i64 %structSize, i32 %nworkers)
  %typedPtr = bitcast i8* %bldPtr to %v0.vm.bld
  
  ; Copy the initial value into the first vector
  %first = call %v0.vm.bld @v0.vm.bld.getPtrIndexed(%v0.vm.bld %typedPtr, i32 0)
  %cloned = call %v0 @v0.clone(%v0 %vec)
  %capacity = call i64 @v0.size(%v0 %vec)
  store %v0 %cloned, %v0.vm.bld %first
  br label %entry
  
entry:
  %cond = icmp ult i32 1, %nworkers
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i32 [ 1, %entry ], [ %i2, %body ]
  %vecPtr = call %v0* @v0.vm.bld.getPtrIndexed(%v0.vm.bld %typedPtr, i32 %i)
  %newVec = call %v0 @v0.new(i64 %capacity)
  call void @v0.zero(%v0 %newVec)
  store %v0 %newVec, %v0* %vecPtr
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done
  
done:
  ret %v0.vm.bld %typedPtr
}

; Returns a pointer to the value an element should be merged into.
; The caller should perform the merge operation on the contents of this pointer
; and then store the resulting value back.
define i8* @v0.vm.bld.merge_ptr(%v0.vm.bld %bldPtr, i64 %index, i32 %workerId) {
  %bldPtrLocal = call %v0* @v0.vm.bld.getPtrIndexed(%v0.vm.bld %bldPtr, i32 %workerId)
  %vec = load %v0, %v0* %bldPtrLocal
  %elem = call %u8* @v0.at(%v0 %vec, i64 %index)
  %elemPtrRaw = bitcast %u8* %elem to i8*
  ret i8* %elemPtrRaw
}

; Initialize and return a new vector with the given size.
define %v0 @v0.new(i64 %size) {
  %elemSizePtr = getelementptr %u8, %u8* null, i32 1
  %elemSize = ptrtoint %u8* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %runId = call i64 @weld_rt_get_run_id()
  %bytes = call i8* @weld_run_malloc(i64 %runId, i64 %allocSize)
  %elements = bitcast i8* %bytes to %u8*
  %1 = insertvalue %v0 undef, %u8* %elements, 0
  %2 = insertvalue %v0 %1, i64 %size, 1
  ret %v0 %2
}

; Zeroes a vector's underlying buffer.
define void @v0.zero(%v0 %v) {
  %elements = extractvalue %v0 %v, 0
  %size = extractvalue %v0 %v, 1
  %bytes = bitcast %u8* %elements to i8*
  
  %elemSizePtr = getelementptr %u8, %u8* null, i32 1
  %elemSize = ptrtoint %u8* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  ret void
}

; Clone a vector.
define %v0 @v0.clone(%v0 %vec) {
  %elements = extractvalue %v0 %vec, 0
  %size = extractvalue %v0 %vec, 1
  %entrySizePtr = getelementptr %u8, %u8* null, i32 1
  %entrySize = ptrtoint %u8* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %size
  %bytes = bitcast %u8* %elements to i8*
  %vec2 = call %v0 @v0.new(i64 %size)
  %elements2 = extractvalue %v0 %vec2, 0
  %bytes2 = bitcast %u8* %elements2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  ret %v0 %vec2
}

; Get a new vec object that starts at the index'th element of the existing vector, and has size size.
; If the specified size is greater than the remaining size, then the remaining size is used.
define %v0 @v0.slice(%v0 %vec, i64 %index, i64 %size) {
  ; Check if size greater than remaining size
  %currSize = extractvalue %v0 %vec, 1
  %remSize = sub i64 %currSize, %index
  %sgtr = icmp ugt i64 %size, %remSize
  %finSize = select i1 %sgtr, i64 %remSize, i64 %size
  
  %elements = extractvalue %v0 %vec, 0
  %newElements = getelementptr %u8, %u8* %elements, i64 %index
  %1 = insertvalue %v0 undef, %u8* %newElements, 0
  %2 = insertvalue %v0 %1, i64 %finSize, 1
  
  ret %v0 %2
}

; Initialize and return a new builder, with the given initial capacity.
define %v0.bld @v0.bld.new(i64 %capacity, %work_t* %cur.work, i32 %fixedSize) {
  %elemSizePtr = getelementptr %u8, %u8* null, i32 1
  %elemSize = ptrtoint %u8* %elemSizePtr to i64
  %newVb = call i8* @weld_rt_new_vb(i64 %elemSize, i64 %capacity, i32 %fixedSize)
  call void @v0.bld.newPiece(%v0.bld %newVb, %work_t* %cur.work)
  ret %v0.bld %newVb
}

define void @v0.bld.newPiece(%v0.bld %bldPtr, %work_t* %cur.work) {
  call void @weld_rt_new_vb_piece(i8* %bldPtr, %work_t* %cur.work)
  ret void
}

; Append a value into a builder, growing its space if needed.
define %v0.bld @v0.bld.merge(%v0.bld %bldPtr, %u8 %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %full = icmp eq i64 %size, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr %u8, %u8* null, i32 1
  %elemSize = ptrtoint %u8* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to %u8*
  %insertPtr = getelementptr %u8, %u8* %elements, i64 %size
  store %u8 %value, %u8* %insertPtr
  %newSize = add i64 %size, 1
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v0.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %v0 @v0.bld.result(%v0.bld %bldPtr) {
  %out = call %vb.out @weld_rt_result_vb(i8* %bldPtr)
  %bytes = extractvalue %vb.out %out, 0
  %size = extractvalue %vb.out %out, 1
  %elems = bitcast i8* %bytes to %u8*
  %1 = insertvalue %v0 undef, %u8* %elems, 0
  %2 = insertvalue %v0 %1, i64 %size, 1
  ret %v0 %2
}

; Get the length of a vector.
define i64 @v0.size(%v0 %vec) {
  %size = extractvalue %v0 %vec, 1
  ret i64 %size
}

; Get a pointer to the index'th element.
define %u8* @v0.at(%v0 %vec, i64 %index) {
  %elements = extractvalue %v0 %vec, 0
  %ptr = getelementptr %u8, %u8* %elements, i64 %index
  ret %u8* %ptr
}


; Get the length of a VecBuilder.
define i64 @v0.bld.size(%v0.bld %bldPtr, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  ret i64 %size
}

; Get a pointer to the index'th element of a VecBuilder.
define %u8* @v0.bld.at(%v0.bld %bldPtr, i64 %index, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %bytes = extractvalue %vb.vp %curPiece, 0
  %elements = bitcast i8* %bytes to %u8*
  %ptr = getelementptr %u8, %u8* %elements, i64 %index
  ret %u8* %ptr
}

; Compute the hash code of a vector.
; TODO: We should hash more bytes at a time if elements are non-pointer types.
define i32 @v0.hash(%v0 %vec) {
entry:
  %elements = extractvalue %v0 %vec, 0
  %size = extractvalue %v0 %vec, 1
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %prevHash = phi i32 [ 0, %entry ], [ %newHash, %body ]
  %ptr = getelementptr %u8, %u8* %elements, i64 %i
  %elem = load %u8, %u8* %ptr
  %elemHash = call i32 @u8.hash(%u8 %elem)
  %newHash = call i32 @hash_combine(i32 %prevHash, i32 %elemHash)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
  
done:
  %res = phi i32 [ 0, %entry ], [ %newHash, %body ]
  ret i32 %res
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v0.bld.hash(%v0.bld %bld) {
  ret i32 0
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v0.vm.bld.hash(%v0.vm.bld %bld) {
  ret i32 0
}

; Dummy comparison function; this is needed for structs that use these vecbuilders as fields.
define i32 @v0.bld.cmp(%v0.bld %bld1, %v0.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function; this is needed for structs that use these vecmergers as fields.
define i32 @v0.vm.bld.cmp(%v0.vm.bld %bld1, %v0.vm.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function for builders.
define i1 @v0.bld.eq(%v0.bld %a, %v0.bld %b) {
  ret i1 0
}

; Dummy comparison function for builders.
define i1 @v0.vm.bld.eq(%v0.vm.bld %a, %v0.vm.bld %b) {
  ret i1 0
}

; Vector extensions for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - VECSIZE: Size of vectors.

; Get a pointer to the index'th element, fetching a vector
define <4 x %u8>* @v0.vat(%v0 %vec, i64 %index) {
  %elements = extractvalue %v0 %vec, 0
  %ptr = getelementptr %u8, %u8* %elements, i64 %index
  %retPtr = bitcast %u8* %ptr to <4 x %u8>*
  ret <4 x %u8>* %retPtr
}

; Append a value into a builder, growing its space if needed.
define %v0.bld @v0.bld.vmerge(%v0.bld %bldPtr, <4 x %u8> %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %adjusted = add i64 %size, 4
  %full = icmp sgt i64 %adjusted, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr %u8, %u8* null, i32 1
  %elemSize = ptrtoint %u8* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to %u8*
  %insertPtr = getelementptr %u8, %u8* %elements, i64 %size
  %vecInsertPtr = bitcast %u8* %insertPtr to <4 x %u8>*
  store <4 x %u8> %value, <4 x %u8>* %vecInsertPtr, align 1
  %newSize = add i64 %size, 4
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v0.bld %bldPtr
}

; Vector extension to generate comparison functions using memcmp.
; This is safe for unsigned integer types and booleans.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix

; Compare two vectors of characters lexicographically.
define i32 @v0.cmp(%v0 %a, %v0 %b) {
entry:
  %elemsA = extractvalue %v0 %a, 0
  %elemsB = extractvalue %v0 %b, 0
  %sizeA = extractvalue %v0 %a, 1
  %sizeB = extractvalue %v0 %b, 1
  %cond1 = icmp ult i64 %sizeA, %sizeB
  %minSize = select i1 %cond1, i64 %sizeA, i64 %sizeB
  %cond = icmp ult i64 0, %minSize
  br i1 %cond, label %body, label %done
  
body:
  %cmp = call i32 @memcmp(i8* %elemsA, i8* %elemsB, i64 %minSize)
  %ne = icmp ne i32 %cmp, 0
  br i1 %ne, label %return, label %done
  
return:
  ret i32 %cmp
  
done:
  %res = call i32 @i64.cmp(i64 %sizeA, i64 %sizeB)
  ret i32 %res
}

; Compare two vectors for equality.
define i1 @v0.eq(%v0 %a, %v0 %b) {
  %sizeA = extractvalue %v0 %a, 1
  %sizeB = extractvalue %v0 %b, 1
  %cond = icmp eq i64 %sizeA, %sizeB
  br i1 %cond, label %same_size, label %different_size
same_size:
  %elemsA = extractvalue %v0 %a, 0
  %elemsB = extractvalue %v0 %b, 0
  %cmp = call i32 @memcmp(i8* %elemsA, i8* %elemsB, i64 %sizeA)
  %res = icmp eq i32 %cmp, 0
  ret i1 %res
different_size:
  ret i1 0
}
%s1 = type { i1, %v0 }
define i32 @s1.hash(%s1 %value) {
  %p.p0 = extractvalue %s1 %value, 0
  %p.p1 = call i32 @i1.hash(i1 %p.p0)
  %p.p2 = call i32 @hash_combine(i32 0, i32 %p.p1)
  %p.p3 = extractvalue %s1 %value, 1
  %p.p4 = call i32 @v0.hash(%v0 %p.p3)
  %p.p5 = call i32 @hash_combine(i32 %p.p2, i32 %p.p4)
  ret i32 %p.p5
}

define i32 @s1.cmp(%s1 %a, %s1 %b) {
  %p.p6 = extractvalue %s1 %a , 0
  %p.p7 = extractvalue %s1 %b, 0
  %p.p8 = call i32 @i1.cmp(i1 %p.p6, i1 %p.p7)
  %p.p9 = icmp ne i32 %p.p8, 0
  br i1 %p.p9, label %l0, label %l1
l0:
  ret i32 %p.p8
l1:
  %p.p10 = extractvalue %s1 %a , 1
  %p.p11 = extractvalue %s1 %b, 1
  %p.p12 = call i32 @v0.cmp(%v0 %p.p10, %v0 %p.p11)
  %p.p13 = icmp ne i32 %p.p12, 0
  br i1 %p.p13, label %l2, label %l3
l2:
  ret i32 %p.p12
l3:
  ret i32 0
}

define i1 @s1.eq(%s1 %a, %s1 %b) {
  %p.p14 = extractvalue %s1 %a , 0
  %p.p15 = extractvalue %s1 %b, 0
  %p.p16 = call i1 @i1.eq(i1 %p.p14, i1 %p.p15)
  br i1 %p.p16, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p17 = extractvalue %s1 %a , 1
  %p.p18 = extractvalue %s1 %b, 1
  %p.p19 = call i1 @v0.eq(%v0 %p.p17, %v0 %p.p18)
  br i1 %p.p19, label %l3, label %l2
l2:
  ret i1 0
l3:
  ret i1 1
}

%s2 = type { double, i64 }
define i32 @s2.hash(%s2 %value) {
  %p.p20 = extractvalue %s2 %value, 0
  %p.p21 = call i32 @double.hash(double %p.p20)
  %p.p22 = call i32 @hash_combine(i32 0, i32 %p.p21)
  %p.p23 = extractvalue %s2 %value, 1
  %p.p24 = call i32 @i64.hash(i64 %p.p23)
  %p.p25 = call i32 @hash_combine(i32 %p.p22, i32 %p.p24)
  ret i32 %p.p25
}

define i32 @s2.cmp(%s2 %a, %s2 %b) {
  %p.p26 = extractvalue %s2 %a , 0
  %p.p27 = extractvalue %s2 %b, 0
  %p.p28 = call i32 @double.cmp(double %p.p26, double %p.p27)
  %p.p29 = icmp ne i32 %p.p28, 0
  br i1 %p.p29, label %l0, label %l1
l0:
  ret i32 %p.p28
l1:
  %p.p30 = extractvalue %s2 %a , 1
  %p.p31 = extractvalue %s2 %b, 1
  %p.p32 = call i32 @i64.cmp(i64 %p.p30, i64 %p.p31)
  %p.p33 = icmp ne i32 %p.p32, 0
  br i1 %p.p33, label %l2, label %l3
l2:
  ret i32 %p.p32
l3:
  ret i32 0
}

define i1 @s2.eq(%s2 %a, %s2 %b) {
  %p.p34 = extractvalue %s2 %a , 0
  %p.p35 = extractvalue %s2 %b, 0
  %p.p36 = call i1 @double.eq(double %p.p34, double %p.p35)
  br i1 %p.p36, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p37 = extractvalue %s2 %a , 1
  %p.p38 = extractvalue %s2 %b, 1
  %p.p39 = call i1 @i64.eq(i64 %p.p37, i64 %p.p38)
  br i1 %p.p39, label %l3, label %l2
l2:
  ret i1 0
l3:
  ret i1 1
}

%s0 = type { %s1, %s2 }
define i32 @s0.hash(%s0 %value) {
  %p.p40 = extractvalue %s0 %value, 0
  %p.p41 = call i32 @s1.hash(%s1 %p.p40)
  %p.p42 = call i32 @hash_combine(i32 0, i32 %p.p41)
  %p.p43 = extractvalue %s0 %value, 1
  %p.p44 = call i32 @s2.hash(%s2 %p.p43)
  %p.p45 = call i32 @hash_combine(i32 %p.p42, i32 %p.p44)
  ret i32 %p.p45
}

define i32 @s0.cmp(%s0 %a, %s0 %b) {
  %p.p46 = extractvalue %s0 %a , 0
  %p.p47 = extractvalue %s0 %b, 0
  %p.p48 = call i32 @s1.cmp(%s1 %p.p46, %s1 %p.p47)
  %p.p49 = icmp ne i32 %p.p48, 0
  br i1 %p.p49, label %l0, label %l1
l0:
  ret i32 %p.p48
l1:
  %p.p50 = extractvalue %s0 %a , 1
  %p.p51 = extractvalue %s0 %b, 1
  %p.p52 = call i32 @s2.cmp(%s2 %p.p50, %s2 %p.p51)
  %p.p53 = icmp ne i32 %p.p52, 0
  br i1 %p.p53, label %l2, label %l3
l2:
  ret i32 %p.p52
l3:
  ret i32 0
}

define i1 @s0.eq(%s0 %a, %s0 %b) {
  %p.p54 = extractvalue %s0 %a , 0
  %p.p55 = extractvalue %s0 %b, 0
  %p.p56 = call i1 @s1.eq(%s1 %p.p54, %s1 %p.p55)
  br i1 %p.p56, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p57 = extractvalue %s0 %a , 1
  %p.p58 = extractvalue %s0 %b, 1
  %p.p59 = call i1 @s2.eq(%s2 %p.p57, %s2 %p.p58)
  br i1 %p.p59, label %l3, label %l2
l2:
  ret i1 0
l3:
  ret i1 1
}

; Template for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)
; - VECSIZE: Size of vectors.

%v1 = type { %s0*, i64 }           ; elements, size
%v1.bld = type i8*

; VecMerger
%v1.vm.bld = type %v1*

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define %v1.vm.bld @v1.vm.bld.getPtrIndexed(%v1.vm.bld %bldPtr, i32 %i) alwaysinline {
  %mergerPtr = getelementptr %v1, %v1* null, i32 1
  %mergerSize = ptrtoint %v1* %mergerPtr to i64
  %asPtr = bitcast %v1.vm.bld %bldPtr to i8*
  %rawPtr = call i8* @weld_rt_get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %v1.vm.bld
  ret %v1.vm.bld %ptr
}

; Initialize and return a new vecmerger with the given initial vector.
define %v1.vm.bld @v1.vm.bld.new(%v1 %vec) {
  %nworkers = call i32 @weld_rt_get_nworkers()
  %structSizePtr = getelementptr %v1, %v1* null, i32 1
  %structSize = ptrtoint %v1* %structSizePtr to i64
  
  %bldPtr = call i8* @weld_rt_new_merger(i64 %structSize, i32 %nworkers)
  %typedPtr = bitcast i8* %bldPtr to %v1.vm.bld
  
  ; Copy the initial value into the first vector
  %first = call %v1.vm.bld @v1.vm.bld.getPtrIndexed(%v1.vm.bld %typedPtr, i32 0)
  %cloned = call %v1 @v1.clone(%v1 %vec)
  %capacity = call i64 @v1.size(%v1 %vec)
  store %v1 %cloned, %v1.vm.bld %first
  br label %entry
  
entry:
  %cond = icmp ult i32 1, %nworkers
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i32 [ 1, %entry ], [ %i2, %body ]
  %vecPtr = call %v1* @v1.vm.bld.getPtrIndexed(%v1.vm.bld %typedPtr, i32 %i)
  %newVec = call %v1 @v1.new(i64 %capacity)
  call void @v1.zero(%v1 %newVec)
  store %v1 %newVec, %v1* %vecPtr
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done
  
done:
  ret %v1.vm.bld %typedPtr
}

; Returns a pointer to the value an element should be merged into.
; The caller should perform the merge operation on the contents of this pointer
; and then store the resulting value back.
define i8* @v1.vm.bld.merge_ptr(%v1.vm.bld %bldPtr, i64 %index, i32 %workerId) {
  %bldPtrLocal = call %v1* @v1.vm.bld.getPtrIndexed(%v1.vm.bld %bldPtr, i32 %workerId)
  %vec = load %v1, %v1* %bldPtrLocal
  %elem = call %s0* @v1.at(%v1 %vec, i64 %index)
  %elemPtrRaw = bitcast %s0* %elem to i8*
  ret i8* %elemPtrRaw
}

; Initialize and return a new vector with the given size.
define %v1 @v1.new(i64 %size) {
  %elemSizePtr = getelementptr %s0, %s0* null, i32 1
  %elemSize = ptrtoint %s0* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %runId = call i64 @weld_rt_get_run_id()
  %bytes = call i8* @weld_run_malloc(i64 %runId, i64 %allocSize)
  %elements = bitcast i8* %bytes to %s0*
  %1 = insertvalue %v1 undef, %s0* %elements, 0
  %2 = insertvalue %v1 %1, i64 %size, 1
  ret %v1 %2
}

; Zeroes a vector's underlying buffer.
define void @v1.zero(%v1 %v) {
  %elements = extractvalue %v1 %v, 0
  %size = extractvalue %v1 %v, 1
  %bytes = bitcast %s0* %elements to i8*
  
  %elemSizePtr = getelementptr %s0, %s0* null, i32 1
  %elemSize = ptrtoint %s0* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  ret void
}

; Clone a vector.
define %v1 @v1.clone(%v1 %vec) {
  %elements = extractvalue %v1 %vec, 0
  %size = extractvalue %v1 %vec, 1
  %entrySizePtr = getelementptr %s0, %s0* null, i32 1
  %entrySize = ptrtoint %s0* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %size
  %bytes = bitcast %s0* %elements to i8*
  %vec2 = call %v1 @v1.new(i64 %size)
  %elements2 = extractvalue %v1 %vec2, 0
  %bytes2 = bitcast %s0* %elements2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  ret %v1 %vec2
}

; Get a new vec object that starts at the index'th element of the existing vector, and has size size.
; If the specified size is greater than the remaining size, then the remaining size is used.
define %v1 @v1.slice(%v1 %vec, i64 %index, i64 %size) {
  ; Check if size greater than remaining size
  %currSize = extractvalue %v1 %vec, 1
  %remSize = sub i64 %currSize, %index
  %sgtr = icmp ugt i64 %size, %remSize
  %finSize = select i1 %sgtr, i64 %remSize, i64 %size
  
  %elements = extractvalue %v1 %vec, 0
  %newElements = getelementptr %s0, %s0* %elements, i64 %index
  %1 = insertvalue %v1 undef, %s0* %newElements, 0
  %2 = insertvalue %v1 %1, i64 %finSize, 1
  
  ret %v1 %2
}

; Initialize and return a new builder, with the given initial capacity.
define %v1.bld @v1.bld.new(i64 %capacity, %work_t* %cur.work, i32 %fixedSize) {
  %elemSizePtr = getelementptr %s0, %s0* null, i32 1
  %elemSize = ptrtoint %s0* %elemSizePtr to i64
  %newVb = call i8* @weld_rt_new_vb(i64 %elemSize, i64 %capacity, i32 %fixedSize)
  call void @v1.bld.newPiece(%v1.bld %newVb, %work_t* %cur.work)
  ret %v1.bld %newVb
}

define void @v1.bld.newPiece(%v1.bld %bldPtr, %work_t* %cur.work) {
  call void @weld_rt_new_vb_piece(i8* %bldPtr, %work_t* %cur.work)
  ret void
}

; Append a value into a builder, growing its space if needed.
define %v1.bld @v1.bld.merge(%v1.bld %bldPtr, %s0 %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %full = icmp eq i64 %size, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr %s0, %s0* null, i32 1
  %elemSize = ptrtoint %s0* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to %s0*
  %insertPtr = getelementptr %s0, %s0* %elements, i64 %size
  store %s0 %value, %s0* %insertPtr
  %newSize = add i64 %size, 1
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v1.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %v1 @v1.bld.result(%v1.bld %bldPtr) {
  %out = call %vb.out @weld_rt_result_vb(i8* %bldPtr)
  %bytes = extractvalue %vb.out %out, 0
  %size = extractvalue %vb.out %out, 1
  %elems = bitcast i8* %bytes to %s0*
  %1 = insertvalue %v1 undef, %s0* %elems, 0
  %2 = insertvalue %v1 %1, i64 %size, 1
  ret %v1 %2
}

; Get the length of a vector.
define i64 @v1.size(%v1 %vec) {
  %size = extractvalue %v1 %vec, 1
  ret i64 %size
}

; Get a pointer to the index'th element.
define %s0* @v1.at(%v1 %vec, i64 %index) {
  %elements = extractvalue %v1 %vec, 0
  %ptr = getelementptr %s0, %s0* %elements, i64 %index
  ret %s0* %ptr
}


; Get the length of a VecBuilder.
define i64 @v1.bld.size(%v1.bld %bldPtr, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  ret i64 %size
}

; Get a pointer to the index'th element of a VecBuilder.
define %s0* @v1.bld.at(%v1.bld %bldPtr, i64 %index, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %bytes = extractvalue %vb.vp %curPiece, 0
  %elements = bitcast i8* %bytes to %s0*
  %ptr = getelementptr %s0, %s0* %elements, i64 %index
  ret %s0* %ptr
}

; Compute the hash code of a vector.
; TODO: We should hash more bytes at a time if elements are non-pointer types.
define i32 @v1.hash(%v1 %vec) {
entry:
  %elements = extractvalue %v1 %vec, 0
  %size = extractvalue %v1 %vec, 1
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %prevHash = phi i32 [ 0, %entry ], [ %newHash, %body ]
  %ptr = getelementptr %s0, %s0* %elements, i64 %i
  %elem = load %s0, %s0* %ptr
  %elemHash = call i32 @s0.hash(%s0 %elem)
  %newHash = call i32 @hash_combine(i32 %prevHash, i32 %elemHash)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
  
done:
  %res = phi i32 [ 0, %entry ], [ %newHash, %body ]
  ret i32 %res
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v1.bld.hash(%v1.bld %bld) {
  ret i32 0
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v1.vm.bld.hash(%v1.vm.bld %bld) {
  ret i32 0
}

; Dummy comparison function; this is needed for structs that use these vecbuilders as fields.
define i32 @v1.bld.cmp(%v1.bld %bld1, %v1.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function; this is needed for structs that use these vecmergers as fields.
define i32 @v1.vm.bld.cmp(%v1.vm.bld %bld1, %v1.vm.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function for builders.
define i1 @v1.bld.eq(%v1.bld %a, %v1.bld %b) {
  ret i1 0
}

; Dummy comparison function for builders.
define i1 @v1.vm.bld.eq(%v1.vm.bld %a, %v1.vm.bld %b) {
  ret i1 0
}

; Vector extension to generate comparison functions. We use a different file,
; vector_comparisons_i8.ll, for comparisons on arrays of bytes, which are optimized.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)



; Compare two vectors lexicographically.
define i32 @v1.cmp(%v1 %a, %v1 %b) {
entry:
  %elemsA = extractvalue %v1 %a, 0
  %elemsB = extractvalue %v1 %b, 0
  %sizeA = extractvalue %v1 %a, 1
  %sizeB = extractvalue %v1 %b, 1
  %cond1 = icmp ult i64 %sizeA, %sizeB
  %minSize = select i1 %cond1, i64 %sizeA, i64 %sizeB
  %cond = icmp ult i64 0, %minSize
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body2 ]
  %ptrA = getelementptr %s0, %s0* %elemsA, i64 %i
  %ptrB = getelementptr %s0, %s0* %elemsB, i64 %i
  %elemA = load %s0, %s0* %ptrA
  %elemB = load %s0, %s0* %ptrB
  %cmp = call i32 @s0.cmp(%s0 %elemA, %s0 %elemB)
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

; Compare two vectors for equality.
define i1 @v1.eq(%v1 %a, %v1 %b) {
  %sizeA = extractvalue %v1 %a, 1
  %sizeB = extractvalue %v1 %b, 1
  %cond = icmp eq i64 %sizeA, %sizeB
  br i1 %cond, label %same_size, label %different_size
same_size:
  %cmp = call i32 @v1.cmp(%v1 %a, %v1 %b)
  %res = icmp eq i32 %cmp, 0
  ret i1 %res
different_size:
  ret i1 0
}
%s3 = type { i1, %v0, i1, double, i1, i64 }
define i32 @s3.hash(%s3 %value) {
  %p.p60 = extractvalue %s3 %value, 0
  %p.p61 = call i32 @i1.hash(i1 %p.p60)
  %p.p62 = call i32 @hash_combine(i32 0, i32 %p.p61)
  %p.p63 = extractvalue %s3 %value, 1
  %p.p64 = call i32 @v0.hash(%v0 %p.p63)
  %p.p65 = call i32 @hash_combine(i32 %p.p62, i32 %p.p64)
  %p.p66 = extractvalue %s3 %value, 2
  %p.p67 = call i32 @i1.hash(i1 %p.p66)
  %p.p68 = call i32 @hash_combine(i32 %p.p65, i32 %p.p67)
  %p.p69 = extractvalue %s3 %value, 3
  %p.p70 = call i32 @double.hash(double %p.p69)
  %p.p71 = call i32 @hash_combine(i32 %p.p68, i32 %p.p70)
  %p.p72 = extractvalue %s3 %value, 4
  %p.p73 = call i32 @i1.hash(i1 %p.p72)
  %p.p74 = call i32 @hash_combine(i32 %p.p71, i32 %p.p73)
  %p.p75 = extractvalue %s3 %value, 5
  %p.p76 = call i32 @i64.hash(i64 %p.p75)
  %p.p77 = call i32 @hash_combine(i32 %p.p74, i32 %p.p76)
  ret i32 %p.p77
}

define i32 @s3.cmp(%s3 %a, %s3 %b) {
  %p.p78 = extractvalue %s3 %a , 0
  %p.p79 = extractvalue %s3 %b, 0
  %p.p80 = call i32 @i1.cmp(i1 %p.p78, i1 %p.p79)
  %p.p81 = icmp ne i32 %p.p80, 0
  br i1 %p.p81, label %l0, label %l1
l0:
  ret i32 %p.p80
l1:
  %p.p82 = extractvalue %s3 %a , 1
  %p.p83 = extractvalue %s3 %b, 1
  %p.p84 = call i32 @v0.cmp(%v0 %p.p82, %v0 %p.p83)
  %p.p85 = icmp ne i32 %p.p84, 0
  br i1 %p.p85, label %l2, label %l3
l2:
  ret i32 %p.p84
l3:
  %p.p86 = extractvalue %s3 %a , 2
  %p.p87 = extractvalue %s3 %b, 2
  %p.p88 = call i32 @i1.cmp(i1 %p.p86, i1 %p.p87)
  %p.p89 = icmp ne i32 %p.p88, 0
  br i1 %p.p89, label %l4, label %l5
l4:
  ret i32 %p.p88
l5:
  %p.p90 = extractvalue %s3 %a , 3
  %p.p91 = extractvalue %s3 %b, 3
  %p.p92 = call i32 @double.cmp(double %p.p90, double %p.p91)
  %p.p93 = icmp ne i32 %p.p92, 0
  br i1 %p.p93, label %l6, label %l7
l6:
  ret i32 %p.p92
l7:
  %p.p94 = extractvalue %s3 %a , 4
  %p.p95 = extractvalue %s3 %b, 4
  %p.p96 = call i32 @i1.cmp(i1 %p.p94, i1 %p.p95)
  %p.p97 = icmp ne i32 %p.p96, 0
  br i1 %p.p97, label %l8, label %l9
l8:
  ret i32 %p.p96
l9:
  %p.p98 = extractvalue %s3 %a , 5
  %p.p99 = extractvalue %s3 %b, 5
  %p.p100 = call i32 @i64.cmp(i64 %p.p98, i64 %p.p99)
  %p.p101 = icmp ne i32 %p.p100, 0
  br i1 %p.p101, label %l10, label %l11
l10:
  ret i32 %p.p100
l11:
  ret i32 0
}

define i1 @s3.eq(%s3 %a, %s3 %b) {
  %p.p102 = extractvalue %s3 %a , 0
  %p.p103 = extractvalue %s3 %b, 0
  %p.p104 = call i1 @i1.eq(i1 %p.p102, i1 %p.p103)
  br i1 %p.p104, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p105 = extractvalue %s3 %a , 1
  %p.p106 = extractvalue %s3 %b, 1
  %p.p107 = call i1 @v0.eq(%v0 %p.p105, %v0 %p.p106)
  br i1 %p.p107, label %l3, label %l2
l2:
  ret i1 0
l3:
  %p.p108 = extractvalue %s3 %a , 2
  %p.p109 = extractvalue %s3 %b, 2
  %p.p110 = call i1 @i1.eq(i1 %p.p108, i1 %p.p109)
  br i1 %p.p110, label %l5, label %l4
l4:
  ret i1 0
l5:
  %p.p111 = extractvalue %s3 %a , 3
  %p.p112 = extractvalue %s3 %b, 3
  %p.p113 = call i1 @double.eq(double %p.p111, double %p.p112)
  br i1 %p.p113, label %l7, label %l6
l6:
  ret i1 0
l7:
  %p.p114 = extractvalue %s3 %a , 4
  %p.p115 = extractvalue %s3 %b, 4
  %p.p116 = call i1 @i1.eq(i1 %p.p114, i1 %p.p115)
  br i1 %p.p116, label %l9, label %l8
l8:
  ret i1 0
l9:
  %p.p117 = extractvalue %s3 %a , 5
  %p.p118 = extractvalue %s3 %b, 5
  %p.p119 = call i1 @i64.eq(i64 %p.p117, i64 %p.p118)
  br i1 %p.p119, label %l11, label %l10
l10:
  ret i1 0
l11:
  ret i1 1
}

; Template for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)
; - VECSIZE: Size of vectors.

%v2 = type { %s3*, i64 }           ; elements, size
%v2.bld = type i8*

; VecMerger
%v2.vm.bld = type %v2*

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define %v2.vm.bld @v2.vm.bld.getPtrIndexed(%v2.vm.bld %bldPtr, i32 %i) alwaysinline {
  %mergerPtr = getelementptr %v2, %v2* null, i32 1
  %mergerSize = ptrtoint %v2* %mergerPtr to i64
  %asPtr = bitcast %v2.vm.bld %bldPtr to i8*
  %rawPtr = call i8* @weld_rt_get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %v2.vm.bld
  ret %v2.vm.bld %ptr
}

; Initialize and return a new vecmerger with the given initial vector.
define %v2.vm.bld @v2.vm.bld.new(%v2 %vec) {
  %nworkers = call i32 @weld_rt_get_nworkers()
  %structSizePtr = getelementptr %v2, %v2* null, i32 1
  %structSize = ptrtoint %v2* %structSizePtr to i64
  
  %bldPtr = call i8* @weld_rt_new_merger(i64 %structSize, i32 %nworkers)
  %typedPtr = bitcast i8* %bldPtr to %v2.vm.bld
  
  ; Copy the initial value into the first vector
  %first = call %v2.vm.bld @v2.vm.bld.getPtrIndexed(%v2.vm.bld %typedPtr, i32 0)
  %cloned = call %v2 @v2.clone(%v2 %vec)
  %capacity = call i64 @v2.size(%v2 %vec)
  store %v2 %cloned, %v2.vm.bld %first
  br label %entry
  
entry:
  %cond = icmp ult i32 1, %nworkers
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i32 [ 1, %entry ], [ %i2, %body ]
  %vecPtr = call %v2* @v2.vm.bld.getPtrIndexed(%v2.vm.bld %typedPtr, i32 %i)
  %newVec = call %v2 @v2.new(i64 %capacity)
  call void @v2.zero(%v2 %newVec)
  store %v2 %newVec, %v2* %vecPtr
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done
  
done:
  ret %v2.vm.bld %typedPtr
}

; Returns a pointer to the value an element should be merged into.
; The caller should perform the merge operation on the contents of this pointer
; and then store the resulting value back.
define i8* @v2.vm.bld.merge_ptr(%v2.vm.bld %bldPtr, i64 %index, i32 %workerId) {
  %bldPtrLocal = call %v2* @v2.vm.bld.getPtrIndexed(%v2.vm.bld %bldPtr, i32 %workerId)
  %vec = load %v2, %v2* %bldPtrLocal
  %elem = call %s3* @v2.at(%v2 %vec, i64 %index)
  %elemPtrRaw = bitcast %s3* %elem to i8*
  ret i8* %elemPtrRaw
}

; Initialize and return a new vector with the given size.
define %v2 @v2.new(i64 %size) {
  %elemSizePtr = getelementptr %s3, %s3* null, i32 1
  %elemSize = ptrtoint %s3* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %runId = call i64 @weld_rt_get_run_id()
  %bytes = call i8* @weld_run_malloc(i64 %runId, i64 %allocSize)
  %elements = bitcast i8* %bytes to %s3*
  %1 = insertvalue %v2 undef, %s3* %elements, 0
  %2 = insertvalue %v2 %1, i64 %size, 1
  ret %v2 %2
}

; Zeroes a vector's underlying buffer.
define void @v2.zero(%v2 %v) {
  %elements = extractvalue %v2 %v, 0
  %size = extractvalue %v2 %v, 1
  %bytes = bitcast %s3* %elements to i8*
  
  %elemSizePtr = getelementptr %s3, %s3* null, i32 1
  %elemSize = ptrtoint %s3* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  ret void
}

; Clone a vector.
define %v2 @v2.clone(%v2 %vec) {
  %elements = extractvalue %v2 %vec, 0
  %size = extractvalue %v2 %vec, 1
  %entrySizePtr = getelementptr %s3, %s3* null, i32 1
  %entrySize = ptrtoint %s3* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %size
  %bytes = bitcast %s3* %elements to i8*
  %vec2 = call %v2 @v2.new(i64 %size)
  %elements2 = extractvalue %v2 %vec2, 0
  %bytes2 = bitcast %s3* %elements2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  ret %v2 %vec2
}

; Get a new vec object that starts at the index'th element of the existing vector, and has size size.
; If the specified size is greater than the remaining size, then the remaining size is used.
define %v2 @v2.slice(%v2 %vec, i64 %index, i64 %size) {
  ; Check if size greater than remaining size
  %currSize = extractvalue %v2 %vec, 1
  %remSize = sub i64 %currSize, %index
  %sgtr = icmp ugt i64 %size, %remSize
  %finSize = select i1 %sgtr, i64 %remSize, i64 %size
  
  %elements = extractvalue %v2 %vec, 0
  %newElements = getelementptr %s3, %s3* %elements, i64 %index
  %1 = insertvalue %v2 undef, %s3* %newElements, 0
  %2 = insertvalue %v2 %1, i64 %finSize, 1
  
  ret %v2 %2
}

; Initialize and return a new builder, with the given initial capacity.
define %v2.bld @v2.bld.new(i64 %capacity, %work_t* %cur.work, i32 %fixedSize) {
  %elemSizePtr = getelementptr %s3, %s3* null, i32 1
  %elemSize = ptrtoint %s3* %elemSizePtr to i64
  %newVb = call i8* @weld_rt_new_vb(i64 %elemSize, i64 %capacity, i32 %fixedSize)
  call void @v2.bld.newPiece(%v2.bld %newVb, %work_t* %cur.work)
  ret %v2.bld %newVb
}

define void @v2.bld.newPiece(%v2.bld %bldPtr, %work_t* %cur.work) {
  call void @weld_rt_new_vb_piece(i8* %bldPtr, %work_t* %cur.work)
  ret void
}

; Append a value into a builder, growing its space if needed.
define %v2.bld @v2.bld.merge(%v2.bld %bldPtr, %s3 %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %full = icmp eq i64 %size, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr %s3, %s3* null, i32 1
  %elemSize = ptrtoint %s3* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to %s3*
  %insertPtr = getelementptr %s3, %s3* %elements, i64 %size
  store %s3 %value, %s3* %insertPtr
  %newSize = add i64 %size, 1
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v2.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %v2 @v2.bld.result(%v2.bld %bldPtr) {
  %out = call %vb.out @weld_rt_result_vb(i8* %bldPtr)
  %bytes = extractvalue %vb.out %out, 0
  %size = extractvalue %vb.out %out, 1
  %elems = bitcast i8* %bytes to %s3*
  %1 = insertvalue %v2 undef, %s3* %elems, 0
  %2 = insertvalue %v2 %1, i64 %size, 1
  ret %v2 %2
}

; Get the length of a vector.
define i64 @v2.size(%v2 %vec) {
  %size = extractvalue %v2 %vec, 1
  ret i64 %size
}

; Get a pointer to the index'th element.
define %s3* @v2.at(%v2 %vec, i64 %index) {
  %elements = extractvalue %v2 %vec, 0
  %ptr = getelementptr %s3, %s3* %elements, i64 %index
  ret %s3* %ptr
}


; Get the length of a VecBuilder.
define i64 @v2.bld.size(%v2.bld %bldPtr, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  ret i64 %size
}

; Get a pointer to the index'th element of a VecBuilder.
define %s3* @v2.bld.at(%v2.bld %bldPtr, i64 %index, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %bytes = extractvalue %vb.vp %curPiece, 0
  %elements = bitcast i8* %bytes to %s3*
  %ptr = getelementptr %s3, %s3* %elements, i64 %index
  ret %s3* %ptr
}

; Compute the hash code of a vector.
; TODO: We should hash more bytes at a time if elements are non-pointer types.
define i32 @v2.hash(%v2 %vec) {
entry:
  %elements = extractvalue %v2 %vec, 0
  %size = extractvalue %v2 %vec, 1
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %prevHash = phi i32 [ 0, %entry ], [ %newHash, %body ]
  %ptr = getelementptr %s3, %s3* %elements, i64 %i
  %elem = load %s3, %s3* %ptr
  %elemHash = call i32 @s3.hash(%s3 %elem)
  %newHash = call i32 @hash_combine(i32 %prevHash, i32 %elemHash)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
  
done:
  %res = phi i32 [ 0, %entry ], [ %newHash, %body ]
  ret i32 %res
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v2.bld.hash(%v2.bld %bld) {
  ret i32 0
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v2.vm.bld.hash(%v2.vm.bld %bld) {
  ret i32 0
}

; Dummy comparison function; this is needed for structs that use these vecbuilders as fields.
define i32 @v2.bld.cmp(%v2.bld %bld1, %v2.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function; this is needed for structs that use these vecmergers as fields.
define i32 @v2.vm.bld.cmp(%v2.vm.bld %bld1, %v2.vm.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function for builders.
define i1 @v2.bld.eq(%v2.bld %a, %v2.bld %b) {
  ret i1 0
}

; Dummy comparison function for builders.
define i1 @v2.vm.bld.eq(%v2.vm.bld %a, %v2.vm.bld %b) {
  ret i1 0
}

; Vector extension to generate comparison functions. We use a different file,
; vector_comparisons_i8.ll, for comparisons on arrays of bytes, which are optimized.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)



; Compare two vectors lexicographically.
define i32 @v2.cmp(%v2 %a, %v2 %b) {
entry:
  %elemsA = extractvalue %v2 %a, 0
  %elemsB = extractvalue %v2 %b, 0
  %sizeA = extractvalue %v2 %a, 1
  %sizeB = extractvalue %v2 %b, 1
  %cond1 = icmp ult i64 %sizeA, %sizeB
  %minSize = select i1 %cond1, i64 %sizeA, i64 %sizeB
  %cond = icmp ult i64 0, %minSize
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body2 ]
  %ptrA = getelementptr %s3, %s3* %elemsA, i64 %i
  %ptrB = getelementptr %s3, %s3* %elemsB, i64 %i
  %elemA = load %s3, %s3* %ptrA
  %elemB = load %s3, %s3* %ptrB
  %cmp = call i32 @s3.cmp(%s3 %elemA, %s3 %elemB)
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

; Compare two vectors for equality.
define i1 @v2.eq(%v2 %a, %v2 %b) {
  %sizeA = extractvalue %v2 %a, 1
  %sizeB = extractvalue %v2 %b, 1
  %cond = icmp eq i64 %sizeA, %sizeB
  br i1 %cond, label %same_size, label %different_size
same_size:
  %cmp = call i32 @v2.cmp(%v2 %a, %v2 %b)
  %res = icmp eq i32 %cmp, 0
  ret i1 %res
different_size:
  ret i1 0
}
; Template for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)
; - VECSIZE: Size of vectors.

%v3 = type { i1*, i64 }           ; elements, size
%v3.bld = type i8*

; VecMerger
%v3.vm.bld = type %v3*

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define %v3.vm.bld @v3.vm.bld.getPtrIndexed(%v3.vm.bld %bldPtr, i32 %i) alwaysinline {
  %mergerPtr = getelementptr %v3, %v3* null, i32 1
  %mergerSize = ptrtoint %v3* %mergerPtr to i64
  %asPtr = bitcast %v3.vm.bld %bldPtr to i8*
  %rawPtr = call i8* @weld_rt_get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %v3.vm.bld
  ret %v3.vm.bld %ptr
}

; Initialize and return a new vecmerger with the given initial vector.
define %v3.vm.bld @v3.vm.bld.new(%v3 %vec) {
  %nworkers = call i32 @weld_rt_get_nworkers()
  %structSizePtr = getelementptr %v3, %v3* null, i32 1
  %structSize = ptrtoint %v3* %structSizePtr to i64
  
  %bldPtr = call i8* @weld_rt_new_merger(i64 %structSize, i32 %nworkers)
  %typedPtr = bitcast i8* %bldPtr to %v3.vm.bld
  
  ; Copy the initial value into the first vector
  %first = call %v3.vm.bld @v3.vm.bld.getPtrIndexed(%v3.vm.bld %typedPtr, i32 0)
  %cloned = call %v3 @v3.clone(%v3 %vec)
  %capacity = call i64 @v3.size(%v3 %vec)
  store %v3 %cloned, %v3.vm.bld %first
  br label %entry
  
entry:
  %cond = icmp ult i32 1, %nworkers
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i32 [ 1, %entry ], [ %i2, %body ]
  %vecPtr = call %v3* @v3.vm.bld.getPtrIndexed(%v3.vm.bld %typedPtr, i32 %i)
  %newVec = call %v3 @v3.new(i64 %capacity)
  call void @v3.zero(%v3 %newVec)
  store %v3 %newVec, %v3* %vecPtr
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done
  
done:
  ret %v3.vm.bld %typedPtr
}

; Returns a pointer to the value an element should be merged into.
; The caller should perform the merge operation on the contents of this pointer
; and then store the resulting value back.
define i8* @v3.vm.bld.merge_ptr(%v3.vm.bld %bldPtr, i64 %index, i32 %workerId) {
  %bldPtrLocal = call %v3* @v3.vm.bld.getPtrIndexed(%v3.vm.bld %bldPtr, i32 %workerId)
  %vec = load %v3, %v3* %bldPtrLocal
  %elem = call i1* @v3.at(%v3 %vec, i64 %index)
  %elemPtrRaw = bitcast i1* %elem to i8*
  ret i8* %elemPtrRaw
}

; Initialize and return a new vector with the given size.
define %v3 @v3.new(i64 %size) {
  %elemSizePtr = getelementptr i1, i1* null, i32 1
  %elemSize = ptrtoint i1* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %runId = call i64 @weld_rt_get_run_id()
  %bytes = call i8* @weld_run_malloc(i64 %runId, i64 %allocSize)
  %elements = bitcast i8* %bytes to i1*
  %1 = insertvalue %v3 undef, i1* %elements, 0
  %2 = insertvalue %v3 %1, i64 %size, 1
  ret %v3 %2
}

; Zeroes a vector's underlying buffer.
define void @v3.zero(%v3 %v) {
  %elements = extractvalue %v3 %v, 0
  %size = extractvalue %v3 %v, 1
  %bytes = bitcast i1* %elements to i8*
  
  %elemSizePtr = getelementptr i1, i1* null, i32 1
  %elemSize = ptrtoint i1* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  ret void
}

; Clone a vector.
define %v3 @v3.clone(%v3 %vec) {
  %elements = extractvalue %v3 %vec, 0
  %size = extractvalue %v3 %vec, 1
  %entrySizePtr = getelementptr i1, i1* null, i32 1
  %entrySize = ptrtoint i1* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %size
  %bytes = bitcast i1* %elements to i8*
  %vec2 = call %v3 @v3.new(i64 %size)
  %elements2 = extractvalue %v3 %vec2, 0
  %bytes2 = bitcast i1* %elements2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  ret %v3 %vec2
}

; Get a new vec object that starts at the index'th element of the existing vector, and has size size.
; If the specified size is greater than the remaining size, then the remaining size is used.
define %v3 @v3.slice(%v3 %vec, i64 %index, i64 %size) {
  ; Check if size greater than remaining size
  %currSize = extractvalue %v3 %vec, 1
  %remSize = sub i64 %currSize, %index
  %sgtr = icmp ugt i64 %size, %remSize
  %finSize = select i1 %sgtr, i64 %remSize, i64 %size
  
  %elements = extractvalue %v3 %vec, 0
  %newElements = getelementptr i1, i1* %elements, i64 %index
  %1 = insertvalue %v3 undef, i1* %newElements, 0
  %2 = insertvalue %v3 %1, i64 %finSize, 1
  
  ret %v3 %2
}

; Initialize and return a new builder, with the given initial capacity.
define %v3.bld @v3.bld.new(i64 %capacity, %work_t* %cur.work, i32 %fixedSize) {
  %elemSizePtr = getelementptr i1, i1* null, i32 1
  %elemSize = ptrtoint i1* %elemSizePtr to i64
  %newVb = call i8* @weld_rt_new_vb(i64 %elemSize, i64 %capacity, i32 %fixedSize)
  call void @v3.bld.newPiece(%v3.bld %newVb, %work_t* %cur.work)
  ret %v3.bld %newVb
}

define void @v3.bld.newPiece(%v3.bld %bldPtr, %work_t* %cur.work) {
  call void @weld_rt_new_vb_piece(i8* %bldPtr, %work_t* %cur.work)
  ret void
}

; Append a value into a builder, growing its space if needed.
define %v3.bld @v3.bld.merge(%v3.bld %bldPtr, i1 %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %full = icmp eq i64 %size, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr i1, i1* null, i32 1
  %elemSize = ptrtoint i1* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to i1*
  %insertPtr = getelementptr i1, i1* %elements, i64 %size
  store i1 %value, i1* %insertPtr
  %newSize = add i64 %size, 1
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v3.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %v3 @v3.bld.result(%v3.bld %bldPtr) {
  %out = call %vb.out @weld_rt_result_vb(i8* %bldPtr)
  %bytes = extractvalue %vb.out %out, 0
  %size = extractvalue %vb.out %out, 1
  %elems = bitcast i8* %bytes to i1*
  %1 = insertvalue %v3 undef, i1* %elems, 0
  %2 = insertvalue %v3 %1, i64 %size, 1
  ret %v3 %2
}

; Get the length of a vector.
define i64 @v3.size(%v3 %vec) {
  %size = extractvalue %v3 %vec, 1
  ret i64 %size
}

; Get a pointer to the index'th element.
define i1* @v3.at(%v3 %vec, i64 %index) {
  %elements = extractvalue %v3 %vec, 0
  %ptr = getelementptr i1, i1* %elements, i64 %index
  ret i1* %ptr
}


; Get the length of a VecBuilder.
define i64 @v3.bld.size(%v3.bld %bldPtr, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  ret i64 %size
}

; Get a pointer to the index'th element of a VecBuilder.
define i1* @v3.bld.at(%v3.bld %bldPtr, i64 %index, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %bytes = extractvalue %vb.vp %curPiece, 0
  %elements = bitcast i8* %bytes to i1*
  %ptr = getelementptr i1, i1* %elements, i64 %index
  ret i1* %ptr
}

; Compute the hash code of a vector.
; TODO: We should hash more bytes at a time if elements are non-pointer types.
define i32 @v3.hash(%v3 %vec) {
entry:
  %elements = extractvalue %v3 %vec, 0
  %size = extractvalue %v3 %vec, 1
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %prevHash = phi i32 [ 0, %entry ], [ %newHash, %body ]
  %ptr = getelementptr i1, i1* %elements, i64 %i
  %elem = load i1, i1* %ptr
  %elemHash = call i32 @i1.hash(i1 %elem)
  %newHash = call i32 @hash_combine(i32 %prevHash, i32 %elemHash)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
  
done:
  %res = phi i32 [ 0, %entry ], [ %newHash, %body ]
  ret i32 %res
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v3.bld.hash(%v3.bld %bld) {
  ret i32 0
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v3.vm.bld.hash(%v3.vm.bld %bld) {
  ret i32 0
}

; Dummy comparison function; this is needed for structs that use these vecbuilders as fields.
define i32 @v3.bld.cmp(%v3.bld %bld1, %v3.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function; this is needed for structs that use these vecmergers as fields.
define i32 @v3.vm.bld.cmp(%v3.vm.bld %bld1, %v3.vm.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function for builders.
define i1 @v3.bld.eq(%v3.bld %a, %v3.bld %b) {
  ret i1 0
}

; Dummy comparison function for builders.
define i1 @v3.vm.bld.eq(%v3.vm.bld %a, %v3.vm.bld %b) {
  ret i1 0
}

; Vector extensions for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - VECSIZE: Size of vectors.

; Get a pointer to the index'th element, fetching a vector
define <4 x i1>* @v3.vat(%v3 %vec, i64 %index) {
  %elements = extractvalue %v3 %vec, 0
  %ptr = getelementptr i1, i1* %elements, i64 %index
  %retPtr = bitcast i1* %ptr to <4 x i1>*
  ret <4 x i1>* %retPtr
}

; Append a value into a builder, growing its space if needed.
define %v3.bld @v3.bld.vmerge(%v3.bld %bldPtr, <4 x i1> %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %adjusted = add i64 %size, 4
  %full = icmp sgt i64 %adjusted, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr i1, i1* null, i32 1
  %elemSize = ptrtoint i1* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to i1*
  %insertPtr = getelementptr i1, i1* %elements, i64 %size
  %vecInsertPtr = bitcast i1* %insertPtr to <4 x i1>*
  store <4 x i1> %value, <4 x i1>* %vecInsertPtr, align 1
  %newSize = add i64 %size, 4
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v3.bld %bldPtr
}

; Vector extension to generate comparison functions. We use a different file,
; vector_comparisons_i8.ll, for comparisons on arrays of bytes, which are optimized.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)



; Compare two vectors lexicographically.
define i32 @v3.cmp(%v3 %a, %v3 %b) {
entry:
  %elemsA = extractvalue %v3 %a, 0
  %elemsB = extractvalue %v3 %b, 0
  %sizeA = extractvalue %v3 %a, 1
  %sizeB = extractvalue %v3 %b, 1
  %cond1 = icmp ult i64 %sizeA, %sizeB
  %minSize = select i1 %cond1, i64 %sizeA, i64 %sizeB
  %cond = icmp ult i64 0, %minSize
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body2 ]
  %ptrA = getelementptr i1, i1* %elemsA, i64 %i
  %ptrB = getelementptr i1, i1* %elemsB, i64 %i
  %elemA = load i1, i1* %ptrA
  %elemB = load i1, i1* %ptrB
  %cmp = call i32 @i1.cmp(i1 %elemA, i1 %elemB)
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

; Compare two vectors for equality.
define i1 @v3.eq(%v3 %a, %v3 %b) {
  %sizeA = extractvalue %v3 %a, 1
  %sizeB = extractvalue %v3 %b, 1
  %cond = icmp eq i64 %sizeA, %sizeB
  br i1 %cond, label %same_size, label %different_size
same_size:
  %cmp = call i32 @v3.cmp(%v3 %a, %v3 %b)
  %res = icmp eq i32 %cmp, 0
  ret i1 %res
different_size:
  ret i1 0
}
; Template for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)
; - VECSIZE: Size of vectors.

%v4 = type { i32*, i64 }           ; elements, size
%v4.bld = type i8*

; VecMerger
%v4.vm.bld = type %v4*

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define %v4.vm.bld @v4.vm.bld.getPtrIndexed(%v4.vm.bld %bldPtr, i32 %i) alwaysinline {
  %mergerPtr = getelementptr %v4, %v4* null, i32 1
  %mergerSize = ptrtoint %v4* %mergerPtr to i64
  %asPtr = bitcast %v4.vm.bld %bldPtr to i8*
  %rawPtr = call i8* @weld_rt_get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %v4.vm.bld
  ret %v4.vm.bld %ptr
}

; Initialize and return a new vecmerger with the given initial vector.
define %v4.vm.bld @v4.vm.bld.new(%v4 %vec) {
  %nworkers = call i32 @weld_rt_get_nworkers()
  %structSizePtr = getelementptr %v4, %v4* null, i32 1
  %structSize = ptrtoint %v4* %structSizePtr to i64
  
  %bldPtr = call i8* @weld_rt_new_merger(i64 %structSize, i32 %nworkers)
  %typedPtr = bitcast i8* %bldPtr to %v4.vm.bld
  
  ; Copy the initial value into the first vector
  %first = call %v4.vm.bld @v4.vm.bld.getPtrIndexed(%v4.vm.bld %typedPtr, i32 0)
  %cloned = call %v4 @v4.clone(%v4 %vec)
  %capacity = call i64 @v4.size(%v4 %vec)
  store %v4 %cloned, %v4.vm.bld %first
  br label %entry
  
entry:
  %cond = icmp ult i32 1, %nworkers
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i32 [ 1, %entry ], [ %i2, %body ]
  %vecPtr = call %v4* @v4.vm.bld.getPtrIndexed(%v4.vm.bld %typedPtr, i32 %i)
  %newVec = call %v4 @v4.new(i64 %capacity)
  call void @v4.zero(%v4 %newVec)
  store %v4 %newVec, %v4* %vecPtr
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done
  
done:
  ret %v4.vm.bld %typedPtr
}

; Returns a pointer to the value an element should be merged into.
; The caller should perform the merge operation on the contents of this pointer
; and then store the resulting value back.
define i8* @v4.vm.bld.merge_ptr(%v4.vm.bld %bldPtr, i64 %index, i32 %workerId) {
  %bldPtrLocal = call %v4* @v4.vm.bld.getPtrIndexed(%v4.vm.bld %bldPtr, i32 %workerId)
  %vec = load %v4, %v4* %bldPtrLocal
  %elem = call i32* @v4.at(%v4 %vec, i64 %index)
  %elemPtrRaw = bitcast i32* %elem to i8*
  ret i8* %elemPtrRaw
}

; Initialize and return a new vector with the given size.
define %v4 @v4.new(i64 %size) {
  %elemSizePtr = getelementptr i32, i32* null, i32 1
  %elemSize = ptrtoint i32* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %runId = call i64 @weld_rt_get_run_id()
  %bytes = call i8* @weld_run_malloc(i64 %runId, i64 %allocSize)
  %elements = bitcast i8* %bytes to i32*
  %1 = insertvalue %v4 undef, i32* %elements, 0
  %2 = insertvalue %v4 %1, i64 %size, 1
  ret %v4 %2
}

; Zeroes a vector's underlying buffer.
define void @v4.zero(%v4 %v) {
  %elements = extractvalue %v4 %v, 0
  %size = extractvalue %v4 %v, 1
  %bytes = bitcast i32* %elements to i8*
  
  %elemSizePtr = getelementptr i32, i32* null, i32 1
  %elemSize = ptrtoint i32* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  ret void
}

; Clone a vector.
define %v4 @v4.clone(%v4 %vec) {
  %elements = extractvalue %v4 %vec, 0
  %size = extractvalue %v4 %vec, 1
  %entrySizePtr = getelementptr i32, i32* null, i32 1
  %entrySize = ptrtoint i32* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %size
  %bytes = bitcast i32* %elements to i8*
  %vec2 = call %v4 @v4.new(i64 %size)
  %elements2 = extractvalue %v4 %vec2, 0
  %bytes2 = bitcast i32* %elements2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  ret %v4 %vec2
}

; Get a new vec object that starts at the index'th element of the existing vector, and has size size.
; If the specified size is greater than the remaining size, then the remaining size is used.
define %v4 @v4.slice(%v4 %vec, i64 %index, i64 %size) {
  ; Check if size greater than remaining size
  %currSize = extractvalue %v4 %vec, 1
  %remSize = sub i64 %currSize, %index
  %sgtr = icmp ugt i64 %size, %remSize
  %finSize = select i1 %sgtr, i64 %remSize, i64 %size
  
  %elements = extractvalue %v4 %vec, 0
  %newElements = getelementptr i32, i32* %elements, i64 %index
  %1 = insertvalue %v4 undef, i32* %newElements, 0
  %2 = insertvalue %v4 %1, i64 %finSize, 1
  
  ret %v4 %2
}

; Initialize and return a new builder, with the given initial capacity.
define %v4.bld @v4.bld.new(i64 %capacity, %work_t* %cur.work, i32 %fixedSize) {
  %elemSizePtr = getelementptr i32, i32* null, i32 1
  %elemSize = ptrtoint i32* %elemSizePtr to i64
  %newVb = call i8* @weld_rt_new_vb(i64 %elemSize, i64 %capacity, i32 %fixedSize)
  call void @v4.bld.newPiece(%v4.bld %newVb, %work_t* %cur.work)
  ret %v4.bld %newVb
}

define void @v4.bld.newPiece(%v4.bld %bldPtr, %work_t* %cur.work) {
  call void @weld_rt_new_vb_piece(i8* %bldPtr, %work_t* %cur.work)
  ret void
}

; Append a value into a builder, growing its space if needed.
define %v4.bld @v4.bld.merge(%v4.bld %bldPtr, i32 %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %full = icmp eq i64 %size, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr i32, i32* null, i32 1
  %elemSize = ptrtoint i32* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to i32*
  %insertPtr = getelementptr i32, i32* %elements, i64 %size
  store i32 %value, i32* %insertPtr
  %newSize = add i64 %size, 1
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v4.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %v4 @v4.bld.result(%v4.bld %bldPtr) {
  %out = call %vb.out @weld_rt_result_vb(i8* %bldPtr)
  %bytes = extractvalue %vb.out %out, 0
  %size = extractvalue %vb.out %out, 1
  %elems = bitcast i8* %bytes to i32*
  %1 = insertvalue %v4 undef, i32* %elems, 0
  %2 = insertvalue %v4 %1, i64 %size, 1
  ret %v4 %2
}

; Get the length of a vector.
define i64 @v4.size(%v4 %vec) {
  %size = extractvalue %v4 %vec, 1
  ret i64 %size
}

; Get a pointer to the index'th element.
define i32* @v4.at(%v4 %vec, i64 %index) {
  %elements = extractvalue %v4 %vec, 0
  %ptr = getelementptr i32, i32* %elements, i64 %index
  ret i32* %ptr
}


; Get the length of a VecBuilder.
define i64 @v4.bld.size(%v4.bld %bldPtr, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  ret i64 %size
}

; Get a pointer to the index'th element of a VecBuilder.
define i32* @v4.bld.at(%v4.bld %bldPtr, i64 %index, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %bytes = extractvalue %vb.vp %curPiece, 0
  %elements = bitcast i8* %bytes to i32*
  %ptr = getelementptr i32, i32* %elements, i64 %index
  ret i32* %ptr
}

; Compute the hash code of a vector.
; TODO: We should hash more bytes at a time if elements are non-pointer types.
define i32 @v4.hash(%v4 %vec) {
entry:
  %elements = extractvalue %v4 %vec, 0
  %size = extractvalue %v4 %vec, 1
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %prevHash = phi i32 [ 0, %entry ], [ %newHash, %body ]
  %ptr = getelementptr i32, i32* %elements, i64 %i
  %elem = load i32, i32* %ptr
  %elemHash = call i32 @i32.hash(i32 %elem)
  %newHash = call i32 @hash_combine(i32 %prevHash, i32 %elemHash)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
  
done:
  %res = phi i32 [ 0, %entry ], [ %newHash, %body ]
  ret i32 %res
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v4.bld.hash(%v4.bld %bld) {
  ret i32 0
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v4.vm.bld.hash(%v4.vm.bld %bld) {
  ret i32 0
}

; Dummy comparison function; this is needed for structs that use these vecbuilders as fields.
define i32 @v4.bld.cmp(%v4.bld %bld1, %v4.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function; this is needed for structs that use these vecmergers as fields.
define i32 @v4.vm.bld.cmp(%v4.vm.bld %bld1, %v4.vm.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function for builders.
define i1 @v4.bld.eq(%v4.bld %a, %v4.bld %b) {
  ret i1 0
}

; Dummy comparison function for builders.
define i1 @v4.vm.bld.eq(%v4.vm.bld %a, %v4.vm.bld %b) {
  ret i1 0
}

; Vector extensions for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - VECSIZE: Size of vectors.

; Get a pointer to the index'th element, fetching a vector
define <4 x i32>* @v4.vat(%v4 %vec, i64 %index) {
  %elements = extractvalue %v4 %vec, 0
  %ptr = getelementptr i32, i32* %elements, i64 %index
  %retPtr = bitcast i32* %ptr to <4 x i32>*
  ret <4 x i32>* %retPtr
}

; Append a value into a builder, growing its space if needed.
define %v4.bld @v4.bld.vmerge(%v4.bld %bldPtr, <4 x i32> %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %adjusted = add i64 %size, 4
  %full = icmp sgt i64 %adjusted, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr i32, i32* null, i32 1
  %elemSize = ptrtoint i32* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to i32*
  %insertPtr = getelementptr i32, i32* %elements, i64 %size
  %vecInsertPtr = bitcast i32* %insertPtr to <4 x i32>*
  store <4 x i32> %value, <4 x i32>* %vecInsertPtr, align 1
  %newSize = add i64 %size, 4
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v4.bld %bldPtr
}

; Vector extension to generate comparison functions. We use a different file,
; vector_comparisons_i8.ll, for comparisons on arrays of bytes, which are optimized.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)



; Compare two vectors lexicographically.
define i32 @v4.cmp(%v4 %a, %v4 %b) {
entry:
  %elemsA = extractvalue %v4 %a, 0
  %elemsB = extractvalue %v4 %b, 0
  %sizeA = extractvalue %v4 %a, 1
  %sizeB = extractvalue %v4 %b, 1
  %cond1 = icmp ult i64 %sizeA, %sizeB
  %minSize = select i1 %cond1, i64 %sizeA, i64 %sizeB
  %cond = icmp ult i64 0, %minSize
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body2 ]
  %ptrA = getelementptr i32, i32* %elemsA, i64 %i
  %ptrB = getelementptr i32, i32* %elemsB, i64 %i
  %elemA = load i32, i32* %ptrA
  %elemB = load i32, i32* %ptrB
  %cmp = call i32 @i32.cmp(i32 %elemA, i32 %elemB)
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

; Compare two vectors for equality.
define i1 @v4.eq(%v4 %a, %v4 %b) {
  %sizeA = extractvalue %v4 %a, 1
  %sizeB = extractvalue %v4 %b, 1
  %cond = icmp eq i64 %sizeA, %sizeB
  br i1 %cond, label %same_size, label %different_size
same_size:
  %cmp = call i32 @v4.cmp(%v4 %a, %v4 %b)
  %res = icmp eq i32 %cmp, 0
  ret i1 %res
different_size:
  ret i1 0
}
%s4 = type { i64, %v3.bld, i32, %v4.bld, %v4.bld, %v0.bld }
define i32 @s4.hash(%s4 %value) {
  %p.p120 = extractvalue %s4 %value, 0
  %p.p121 = call i32 @i64.hash(i64 %p.p120)
  %p.p122 = call i32 @hash_combine(i32 0, i32 %p.p121)
  %p.p123 = extractvalue %s4 %value, 1
  %p.p124 = call i32 @v3.bld.hash(%v3.bld %p.p123)
  %p.p125 = call i32 @hash_combine(i32 %p.p122, i32 %p.p124)
  %p.p126 = extractvalue %s4 %value, 2
  %p.p127 = call i32 @i32.hash(i32 %p.p126)
  %p.p128 = call i32 @hash_combine(i32 %p.p125, i32 %p.p127)
  %p.p129 = extractvalue %s4 %value, 3
  %p.p130 = call i32 @v4.bld.hash(%v4.bld %p.p129)
  %p.p131 = call i32 @hash_combine(i32 %p.p128, i32 %p.p130)
  %p.p132 = extractvalue %s4 %value, 4
  %p.p133 = call i32 @v4.bld.hash(%v4.bld %p.p132)
  %p.p134 = call i32 @hash_combine(i32 %p.p131, i32 %p.p133)
  %p.p135 = extractvalue %s4 %value, 5
  %p.p136 = call i32 @v0.bld.hash(%v0.bld %p.p135)
  %p.p137 = call i32 @hash_combine(i32 %p.p134, i32 %p.p136)
  ret i32 %p.p137
}

define i32 @s4.cmp(%s4 %a, %s4 %b) {
  %p.p138 = extractvalue %s4 %a , 0
  %p.p139 = extractvalue %s4 %b, 0
  %p.p140 = call i32 @i64.cmp(i64 %p.p138, i64 %p.p139)
  %p.p141 = icmp ne i32 %p.p140, 0
  br i1 %p.p141, label %l0, label %l1
l0:
  ret i32 %p.p140
l1:
  %p.p142 = extractvalue %s4 %a , 1
  %p.p143 = extractvalue %s4 %b, 1
  %p.p144 = call i32 @v3.bld.cmp(%v3.bld %p.p142, %v3.bld %p.p143)
  %p.p145 = icmp ne i32 %p.p144, 0
  br i1 %p.p145, label %l2, label %l3
l2:
  ret i32 %p.p144
l3:
  %p.p146 = extractvalue %s4 %a , 2
  %p.p147 = extractvalue %s4 %b, 2
  %p.p148 = call i32 @i32.cmp(i32 %p.p146, i32 %p.p147)
  %p.p149 = icmp ne i32 %p.p148, 0
  br i1 %p.p149, label %l4, label %l5
l4:
  ret i32 %p.p148
l5:
  %p.p150 = extractvalue %s4 %a , 3
  %p.p151 = extractvalue %s4 %b, 3
  %p.p152 = call i32 @v4.bld.cmp(%v4.bld %p.p150, %v4.bld %p.p151)
  %p.p153 = icmp ne i32 %p.p152, 0
  br i1 %p.p153, label %l6, label %l7
l6:
  ret i32 %p.p152
l7:
  %p.p154 = extractvalue %s4 %a , 4
  %p.p155 = extractvalue %s4 %b, 4
  %p.p156 = call i32 @v4.bld.cmp(%v4.bld %p.p154, %v4.bld %p.p155)
  %p.p157 = icmp ne i32 %p.p156, 0
  br i1 %p.p157, label %l8, label %l9
l8:
  ret i32 %p.p156
l9:
  %p.p158 = extractvalue %s4 %a , 5
  %p.p159 = extractvalue %s4 %b, 5
  %p.p160 = call i32 @v0.bld.cmp(%v0.bld %p.p158, %v0.bld %p.p159)
  %p.p161 = icmp ne i32 %p.p160, 0
  br i1 %p.p161, label %l10, label %l11
l10:
  ret i32 %p.p160
l11:
  ret i32 0
}

define i1 @s4.eq(%s4 %a, %s4 %b) {
  %p.p162 = extractvalue %s4 %a , 0
  %p.p163 = extractvalue %s4 %b, 0
  %p.p164 = call i1 @i64.eq(i64 %p.p162, i64 %p.p163)
  br i1 %p.p164, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p165 = extractvalue %s4 %a , 1
  %p.p166 = extractvalue %s4 %b, 1
  %p.p167 = call i1 @v3.bld.eq(%v3.bld %p.p165, %v3.bld %p.p166)
  br i1 %p.p167, label %l3, label %l2
l2:
  ret i1 0
l3:
  %p.p168 = extractvalue %s4 %a , 2
  %p.p169 = extractvalue %s4 %b, 2
  %p.p170 = call i1 @i32.eq(i32 %p.p168, i32 %p.p169)
  br i1 %p.p170, label %l5, label %l4
l4:
  ret i1 0
l5:
  %p.p171 = extractvalue %s4 %a , 3
  %p.p172 = extractvalue %s4 %b, 3
  %p.p173 = call i1 @v4.bld.eq(%v4.bld %p.p171, %v4.bld %p.p172)
  br i1 %p.p173, label %l7, label %l6
l6:
  ret i1 0
l7:
  %p.p174 = extractvalue %s4 %a , 4
  %p.p175 = extractvalue %s4 %b, 4
  %p.p176 = call i1 @v4.bld.eq(%v4.bld %p.p174, %v4.bld %p.p175)
  br i1 %p.p176, label %l9, label %l8
l8:
  ret i1 0
l9:
  %p.p177 = extractvalue %s4 %a , 5
  %p.p178 = extractvalue %s4 %b, 5
  %p.p179 = call i1 @v0.bld.eq(%v0.bld %p.p177, %v0.bld %p.p178)
  br i1 %p.p179, label %l11, label %l10
l10:
  ret i1 0
l11:
  ret i1 1
}

%s5 = type { %s4, i1 }
define i32 @s5.hash(%s5 %value) {
  %p.p180 = extractvalue %s5 %value, 0
  %p.p181 = call i32 @s4.hash(%s4 %p.p180)
  %p.p182 = call i32 @hash_combine(i32 0, i32 %p.p181)
  %p.p183 = extractvalue %s5 %value, 1
  %p.p184 = call i32 @i1.hash(i1 %p.p183)
  %p.p185 = call i32 @hash_combine(i32 %p.p182, i32 %p.p184)
  ret i32 %p.p185
}

define i32 @s5.cmp(%s5 %a, %s5 %b) {
  %p.p186 = extractvalue %s5 %a , 0
  %p.p187 = extractvalue %s5 %b, 0
  %p.p188 = call i32 @s4.cmp(%s4 %p.p186, %s4 %p.p187)
  %p.p189 = icmp ne i32 %p.p188, 0
  br i1 %p.p189, label %l0, label %l1
l0:
  ret i32 %p.p188
l1:
  %p.p190 = extractvalue %s5 %a , 1
  %p.p191 = extractvalue %s5 %b, 1
  %p.p192 = call i32 @i1.cmp(i1 %p.p190, i1 %p.p191)
  %p.p193 = icmp ne i32 %p.p192, 0
  br i1 %p.p193, label %l2, label %l3
l2:
  ret i32 %p.p192
l3:
  ret i32 0
}

define i1 @s5.eq(%s5 %a, %s5 %b) {
  %p.p194 = extractvalue %s5 %a , 0
  %p.p195 = extractvalue %s5 %b, 0
  %p.p196 = call i1 @s4.eq(%s4 %p.p194, %s4 %p.p195)
  br i1 %p.p196, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p197 = extractvalue %s5 %a , 1
  %p.p198 = extractvalue %s5 %b, 1
  %p.p199 = call i1 @i1.eq(i1 %p.p197, i1 %p.p198)
  br i1 %p.p199, label %l3, label %l2
l2:
  ret i1 0
l3:
  ret i1 1
}

; Template for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)
; - VECSIZE: Size of vectors.

%v5 = type { i64*, i64 }           ; elements, size
%v5.bld = type i8*

; VecMerger
%v5.vm.bld = type %v5*

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define %v5.vm.bld @v5.vm.bld.getPtrIndexed(%v5.vm.bld %bldPtr, i32 %i) alwaysinline {
  %mergerPtr = getelementptr %v5, %v5* null, i32 1
  %mergerSize = ptrtoint %v5* %mergerPtr to i64
  %asPtr = bitcast %v5.vm.bld %bldPtr to i8*
  %rawPtr = call i8* @weld_rt_get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %v5.vm.bld
  ret %v5.vm.bld %ptr
}

; Initialize and return a new vecmerger with the given initial vector.
define %v5.vm.bld @v5.vm.bld.new(%v5 %vec) {
  %nworkers = call i32 @weld_rt_get_nworkers()
  %structSizePtr = getelementptr %v5, %v5* null, i32 1
  %structSize = ptrtoint %v5* %structSizePtr to i64
  
  %bldPtr = call i8* @weld_rt_new_merger(i64 %structSize, i32 %nworkers)
  %typedPtr = bitcast i8* %bldPtr to %v5.vm.bld
  
  ; Copy the initial value into the first vector
  %first = call %v5.vm.bld @v5.vm.bld.getPtrIndexed(%v5.vm.bld %typedPtr, i32 0)
  %cloned = call %v5 @v5.clone(%v5 %vec)
  %capacity = call i64 @v5.size(%v5 %vec)
  store %v5 %cloned, %v5.vm.bld %first
  br label %entry
  
entry:
  %cond = icmp ult i32 1, %nworkers
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i32 [ 1, %entry ], [ %i2, %body ]
  %vecPtr = call %v5* @v5.vm.bld.getPtrIndexed(%v5.vm.bld %typedPtr, i32 %i)
  %newVec = call %v5 @v5.new(i64 %capacity)
  call void @v5.zero(%v5 %newVec)
  store %v5 %newVec, %v5* %vecPtr
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done
  
done:
  ret %v5.vm.bld %typedPtr
}

; Returns a pointer to the value an element should be merged into.
; The caller should perform the merge operation on the contents of this pointer
; and then store the resulting value back.
define i8* @v5.vm.bld.merge_ptr(%v5.vm.bld %bldPtr, i64 %index, i32 %workerId) {
  %bldPtrLocal = call %v5* @v5.vm.bld.getPtrIndexed(%v5.vm.bld %bldPtr, i32 %workerId)
  %vec = load %v5, %v5* %bldPtrLocal
  %elem = call i64* @v5.at(%v5 %vec, i64 %index)
  %elemPtrRaw = bitcast i64* %elem to i8*
  ret i8* %elemPtrRaw
}

; Initialize and return a new vector with the given size.
define %v5 @v5.new(i64 %size) {
  %elemSizePtr = getelementptr i64, i64* null, i32 1
  %elemSize = ptrtoint i64* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %runId = call i64 @weld_rt_get_run_id()
  %bytes = call i8* @weld_run_malloc(i64 %runId, i64 %allocSize)
  %elements = bitcast i8* %bytes to i64*
  %1 = insertvalue %v5 undef, i64* %elements, 0
  %2 = insertvalue %v5 %1, i64 %size, 1
  ret %v5 %2
}

; Zeroes a vector's underlying buffer.
define void @v5.zero(%v5 %v) {
  %elements = extractvalue %v5 %v, 0
  %size = extractvalue %v5 %v, 1
  %bytes = bitcast i64* %elements to i8*
  
  %elemSizePtr = getelementptr i64, i64* null, i32 1
  %elemSize = ptrtoint i64* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  ret void
}

; Clone a vector.
define %v5 @v5.clone(%v5 %vec) {
  %elements = extractvalue %v5 %vec, 0
  %size = extractvalue %v5 %vec, 1
  %entrySizePtr = getelementptr i64, i64* null, i32 1
  %entrySize = ptrtoint i64* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %size
  %bytes = bitcast i64* %elements to i8*
  %vec2 = call %v5 @v5.new(i64 %size)
  %elements2 = extractvalue %v5 %vec2, 0
  %bytes2 = bitcast i64* %elements2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  ret %v5 %vec2
}

; Get a new vec object that starts at the index'th element of the existing vector, and has size size.
; If the specified size is greater than the remaining size, then the remaining size is used.
define %v5 @v5.slice(%v5 %vec, i64 %index, i64 %size) {
  ; Check if size greater than remaining size
  %currSize = extractvalue %v5 %vec, 1
  %remSize = sub i64 %currSize, %index
  %sgtr = icmp ugt i64 %size, %remSize
  %finSize = select i1 %sgtr, i64 %remSize, i64 %size
  
  %elements = extractvalue %v5 %vec, 0
  %newElements = getelementptr i64, i64* %elements, i64 %index
  %1 = insertvalue %v5 undef, i64* %newElements, 0
  %2 = insertvalue %v5 %1, i64 %finSize, 1
  
  ret %v5 %2
}

; Initialize and return a new builder, with the given initial capacity.
define %v5.bld @v5.bld.new(i64 %capacity, %work_t* %cur.work, i32 %fixedSize) {
  %elemSizePtr = getelementptr i64, i64* null, i32 1
  %elemSize = ptrtoint i64* %elemSizePtr to i64
  %newVb = call i8* @weld_rt_new_vb(i64 %elemSize, i64 %capacity, i32 %fixedSize)
  call void @v5.bld.newPiece(%v5.bld %newVb, %work_t* %cur.work)
  ret %v5.bld %newVb
}

define void @v5.bld.newPiece(%v5.bld %bldPtr, %work_t* %cur.work) {
  call void @weld_rt_new_vb_piece(i8* %bldPtr, %work_t* %cur.work)
  ret void
}

; Append a value into a builder, growing its space if needed.
define %v5.bld @v5.bld.merge(%v5.bld %bldPtr, i64 %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %full = icmp eq i64 %size, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr i64, i64* null, i32 1
  %elemSize = ptrtoint i64* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to i64*
  %insertPtr = getelementptr i64, i64* %elements, i64 %size
  store i64 %value, i64* %insertPtr
  %newSize = add i64 %size, 1
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v5.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %v5 @v5.bld.result(%v5.bld %bldPtr) {
  %out = call %vb.out @weld_rt_result_vb(i8* %bldPtr)
  %bytes = extractvalue %vb.out %out, 0
  %size = extractvalue %vb.out %out, 1
  %elems = bitcast i8* %bytes to i64*
  %1 = insertvalue %v5 undef, i64* %elems, 0
  %2 = insertvalue %v5 %1, i64 %size, 1
  ret %v5 %2
}

; Get the length of a vector.
define i64 @v5.size(%v5 %vec) {
  %size = extractvalue %v5 %vec, 1
  ret i64 %size
}

; Get a pointer to the index'th element.
define i64* @v5.at(%v5 %vec, i64 %index) {
  %elements = extractvalue %v5 %vec, 0
  %ptr = getelementptr i64, i64* %elements, i64 %index
  ret i64* %ptr
}


; Get the length of a VecBuilder.
define i64 @v5.bld.size(%v5.bld %bldPtr, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  ret i64 %size
}

; Get a pointer to the index'th element of a VecBuilder.
define i64* @v5.bld.at(%v5.bld %bldPtr, i64 %index, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %bytes = extractvalue %vb.vp %curPiece, 0
  %elements = bitcast i8* %bytes to i64*
  %ptr = getelementptr i64, i64* %elements, i64 %index
  ret i64* %ptr
}

; Compute the hash code of a vector.
; TODO: We should hash more bytes at a time if elements are non-pointer types.
define i32 @v5.hash(%v5 %vec) {
entry:
  %elements = extractvalue %v5 %vec, 0
  %size = extractvalue %v5 %vec, 1
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %prevHash = phi i32 [ 0, %entry ], [ %newHash, %body ]
  %ptr = getelementptr i64, i64* %elements, i64 %i
  %elem = load i64, i64* %ptr
  %elemHash = call i32 @i64.hash(i64 %elem)
  %newHash = call i32 @hash_combine(i32 %prevHash, i32 %elemHash)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
  
done:
  %res = phi i32 [ 0, %entry ], [ %newHash, %body ]
  ret i32 %res
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v5.bld.hash(%v5.bld %bld) {
  ret i32 0
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v5.vm.bld.hash(%v5.vm.bld %bld) {
  ret i32 0
}

; Dummy comparison function; this is needed for structs that use these vecbuilders as fields.
define i32 @v5.bld.cmp(%v5.bld %bld1, %v5.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function; this is needed for structs that use these vecmergers as fields.
define i32 @v5.vm.bld.cmp(%v5.vm.bld %bld1, %v5.vm.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function for builders.
define i1 @v5.bld.eq(%v5.bld %a, %v5.bld %b) {
  ret i1 0
}

; Dummy comparison function for builders.
define i1 @v5.vm.bld.eq(%v5.vm.bld %a, %v5.vm.bld %b) {
  ret i1 0
}

; Vector extensions for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - VECSIZE: Size of vectors.

; Get a pointer to the index'th element, fetching a vector
define <4 x i64>* @v5.vat(%v5 %vec, i64 %index) {
  %elements = extractvalue %v5 %vec, 0
  %ptr = getelementptr i64, i64* %elements, i64 %index
  %retPtr = bitcast i64* %ptr to <4 x i64>*
  ret <4 x i64>* %retPtr
}

; Append a value into a builder, growing its space if needed.
define %v5.bld @v5.bld.vmerge(%v5.bld %bldPtr, <4 x i64> %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %adjusted = add i64 %size, 4
  %full = icmp sgt i64 %adjusted, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr i64, i64* null, i32 1
  %elemSize = ptrtoint i64* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to i64*
  %insertPtr = getelementptr i64, i64* %elements, i64 %size
  %vecInsertPtr = bitcast i64* %insertPtr to <4 x i64>*
  store <4 x i64> %value, <4 x i64>* %vecInsertPtr, align 1
  %newSize = add i64 %size, 4
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v5.bld %bldPtr
}

; Vector extension to generate comparison functions. We use a different file,
; vector_comparisons_i8.ll, for comparisons on arrays of bytes, which are optimized.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)



; Compare two vectors lexicographically.
define i32 @v5.cmp(%v5 %a, %v5 %b) {
entry:
  %elemsA = extractvalue %v5 %a, 0
  %elemsB = extractvalue %v5 %b, 0
  %sizeA = extractvalue %v5 %a, 1
  %sizeB = extractvalue %v5 %b, 1
  %cond1 = icmp ult i64 %sizeA, %sizeB
  %minSize = select i1 %cond1, i64 %sizeA, i64 %sizeB
  %cond = icmp ult i64 0, %minSize
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body2 ]
  %ptrA = getelementptr i64, i64* %elemsA, i64 %i
  %ptrB = getelementptr i64, i64* %elemsB, i64 %i
  %elemA = load i64, i64* %ptrA
  %elemB = load i64, i64* %ptrB
  %cmp = call i32 @i64.cmp(i64 %elemA, i64 %elemB)
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

; Compare two vectors for equality.
define i1 @v5.eq(%v5 %a, %v5 %b) {
  %sizeA = extractvalue %v5 %a, 1
  %sizeB = extractvalue %v5 %b, 1
  %cond = icmp eq i64 %sizeA, %sizeB
  br i1 %cond, label %same_size, label %different_size
same_size:
  %cmp = call i32 @v5.cmp(%v5 %a, %v5 %b)
  %res = icmp eq i32 %cmp, 0
  ret i1 %res
different_size:
  ret i1 0
}
; Template for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)
; - VECSIZE: Size of vectors.

%v6 = type { double*, i64 }           ; elements, size
%v6.bld = type i8*

; VecMerger
%v6.vm.bld = type %v6*

; Returns a pointer to builder data for index i (generally, i is the thread ID).
define %v6.vm.bld @v6.vm.bld.getPtrIndexed(%v6.vm.bld %bldPtr, i32 %i) alwaysinline {
  %mergerPtr = getelementptr %v6, %v6* null, i32 1
  %mergerSize = ptrtoint %v6* %mergerPtr to i64
  %asPtr = bitcast %v6.vm.bld %bldPtr to i8*
  %rawPtr = call i8* @weld_rt_get_merger_at_index(i8* %asPtr, i64 %mergerSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %v6.vm.bld
  ret %v6.vm.bld %ptr
}

; Initialize and return a new vecmerger with the given initial vector.
define %v6.vm.bld @v6.vm.bld.new(%v6 %vec) {
  %nworkers = call i32 @weld_rt_get_nworkers()
  %structSizePtr = getelementptr %v6, %v6* null, i32 1
  %structSize = ptrtoint %v6* %structSizePtr to i64
  
  %bldPtr = call i8* @weld_rt_new_merger(i64 %structSize, i32 %nworkers)
  %typedPtr = bitcast i8* %bldPtr to %v6.vm.bld
  
  ; Copy the initial value into the first vector
  %first = call %v6.vm.bld @v6.vm.bld.getPtrIndexed(%v6.vm.bld %typedPtr, i32 0)
  %cloned = call %v6 @v6.clone(%v6 %vec)
  %capacity = call i64 @v6.size(%v6 %vec)
  store %v6 %cloned, %v6.vm.bld %first
  br label %entry
  
entry:
  %cond = icmp ult i32 1, %nworkers
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i32 [ 1, %entry ], [ %i2, %body ]
  %vecPtr = call %v6* @v6.vm.bld.getPtrIndexed(%v6.vm.bld %typedPtr, i32 %i)
  %newVec = call %v6 @v6.new(i64 %capacity)
  call void @v6.zero(%v6 %newVec)
  store %v6 %newVec, %v6* %vecPtr
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done
  
done:
  ret %v6.vm.bld %typedPtr
}

; Returns a pointer to the value an element should be merged into.
; The caller should perform the merge operation on the contents of this pointer
; and then store the resulting value back.
define i8* @v6.vm.bld.merge_ptr(%v6.vm.bld %bldPtr, i64 %index, i32 %workerId) {
  %bldPtrLocal = call %v6* @v6.vm.bld.getPtrIndexed(%v6.vm.bld %bldPtr, i32 %workerId)
  %vec = load %v6, %v6* %bldPtrLocal
  %elem = call double* @v6.at(%v6 %vec, i64 %index)
  %elemPtrRaw = bitcast double* %elem to i8*
  ret i8* %elemPtrRaw
}

; Initialize and return a new vector with the given size.
define %v6 @v6.new(i64 %size) {
  %elemSizePtr = getelementptr double, double* null, i32 1
  %elemSize = ptrtoint double* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  %runId = call i64 @weld_rt_get_run_id()
  %bytes = call i8* @weld_run_malloc(i64 %runId, i64 %allocSize)
  %elements = bitcast i8* %bytes to double*
  %1 = insertvalue %v6 undef, double* %elements, 0
  %2 = insertvalue %v6 %1, i64 %size, 1
  ret %v6 %2
}

; Zeroes a vector's underlying buffer.
define void @v6.zero(%v6 %v) {
  %elements = extractvalue %v6 %v, 0
  %size = extractvalue %v6 %v, 1
  %bytes = bitcast double* %elements to i8*
  
  %elemSizePtr = getelementptr double, double* null, i32 1
  %elemSize = ptrtoint double* %elemSizePtr to i64
  %allocSize = mul i64 %elemSize, %size
  call void @llvm.memset.p0i8.i64(i8* %bytes, i8 0, i64 %allocSize, i32 8, i1 0)
  ret void
}

; Clone a vector.
define %v6 @v6.clone(%v6 %vec) {
  %elements = extractvalue %v6 %vec, 0
  %size = extractvalue %v6 %vec, 1
  %entrySizePtr = getelementptr double, double* null, i32 1
  %entrySize = ptrtoint double* %entrySizePtr to i64
  %allocSize = mul i64 %entrySize, %size
  %bytes = bitcast double* %elements to i8*
  %vec2 = call %v6 @v6.new(i64 %size)
  %elements2 = extractvalue %v6 %vec2, 0
  %bytes2 = bitcast double* %elements2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes2, i8* %bytes, i64 %allocSize, i32 8, i1 0)
  ret %v6 %vec2
}

; Get a new vec object that starts at the index'th element of the existing vector, and has size size.
; If the specified size is greater than the remaining size, then the remaining size is used.
define %v6 @v6.slice(%v6 %vec, i64 %index, i64 %size) {
  ; Check if size greater than remaining size
  %currSize = extractvalue %v6 %vec, 1
  %remSize = sub i64 %currSize, %index
  %sgtr = icmp ugt i64 %size, %remSize
  %finSize = select i1 %sgtr, i64 %remSize, i64 %size
  
  %elements = extractvalue %v6 %vec, 0
  %newElements = getelementptr double, double* %elements, i64 %index
  %1 = insertvalue %v6 undef, double* %newElements, 0
  %2 = insertvalue %v6 %1, i64 %finSize, 1
  
  ret %v6 %2
}

; Initialize and return a new builder, with the given initial capacity.
define %v6.bld @v6.bld.new(i64 %capacity, %work_t* %cur.work, i32 %fixedSize) {
  %elemSizePtr = getelementptr double, double* null, i32 1
  %elemSize = ptrtoint double* %elemSizePtr to i64
  %newVb = call i8* @weld_rt_new_vb(i64 %elemSize, i64 %capacity, i32 %fixedSize)
  call void @v6.bld.newPiece(%v6.bld %newVb, %work_t* %cur.work)
  ret %v6.bld %newVb
}

define void @v6.bld.newPiece(%v6.bld %bldPtr, %work_t* %cur.work) {
  call void @weld_rt_new_vb_piece(i8* %bldPtr, %work_t* %cur.work)
  ret void
}

; Append a value into a builder, growing its space if needed.
define %v6.bld @v6.bld.merge(%v6.bld %bldPtr, double %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %full = icmp eq i64 %size, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr double, double* null, i32 1
  %elemSize = ptrtoint double* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to double*
  %insertPtr = getelementptr double, double* %elements, i64 %size
  store double %value, double* %insertPtr
  %newSize = add i64 %size, 1
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v6.bld %bldPtr
}

; Complete building a vector, trimming any extra space left while growing it.
define %v6 @v6.bld.result(%v6.bld %bldPtr) {
  %out = call %vb.out @weld_rt_result_vb(i8* %bldPtr)
  %bytes = extractvalue %vb.out %out, 0
  %size = extractvalue %vb.out %out, 1
  %elems = bitcast i8* %bytes to double*
  %1 = insertvalue %v6 undef, double* %elems, 0
  %2 = insertvalue %v6 %1, i64 %size, 1
  ret %v6 %2
}

; Get the length of a vector.
define i64 @v6.size(%v6 %vec) {
  %size = extractvalue %v6 %vec, 1
  ret i64 %size
}

; Get a pointer to the index'th element.
define double* @v6.at(%v6 %vec, i64 %index) {
  %elements = extractvalue %v6 %vec, 0
  %ptr = getelementptr double, double* %elements, i64 %index
  ret double* %ptr
}


; Get the length of a VecBuilder.
define i64 @v6.bld.size(%v6.bld %bldPtr, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  ret i64 %size
}

; Get a pointer to the index'th element of a VecBuilder.
define double* @v6.bld.at(%v6.bld %bldPtr, i64 %index, i32 %myId) {
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %bytes = extractvalue %vb.vp %curPiece, 0
  %elements = bitcast i8* %bytes to double*
  %ptr = getelementptr double, double* %elements, i64 %index
  ret double* %ptr
}

; Compute the hash code of a vector.
; TODO: We should hash more bytes at a time if elements are non-pointer types.
define i32 @v6.hash(%v6 %vec) {
entry:
  %elements = extractvalue %v6 %vec, 0
  %size = extractvalue %v6 %vec, 1
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %prevHash = phi i32 [ 0, %entry ], [ %newHash, %body ]
  %ptr = getelementptr double, double* %elements, i64 %i
  %elem = load double, double* %ptr
  %elemHash = call i32 @double.hash(double %elem)
  %newHash = call i32 @hash_combine(i32 %prevHash, i32 %elemHash)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done
  
done:
  %res = phi i32 [ 0, %entry ], [ %newHash, %body ]
  ret i32 %res
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v6.bld.hash(%v6.bld %bld) {
  ret i32 0
}

; Dummy hash function; this is needed for structs that use these vecbuilders as fields.
define i32 @v6.vm.bld.hash(%v6.vm.bld %bld) {
  ret i32 0
}

; Dummy comparison function; this is needed for structs that use these vecbuilders as fields.
define i32 @v6.bld.cmp(%v6.bld %bld1, %v6.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function; this is needed for structs that use these vecmergers as fields.
define i32 @v6.vm.bld.cmp(%v6.vm.bld %bld1, %v6.vm.bld %bld2) {
  ret i32 -1
}

; Dummy comparison function for builders.
define i1 @v6.bld.eq(%v6.bld %a, %v6.bld %b) {
  ret i1 0
}

; Dummy comparison function for builders.
define i1 @v6.vm.bld.eq(%v6.vm.bld %a, %v6.vm.bld %b) {
  ret i1 0
}

; Vector extensions for a vector, its builder type, and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - VECSIZE: Size of vectors.

; Get a pointer to the index'th element, fetching a vector
define <4 x double>* @v6.vat(%v6 %vec, i64 %index) {
  %elements = extractvalue %v6 %vec, 0
  %ptr = getelementptr double, double* %elements, i64 %index
  %retPtr = bitcast double* %ptr to <4 x double>*
  ret <4 x double>* %retPtr
}

; Append a value into a builder, growing its space if needed.
define %v6.bld @v6.bld.vmerge(%v6.bld %bldPtr, <4 x double> %value, i32 %myId) {
entry:
  %curPiecePtr = call %vb.vp* @weld_rt_cur_vb_piece(i8* %bldPtr, i32 %myId)
  %curPiece = load %vb.vp, %vb.vp* %curPiecePtr
  %size = extractvalue %vb.vp %curPiece, 1
  %capacity = extractvalue %vb.vp %curPiece, 2
  %adjusted = add i64 %size, 4
  %full = icmp sgt i64 %adjusted, %capacity
  br i1 %full, label %onFull, label %finish
  
onFull:
  %newCapacity = mul i64 %capacity, 2
  %elemSizePtr = getelementptr double, double* null, i32 1
  %elemSize = ptrtoint double* %elemSizePtr to i64
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
  %elements = bitcast i8* %bytes1 to double*
  %insertPtr = getelementptr double, double* %elements, i64 %size
  %vecInsertPtr = bitcast double* %insertPtr to <4 x double>*
  store <4 x double> %value, <4 x double>* %vecInsertPtr, align 1
  %newSize = add i64 %size, 4
  %curPiece4 = insertvalue %vb.vp %curPiece3, i64 %newSize, 1
  store %vb.vp %curPiece4, %vb.vp* %curPiecePtr
  ret %v6.bld %bldPtr
}

; Vector extension to generate comparison functions. We use a different file,
; vector_comparisons_i8.ll, for comparisons on arrays of bytes, which are optimized.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)



; Compare two vectors lexicographically.
define i32 @v6.cmp(%v6 %a, %v6 %b) {
entry:
  %elemsA = extractvalue %v6 %a, 0
  %elemsB = extractvalue %v6 %b, 0
  %sizeA = extractvalue %v6 %a, 1
  %sizeB = extractvalue %v6 %b, 1
  %cond1 = icmp ult i64 %sizeA, %sizeB
  %minSize = select i1 %cond1, i64 %sizeA, i64 %sizeB
  %cond = icmp ult i64 0, %minSize
  br i1 %cond, label %body, label %done
  
body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body2 ]
  %ptrA = getelementptr double, double* %elemsA, i64 %i
  %ptrB = getelementptr double, double* %elemsB, i64 %i
  %elemA = load double, double* %ptrA
  %elemB = load double, double* %ptrB
  %cmp = call i32 @double.cmp(double %elemA, double %elemB)
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

; Compare two vectors for equality.
define i1 @v6.eq(%v6 %a, %v6 %b) {
  %sizeA = extractvalue %v6 %a, 1
  %sizeB = extractvalue %v6 %b, 1
  %cond = icmp eq i64 %sizeA, %sizeB
  br i1 %cond, label %same_size, label %different_size
same_size:
  %cmp = call i32 @v6.cmp(%v6 %a, %v6 %b)
  %res = icmp eq i32 %cmp, 0
  ret i1 %res
different_size:
  ret i1 0
}
%s6 = type { %v3.bld, %v6.bld, %v3.bld, %v5.bld }
define i32 @s6.hash(%s6 %value) {
  %p.p200 = extractvalue %s6 %value, 0
  %p.p201 = call i32 @v3.bld.hash(%v3.bld %p.p200)
  %p.p202 = call i32 @hash_combine(i32 0, i32 %p.p201)
  %p.p203 = extractvalue %s6 %value, 1
  %p.p204 = call i32 @v6.bld.hash(%v6.bld %p.p203)
  %p.p205 = call i32 @hash_combine(i32 %p.p202, i32 %p.p204)
  %p.p206 = extractvalue %s6 %value, 2
  %p.p207 = call i32 @v3.bld.hash(%v3.bld %p.p206)
  %p.p208 = call i32 @hash_combine(i32 %p.p205, i32 %p.p207)
  %p.p209 = extractvalue %s6 %value, 3
  %p.p210 = call i32 @v5.bld.hash(%v5.bld %p.p209)
  %p.p211 = call i32 @hash_combine(i32 %p.p208, i32 %p.p210)
  ret i32 %p.p211
}

define i32 @s6.cmp(%s6 %a, %s6 %b) {
  %p.p212 = extractvalue %s6 %a , 0
  %p.p213 = extractvalue %s6 %b, 0
  %p.p214 = call i32 @v3.bld.cmp(%v3.bld %p.p212, %v3.bld %p.p213)
  %p.p215 = icmp ne i32 %p.p214, 0
  br i1 %p.p215, label %l0, label %l1
l0:
  ret i32 %p.p214
l1:
  %p.p216 = extractvalue %s6 %a , 1
  %p.p217 = extractvalue %s6 %b, 1
  %p.p218 = call i32 @v6.bld.cmp(%v6.bld %p.p216, %v6.bld %p.p217)
  %p.p219 = icmp ne i32 %p.p218, 0
  br i1 %p.p219, label %l2, label %l3
l2:
  ret i32 %p.p218
l3:
  %p.p220 = extractvalue %s6 %a , 2
  %p.p221 = extractvalue %s6 %b, 2
  %p.p222 = call i32 @v3.bld.cmp(%v3.bld %p.p220, %v3.bld %p.p221)
  %p.p223 = icmp ne i32 %p.p222, 0
  br i1 %p.p223, label %l4, label %l5
l4:
  ret i32 %p.p222
l5:
  %p.p224 = extractvalue %s6 %a , 3
  %p.p225 = extractvalue %s6 %b, 3
  %p.p226 = call i32 @v5.bld.cmp(%v5.bld %p.p224, %v5.bld %p.p225)
  %p.p227 = icmp ne i32 %p.p226, 0
  br i1 %p.p227, label %l6, label %l7
l6:
  ret i32 %p.p226
l7:
  ret i32 0
}

define i1 @s6.eq(%s6 %a, %s6 %b) {
  %p.p228 = extractvalue %s6 %a , 0
  %p.p229 = extractvalue %s6 %b, 0
  %p.p230 = call i1 @v3.bld.eq(%v3.bld %p.p228, %v3.bld %p.p229)
  br i1 %p.p230, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p231 = extractvalue %s6 %a , 1
  %p.p232 = extractvalue %s6 %b, 1
  %p.p233 = call i1 @v6.bld.eq(%v6.bld %p.p231, %v6.bld %p.p232)
  br i1 %p.p233, label %l3, label %l2
l2:
  ret i1 0
l3:
  %p.p234 = extractvalue %s6 %a , 2
  %p.p235 = extractvalue %s6 %b, 2
  %p.p236 = call i1 @v3.bld.eq(%v3.bld %p.p234, %v3.bld %p.p235)
  br i1 %p.p236, label %l5, label %l4
l4:
  ret i1 0
l5:
  %p.p237 = extractvalue %s6 %a , 3
  %p.p238 = extractvalue %s6 %b, 3
  %p.p239 = call i1 @v5.bld.eq(%v5.bld %p.p237, %v5.bld %p.p238)
  br i1 %p.p239, label %l7, label %l6
l6:
  ret i1 0
l7:
  ret i1 1
}

%s7 = type { %v3, %v4, %v4, %v0, %v3, %v6, %v3, %v5 }
define i32 @s7.hash(%s7 %value) {
  %p.p240 = extractvalue %s7 %value, 0
  %p.p241 = call i32 @v3.hash(%v3 %p.p240)
  %p.p242 = call i32 @hash_combine(i32 0, i32 %p.p241)
  %p.p243 = extractvalue %s7 %value, 1
  %p.p244 = call i32 @v4.hash(%v4 %p.p243)
  %p.p245 = call i32 @hash_combine(i32 %p.p242, i32 %p.p244)
  %p.p246 = extractvalue %s7 %value, 2
  %p.p247 = call i32 @v4.hash(%v4 %p.p246)
  %p.p248 = call i32 @hash_combine(i32 %p.p245, i32 %p.p247)
  %p.p249 = extractvalue %s7 %value, 3
  %p.p250 = call i32 @v0.hash(%v0 %p.p249)
  %p.p251 = call i32 @hash_combine(i32 %p.p248, i32 %p.p250)
  %p.p252 = extractvalue %s7 %value, 4
  %p.p253 = call i32 @v3.hash(%v3 %p.p252)
  %p.p254 = call i32 @hash_combine(i32 %p.p251, i32 %p.p253)
  %p.p255 = extractvalue %s7 %value, 5
  %p.p256 = call i32 @v6.hash(%v6 %p.p255)
  %p.p257 = call i32 @hash_combine(i32 %p.p254, i32 %p.p256)
  %p.p258 = extractvalue %s7 %value, 6
  %p.p259 = call i32 @v3.hash(%v3 %p.p258)
  %p.p260 = call i32 @hash_combine(i32 %p.p257, i32 %p.p259)
  %p.p261 = extractvalue %s7 %value, 7
  %p.p262 = call i32 @v5.hash(%v5 %p.p261)
  %p.p263 = call i32 @hash_combine(i32 %p.p260, i32 %p.p262)
  ret i32 %p.p263
}

define i32 @s7.cmp(%s7 %a, %s7 %b) {
  %p.p264 = extractvalue %s7 %a , 0
  %p.p265 = extractvalue %s7 %b, 0
  %p.p266 = call i32 @v3.cmp(%v3 %p.p264, %v3 %p.p265)
  %p.p267 = icmp ne i32 %p.p266, 0
  br i1 %p.p267, label %l0, label %l1
l0:
  ret i32 %p.p266
l1:
  %p.p268 = extractvalue %s7 %a , 1
  %p.p269 = extractvalue %s7 %b, 1
  %p.p270 = call i32 @v4.cmp(%v4 %p.p268, %v4 %p.p269)
  %p.p271 = icmp ne i32 %p.p270, 0
  br i1 %p.p271, label %l2, label %l3
l2:
  ret i32 %p.p270
l3:
  %p.p272 = extractvalue %s7 %a , 2
  %p.p273 = extractvalue %s7 %b, 2
  %p.p274 = call i32 @v4.cmp(%v4 %p.p272, %v4 %p.p273)
  %p.p275 = icmp ne i32 %p.p274, 0
  br i1 %p.p275, label %l4, label %l5
l4:
  ret i32 %p.p274
l5:
  %p.p276 = extractvalue %s7 %a , 3
  %p.p277 = extractvalue %s7 %b, 3
  %p.p278 = call i32 @v0.cmp(%v0 %p.p276, %v0 %p.p277)
  %p.p279 = icmp ne i32 %p.p278, 0
  br i1 %p.p279, label %l6, label %l7
l6:
  ret i32 %p.p278
l7:
  %p.p280 = extractvalue %s7 %a , 4
  %p.p281 = extractvalue %s7 %b, 4
  %p.p282 = call i32 @v3.cmp(%v3 %p.p280, %v3 %p.p281)
  %p.p283 = icmp ne i32 %p.p282, 0
  br i1 %p.p283, label %l8, label %l9
l8:
  ret i32 %p.p282
l9:
  %p.p284 = extractvalue %s7 %a , 5
  %p.p285 = extractvalue %s7 %b, 5
  %p.p286 = call i32 @v6.cmp(%v6 %p.p284, %v6 %p.p285)
  %p.p287 = icmp ne i32 %p.p286, 0
  br i1 %p.p287, label %l10, label %l11
l10:
  ret i32 %p.p286
l11:
  %p.p288 = extractvalue %s7 %a , 6
  %p.p289 = extractvalue %s7 %b, 6
  %p.p290 = call i32 @v3.cmp(%v3 %p.p288, %v3 %p.p289)
  %p.p291 = icmp ne i32 %p.p290, 0
  br i1 %p.p291, label %l12, label %l13
l12:
  ret i32 %p.p290
l13:
  %p.p292 = extractvalue %s7 %a , 7
  %p.p293 = extractvalue %s7 %b, 7
  %p.p294 = call i32 @v5.cmp(%v5 %p.p292, %v5 %p.p293)
  %p.p295 = icmp ne i32 %p.p294, 0
  br i1 %p.p295, label %l14, label %l15
l14:
  ret i32 %p.p294
l15:
  ret i32 0
}

define i1 @s7.eq(%s7 %a, %s7 %b) {
  %p.p296 = extractvalue %s7 %a , 0
  %p.p297 = extractvalue %s7 %b, 0
  %p.p298 = call i1 @v3.eq(%v3 %p.p296, %v3 %p.p297)
  br i1 %p.p298, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p299 = extractvalue %s7 %a , 1
  %p.p300 = extractvalue %s7 %b, 1
  %p.p301 = call i1 @v4.eq(%v4 %p.p299, %v4 %p.p300)
  br i1 %p.p301, label %l3, label %l2
l2:
  ret i1 0
l3:
  %p.p302 = extractvalue %s7 %a , 2
  %p.p303 = extractvalue %s7 %b, 2
  %p.p304 = call i1 @v4.eq(%v4 %p.p302, %v4 %p.p303)
  br i1 %p.p304, label %l5, label %l4
l4:
  ret i1 0
l5:
  %p.p305 = extractvalue %s7 %a , 3
  %p.p306 = extractvalue %s7 %b, 3
  %p.p307 = call i1 @v0.eq(%v0 %p.p305, %v0 %p.p306)
  br i1 %p.p307, label %l7, label %l6
l6:
  ret i1 0
l7:
  %p.p308 = extractvalue %s7 %a , 4
  %p.p309 = extractvalue %s7 %b, 4
  %p.p310 = call i1 @v3.eq(%v3 %p.p308, %v3 %p.p309)
  br i1 %p.p310, label %l9, label %l8
l8:
  ret i1 0
l9:
  %p.p311 = extractvalue %s7 %a , 5
  %p.p312 = extractvalue %s7 %b, 5
  %p.p313 = call i1 @v6.eq(%v6 %p.p311, %v6 %p.p312)
  br i1 %p.p313, label %l11, label %l10
l10:
  ret i1 0
l11:
  %p.p314 = extractvalue %s7 %a , 6
  %p.p315 = extractvalue %s7 %b, 6
  %p.p316 = call i1 @v3.eq(%v3 %p.p314, %v3 %p.p315)
  br i1 %p.p316, label %l13, label %l12
l12:
  ret i1 0
l13:
  %p.p317 = extractvalue %s7 %a , 7
  %p.p318 = extractvalue %s7 %b, 7
  %p.p319 = call i1 @v5.eq(%v5 %p.p317, %v5 %p.p318)
  br i1 %p.p319, label %l15, label %l14
l14:
  ret i1 0
l15:
  ret i1 1
}

%s8 = type { %s6, %v2 }
define i32 @s8.hash(%s8 %value) {
  %p.p320 = extractvalue %s8 %value, 0
  %p.p321 = call i32 @s6.hash(%s6 %p.p320)
  %p.p322 = call i32 @hash_combine(i32 0, i32 %p.p321)
  %p.p323 = extractvalue %s8 %value, 1
  %p.p324 = call i32 @v2.hash(%v2 %p.p323)
  %p.p325 = call i32 @hash_combine(i32 %p.p322, i32 %p.p324)
  ret i32 %p.p325
}

define i32 @s8.cmp(%s8 %a, %s8 %b) {
  %p.p326 = extractvalue %s8 %a , 0
  %p.p327 = extractvalue %s8 %b, 0
  %p.p328 = call i32 @s6.cmp(%s6 %p.p326, %s6 %p.p327)
  %p.p329 = icmp ne i32 %p.p328, 0
  br i1 %p.p329, label %l0, label %l1
l0:
  ret i32 %p.p328
l1:
  %p.p330 = extractvalue %s8 %a , 1
  %p.p331 = extractvalue %s8 %b, 1
  %p.p332 = call i32 @v2.cmp(%v2 %p.p330, %v2 %p.p331)
  %p.p333 = icmp ne i32 %p.p332, 0
  br i1 %p.p333, label %l2, label %l3
l2:
  ret i32 %p.p332
l3:
  ret i32 0
}

define i1 @s8.eq(%s8 %a, %s8 %b) {
  %p.p334 = extractvalue %s8 %a , 0
  %p.p335 = extractvalue %s8 %b, 0
  %p.p336 = call i1 @s6.eq(%s6 %p.p334, %s6 %p.p335)
  br i1 %p.p336, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p337 = extractvalue %s8 %a , 1
  %p.p338 = extractvalue %s8 %b, 1
  %p.p339 = call i1 @v2.eq(%v2 %p.p337, %v2 %p.p338)
  br i1 %p.p339, label %l3, label %l2
l2:
  ret i1 0
l3:
  ret i1 1
}

%s9 = type { %s4, %s6 }
define i32 @s9.hash(%s9 %value) {
  %p.p340 = extractvalue %s9 %value, 0
  %p.p341 = call i32 @s4.hash(%s4 %p.p340)
  %p.p342 = call i32 @hash_combine(i32 0, i32 %p.p341)
  %p.p343 = extractvalue %s9 %value, 1
  %p.p344 = call i32 @s6.hash(%s6 %p.p343)
  %p.p345 = call i32 @hash_combine(i32 %p.p342, i32 %p.p344)
  ret i32 %p.p345
}

define i32 @s9.cmp(%s9 %a, %s9 %b) {
  %p.p346 = extractvalue %s9 %a , 0
  %p.p347 = extractvalue %s9 %b, 0
  %p.p348 = call i32 @s4.cmp(%s4 %p.p346, %s4 %p.p347)
  %p.p349 = icmp ne i32 %p.p348, 0
  br i1 %p.p349, label %l0, label %l1
l0:
  ret i32 %p.p348
l1:
  %p.p350 = extractvalue %s9 %a , 1
  %p.p351 = extractvalue %s9 %b, 1
  %p.p352 = call i32 @s6.cmp(%s6 %p.p350, %s6 %p.p351)
  %p.p353 = icmp ne i32 %p.p352, 0
  br i1 %p.p353, label %l2, label %l3
l2:
  ret i32 %p.p352
l3:
  ret i32 0
}

define i1 @s9.eq(%s9 %a, %s9 %b) {
  %p.p354 = extractvalue %s9 %a , 0
  %p.p355 = extractvalue %s9 %b, 0
  %p.p356 = call i1 @s4.eq(%s4 %p.p354, %s4 %p.p355)
  br i1 %p.p356, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p357 = extractvalue %s9 %a , 1
  %p.p358 = extractvalue %s9 %b, 1
  %p.p359 = call i1 @s6.eq(%s6 %p.p357, %s6 %p.p358)
  br i1 %p.p359, label %l3, label %l2
l2:
  ret i1 0
l3:
  ret i1 1
}

%s10 = type { %v0.bld, %v0 }
define i32 @s10.hash(%s10 %value) {
  %p.p360 = extractvalue %s10 %value, 0
  %p.p361 = call i32 @v0.bld.hash(%v0.bld %p.p360)
  %p.p362 = call i32 @hash_combine(i32 0, i32 %p.p361)
  %p.p363 = extractvalue %s10 %value, 1
  %p.p364 = call i32 @v0.hash(%v0 %p.p363)
  %p.p365 = call i32 @hash_combine(i32 %p.p362, i32 %p.p364)
  ret i32 %p.p365
}

define i32 @s10.cmp(%s10 %a, %s10 %b) {
  %p.p366 = extractvalue %s10 %a , 0
  %p.p367 = extractvalue %s10 %b, 0
  %p.p368 = call i32 @v0.bld.cmp(%v0.bld %p.p366, %v0.bld %p.p367)
  %p.p369 = icmp ne i32 %p.p368, 0
  br i1 %p.p369, label %l0, label %l1
l0:
  ret i32 %p.p368
l1:
  %p.p370 = extractvalue %s10 %a , 1
  %p.p371 = extractvalue %s10 %b, 1
  %p.p372 = call i32 @v0.cmp(%v0 %p.p370, %v0 %p.p371)
  %p.p373 = icmp ne i32 %p.p372, 0
  br i1 %p.p373, label %l2, label %l3
l2:
  ret i32 %p.p372
l3:
  ret i32 0
}

define i1 @s10.eq(%s10 %a, %s10 %b) {
  %p.p374 = extractvalue %s10 %a , 0
  %p.p375 = extractvalue %s10 %b, 0
  %p.p376 = call i1 @v0.bld.eq(%v0.bld %p.p374, %v0.bld %p.p375)
  br i1 %p.p376, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p377 = extractvalue %s10 %a , 1
  %p.p378 = extractvalue %s10 %b, 1
  %p.p379 = call i1 @v0.eq(%v0 %p.p377, %v0 %p.p378)
  br i1 %p.p379, label %l3, label %l2
l2:
  ret i1 0
l3:
  ret i1 1
}

%s11 = type { %s4, i64, %v3.bld, i32, %v4.bld, %v4.bld, %v0.bld, %v2 }
define i32 @s11.hash(%s11 %value) {
  %p.p380 = extractvalue %s11 %value, 0
  %p.p381 = call i32 @s4.hash(%s4 %p.p380)
  %p.p382 = call i32 @hash_combine(i32 0, i32 %p.p381)
  %p.p383 = extractvalue %s11 %value, 1
  %p.p384 = call i32 @i64.hash(i64 %p.p383)
  %p.p385 = call i32 @hash_combine(i32 %p.p382, i32 %p.p384)
  %p.p386 = extractvalue %s11 %value, 2
  %p.p387 = call i32 @v3.bld.hash(%v3.bld %p.p386)
  %p.p388 = call i32 @hash_combine(i32 %p.p385, i32 %p.p387)
  %p.p389 = extractvalue %s11 %value, 3
  %p.p390 = call i32 @i32.hash(i32 %p.p389)
  %p.p391 = call i32 @hash_combine(i32 %p.p388, i32 %p.p390)
  %p.p392 = extractvalue %s11 %value, 4
  %p.p393 = call i32 @v4.bld.hash(%v4.bld %p.p392)
  %p.p394 = call i32 @hash_combine(i32 %p.p391, i32 %p.p393)
  %p.p395 = extractvalue %s11 %value, 5
  %p.p396 = call i32 @v4.bld.hash(%v4.bld %p.p395)
  %p.p397 = call i32 @hash_combine(i32 %p.p394, i32 %p.p396)
  %p.p398 = extractvalue %s11 %value, 6
  %p.p399 = call i32 @v0.bld.hash(%v0.bld %p.p398)
  %p.p400 = call i32 @hash_combine(i32 %p.p397, i32 %p.p399)
  %p.p401 = extractvalue %s11 %value, 7
  %p.p402 = call i32 @v2.hash(%v2 %p.p401)
  %p.p403 = call i32 @hash_combine(i32 %p.p400, i32 %p.p402)
  ret i32 %p.p403
}

define i32 @s11.cmp(%s11 %a, %s11 %b) {
  %p.p404 = extractvalue %s11 %a , 0
  %p.p405 = extractvalue %s11 %b, 0
  %p.p406 = call i32 @s4.cmp(%s4 %p.p404, %s4 %p.p405)
  %p.p407 = icmp ne i32 %p.p406, 0
  br i1 %p.p407, label %l0, label %l1
l0:
  ret i32 %p.p406
l1:
  %p.p408 = extractvalue %s11 %a , 1
  %p.p409 = extractvalue %s11 %b, 1
  %p.p410 = call i32 @i64.cmp(i64 %p.p408, i64 %p.p409)
  %p.p411 = icmp ne i32 %p.p410, 0
  br i1 %p.p411, label %l2, label %l3
l2:
  ret i32 %p.p410
l3:
  %p.p412 = extractvalue %s11 %a , 2
  %p.p413 = extractvalue %s11 %b, 2
  %p.p414 = call i32 @v3.bld.cmp(%v3.bld %p.p412, %v3.bld %p.p413)
  %p.p415 = icmp ne i32 %p.p414, 0
  br i1 %p.p415, label %l4, label %l5
l4:
  ret i32 %p.p414
l5:
  %p.p416 = extractvalue %s11 %a , 3
  %p.p417 = extractvalue %s11 %b, 3
  %p.p418 = call i32 @i32.cmp(i32 %p.p416, i32 %p.p417)
  %p.p419 = icmp ne i32 %p.p418, 0
  br i1 %p.p419, label %l6, label %l7
l6:
  ret i32 %p.p418
l7:
  %p.p420 = extractvalue %s11 %a , 4
  %p.p421 = extractvalue %s11 %b, 4
  %p.p422 = call i32 @v4.bld.cmp(%v4.bld %p.p420, %v4.bld %p.p421)
  %p.p423 = icmp ne i32 %p.p422, 0
  br i1 %p.p423, label %l8, label %l9
l8:
  ret i32 %p.p422
l9:
  %p.p424 = extractvalue %s11 %a , 5
  %p.p425 = extractvalue %s11 %b, 5
  %p.p426 = call i32 @v4.bld.cmp(%v4.bld %p.p424, %v4.bld %p.p425)
  %p.p427 = icmp ne i32 %p.p426, 0
  br i1 %p.p427, label %l10, label %l11
l10:
  ret i32 %p.p426
l11:
  %p.p428 = extractvalue %s11 %a , 6
  %p.p429 = extractvalue %s11 %b, 6
  %p.p430 = call i32 @v0.bld.cmp(%v0.bld %p.p428, %v0.bld %p.p429)
  %p.p431 = icmp ne i32 %p.p430, 0
  br i1 %p.p431, label %l12, label %l13
l12:
  ret i32 %p.p430
l13:
  %p.p432 = extractvalue %s11 %a , 7
  %p.p433 = extractvalue %s11 %b, 7
  %p.p434 = call i32 @v2.cmp(%v2 %p.p432, %v2 %p.p433)
  %p.p435 = icmp ne i32 %p.p434, 0
  br i1 %p.p435, label %l14, label %l15
l14:
  ret i32 %p.p434
l15:
  ret i32 0
}

define i1 @s11.eq(%s11 %a, %s11 %b) {
  %p.p436 = extractvalue %s11 %a , 0
  %p.p437 = extractvalue %s11 %b, 0
  %p.p438 = call i1 @s4.eq(%s4 %p.p436, %s4 %p.p437)
  br i1 %p.p438, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p439 = extractvalue %s11 %a , 1
  %p.p440 = extractvalue %s11 %b, 1
  %p.p441 = call i1 @i64.eq(i64 %p.p439, i64 %p.p440)
  br i1 %p.p441, label %l3, label %l2
l2:
  ret i1 0
l3:
  %p.p442 = extractvalue %s11 %a , 2
  %p.p443 = extractvalue %s11 %b, 2
  %p.p444 = call i1 @v3.bld.eq(%v3.bld %p.p442, %v3.bld %p.p443)
  br i1 %p.p444, label %l5, label %l4
l4:
  ret i1 0
l5:
  %p.p445 = extractvalue %s11 %a , 3
  %p.p446 = extractvalue %s11 %b, 3
  %p.p447 = call i1 @i32.eq(i32 %p.p445, i32 %p.p446)
  br i1 %p.p447, label %l7, label %l6
l6:
  ret i1 0
l7:
  %p.p448 = extractvalue %s11 %a , 4
  %p.p449 = extractvalue %s11 %b, 4
  %p.p450 = call i1 @v4.bld.eq(%v4.bld %p.p448, %v4.bld %p.p449)
  br i1 %p.p450, label %l9, label %l8
l8:
  ret i1 0
l9:
  %p.p451 = extractvalue %s11 %a , 5
  %p.p452 = extractvalue %s11 %b, 5
  %p.p453 = call i1 @v4.bld.eq(%v4.bld %p.p451, %v4.bld %p.p452)
  br i1 %p.p453, label %l11, label %l10
l10:
  ret i1 0
l11:
  %p.p454 = extractvalue %s11 %a , 6
  %p.p455 = extractvalue %s11 %b, 6
  %p.p456 = call i1 @v0.bld.eq(%v0.bld %p.p454, %v0.bld %p.p455)
  br i1 %p.p456, label %l13, label %l12
l12:
  ret i1 0
l13:
  %p.p457 = extractvalue %s11 %a , 7
  %p.p458 = extractvalue %s11 %b, 7
  %p.p459 = call i1 @v2.eq(%v2 %p.p457, %v2 %p.p458)
  br i1 %p.p459, label %l15, label %l14
l14:
  ret i1 0
l15:
  ret i1 1
}

%s12 = type { %v2.bld, %v1 }
define i32 @s12.hash(%s12 %value) {
  %p.p460 = extractvalue %s12 %value, 0
  %p.p461 = call i32 @v2.bld.hash(%v2.bld %p.p460)
  %p.p462 = call i32 @hash_combine(i32 0, i32 %p.p461)
  %p.p463 = extractvalue %s12 %value, 1
  %p.p464 = call i32 @v1.hash(%v1 %p.p463)
  %p.p465 = call i32 @hash_combine(i32 %p.p462, i32 %p.p464)
  ret i32 %p.p465
}

define i32 @s12.cmp(%s12 %a, %s12 %b) {
  %p.p466 = extractvalue %s12 %a , 0
  %p.p467 = extractvalue %s12 %b, 0
  %p.p468 = call i32 @v2.bld.cmp(%v2.bld %p.p466, %v2.bld %p.p467)
  %p.p469 = icmp ne i32 %p.p468, 0
  br i1 %p.p469, label %l0, label %l1
l0:
  ret i32 %p.p468
l1:
  %p.p470 = extractvalue %s12 %a , 1
  %p.p471 = extractvalue %s12 %b, 1
  %p.p472 = call i32 @v1.cmp(%v1 %p.p470, %v1 %p.p471)
  %p.p473 = icmp ne i32 %p.p472, 0
  br i1 %p.p473, label %l2, label %l3
l2:
  ret i32 %p.p472
l3:
  ret i32 0
}

define i1 @s12.eq(%s12 %a, %s12 %b) {
  %p.p474 = extractvalue %s12 %a , 0
  %p.p475 = extractvalue %s12 %b, 0
  %p.p476 = call i1 @v2.bld.eq(%v2.bld %p.p474, %v2.bld %p.p475)
  br i1 %p.p476, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p477 = extractvalue %s12 %a , 1
  %p.p478 = extractvalue %s12 %b, 1
  %p.p479 = call i1 @v1.eq(%v1 %p.p477, %v1 %p.p478)
  br i1 %p.p479, label %l3, label %l2
l2:
  ret i1 0
l3:
  ret i1 1
}

%s13 = type { %v2.bld }
define i32 @s13.hash(%s13 %value) {
  %p.p480 = extractvalue %s13 %value, 0
  %p.p481 = call i32 @v2.bld.hash(%v2.bld %p.p480)
  %p.p482 = call i32 @hash_combine(i32 0, i32 %p.p481)
  ret i32 %p.p482
}

define i32 @s13.cmp(%s13 %a, %s13 %b) {
  %p.p483 = extractvalue %s13 %a , 0
  %p.p484 = extractvalue %s13 %b, 0
  %p.p485 = call i32 @v2.bld.cmp(%v2.bld %p.p483, %v2.bld %p.p484)
  %p.p486 = icmp ne i32 %p.p485, 0
  br i1 %p.p486, label %l0, label %l1
l0:
  ret i32 %p.p485
l1:
  ret i32 0
}

define i1 @s13.eq(%s13 %a, %s13 %b) {
  %p.p487 = extractvalue %s13 %a , 0
  %p.p488 = extractvalue %s13 %b, 0
  %p.p489 = call i1 @v2.bld.eq(%v2.bld %p.p487, %v2.bld %p.p488)
  br i1 %p.p489, label %l1, label %l0
l0:
  ret i1 0
l1:
  ret i1 1
}

%s14 = type { %v1, i32 }
define i32 @s14.hash(%s14 %value) {
  %p.p490 = extractvalue %s14 %value, 0
  %p.p491 = call i32 @v1.hash(%v1 %p.p490)
  %p.p492 = call i32 @hash_combine(i32 0, i32 %p.p491)
  %p.p493 = extractvalue %s14 %value, 1
  %p.p494 = call i32 @i32.hash(i32 %p.p493)
  %p.p495 = call i32 @hash_combine(i32 %p.p492, i32 %p.p494)
  ret i32 %p.p495
}

define i32 @s14.cmp(%s14 %a, %s14 %b) {
  %p.p496 = extractvalue %s14 %a , 0
  %p.p497 = extractvalue %s14 %b, 0
  %p.p498 = call i32 @v1.cmp(%v1 %p.p496, %v1 %p.p497)
  %p.p499 = icmp ne i32 %p.p498, 0
  br i1 %p.p499, label %l0, label %l1
l0:
  ret i32 %p.p498
l1:
  %p.p500 = extractvalue %s14 %a , 1
  %p.p501 = extractvalue %s14 %b, 1
  %p.p502 = call i32 @i32.cmp(i32 %p.p500, i32 %p.p501)
  %p.p503 = icmp ne i32 %p.p502, 0
  br i1 %p.p503, label %l2, label %l3
l2:
  ret i32 %p.p502
l3:
  ret i32 0
}

define i1 @s14.eq(%s14 %a, %s14 %b) {
  %p.p504 = extractvalue %s14 %a , 0
  %p.p505 = extractvalue %s14 %b, 0
  %p.p506 = call i1 @v1.eq(%v1 %p.p504, %v1 %p.p505)
  br i1 %p.p506, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p507 = extractvalue %s14 %a , 1
  %p.p508 = extractvalue %s14 %b, 1
  %p.p509 = call i1 @i32.eq(i32 %p.p507, i32 %p.p508)
  br i1 %p.p509, label %l3, label %l2
l2:
  ret i1 0
l3:
  ret i1 1
}

%s15 = type { i32, %v1 }
define i32 @s15.hash(%s15 %value) {
  %p.p510 = extractvalue %s15 %value, 0
  %p.p511 = call i32 @i32.hash(i32 %p.p510)
  %p.p512 = call i32 @hash_combine(i32 0, i32 %p.p511)
  %p.p513 = extractvalue %s15 %value, 1
  %p.p514 = call i32 @v1.hash(%v1 %p.p513)
  %p.p515 = call i32 @hash_combine(i32 %p.p512, i32 %p.p514)
  ret i32 %p.p515
}

define i32 @s15.cmp(%s15 %a, %s15 %b) {
  %p.p516 = extractvalue %s15 %a , 0
  %p.p517 = extractvalue %s15 %b, 0
  %p.p518 = call i32 @i32.cmp(i32 %p.p516, i32 %p.p517)
  %p.p519 = icmp ne i32 %p.p518, 0
  br i1 %p.p519, label %l0, label %l1
l0:
  ret i32 %p.p518
l1:
  %p.p520 = extractvalue %s15 %a , 1
  %p.p521 = extractvalue %s15 %b, 1
  %p.p522 = call i32 @v1.cmp(%v1 %p.p520, %v1 %p.p521)
  %p.p523 = icmp ne i32 %p.p522, 0
  br i1 %p.p523, label %l2, label %l3
l2:
  ret i32 %p.p522
l3:
  ret i32 0
}

define i1 @s15.eq(%s15 %a, %s15 %b) {
  %p.p524 = extractvalue %s15 %a , 0
  %p.p525 = extractvalue %s15 %b, 0
  %p.p526 = call i1 @i32.eq(i32 %p.p524, i32 %p.p525)
  br i1 %p.p526, label %l1, label %l0
l0:
  ret i1 0
l1:
  %p.p527 = extractvalue %s15 %a , 1
  %p.p528 = extractvalue %s15 %b, 1
  %p.p529 = call i1 @v1.eq(%v1 %p.p527, %v1 %p.p528)
  br i1 %p.p529, label %l3, label %l2
l2:
  ret i1 0
l3:
  ret i1 1
}


; BODY:

define void @f9(%s4 %buffer.in, %s6 %fn7_tmp.5.in, %work_t* %cur.work) {
fn.entry:
  %buffer = alloca %s4
  %fn7_tmp.5 = alloca %s6
  %vectors = alloca %s6
  %fn9_tmp.15 = alloca %v5
  %fn9_tmp.3 = alloca %v4
  %fn9_tmp.16 = alloca %s7
  %fn9_tmp.7 = alloca %v0
  %fn9_tmp.12 = alloca %v3.bld
  %fn9_tmp.8 = alloca %v3.bld
  %fn9_tmp.6 = alloca %v0.bld
  %fn9_tmp.10 = alloca %v6.bld
  %fn9_tmp.2 = alloca %v4.bld
  %fn9_tmp.1 = alloca %v3
  %fn9_tmp.13 = alloca %v3
  %fn9_tmp.5 = alloca %v4
  %fn9_tmp.14 = alloca %v5.bld
  %fn9_tmp.9 = alloca %v3
  %fn9_tmp = alloca %v3.bld
  %fn9_tmp.4 = alloca %v4.bld
  %fn9_tmp.11 = alloca %v6
  store %s4 %buffer.in, %s4* %buffer
  store %s6 %fn7_tmp.5.in, %s6* %fn7_tmp.5
  %cur.tid = call i32 @weld_rt_thread_id()
  br label %b.b0
b.b0:
  ; vectors = fn7_tmp#5
  %t.t0 = load %s6, %s6* %fn7_tmp.5
  store %s6 %t.t0, %s6* %vectors
  ; fn9_tmp = buffer.$1
  %t.t1 = getelementptr inbounds %s4, %s4* %buffer, i32 0, i32 1
  %t.t2 = load %v3.bld, %v3.bld* %t.t1
  store %v3.bld %t.t2, %v3.bld* %fn9_tmp
  ; fn9_tmp#1 = result(fn9_tmp)
  %t.t3 = load %v3.bld, %v3.bld* %fn9_tmp
  %t.t4 = call %v3 @v3.bld.result(%v3.bld %t.t3)
  store %v3 %t.t4, %v3* %fn9_tmp.1
  ; fn9_tmp#2 = buffer.$3
  %t.t5 = getelementptr inbounds %s4, %s4* %buffer, i32 0, i32 3
  %t.t6 = load %v4.bld, %v4.bld* %t.t5
  store %v4.bld %t.t6, %v4.bld* %fn9_tmp.2
  ; fn9_tmp#3 = result(fn9_tmp#2)
  %t.t7 = load %v4.bld, %v4.bld* %fn9_tmp.2
  %t.t8 = call %v4 @v4.bld.result(%v4.bld %t.t7)
  store %v4 %t.t8, %v4* %fn9_tmp.3
  ; fn9_tmp#4 = buffer.$4
  %t.t9 = getelementptr inbounds %s4, %s4* %buffer, i32 0, i32 4
  %t.t10 = load %v4.bld, %v4.bld* %t.t9
  store %v4.bld %t.t10, %v4.bld* %fn9_tmp.4
  ; fn9_tmp#5 = result(fn9_tmp#4)
  %t.t11 = load %v4.bld, %v4.bld* %fn9_tmp.4
  %t.t12 = call %v4 @v4.bld.result(%v4.bld %t.t11)
  store %v4 %t.t12, %v4* %fn9_tmp.5
  ; fn9_tmp#6 = buffer.$5
  %t.t13 = getelementptr inbounds %s4, %s4* %buffer, i32 0, i32 5
  %t.t14 = load %v0.bld, %v0.bld* %t.t13
  store %v0.bld %t.t14, %v0.bld* %fn9_tmp.6
  ; fn9_tmp#7 = result(fn9_tmp#6)
  %t.t15 = load %v0.bld, %v0.bld* %fn9_tmp.6
  %t.t16 = call %v0 @v0.bld.result(%v0.bld %t.t15)
  store %v0 %t.t16, %v0* %fn9_tmp.7
  ; fn9_tmp#8 = vectors.$0
  %t.t17 = getelementptr inbounds %s6, %s6* %vectors, i32 0, i32 0
  %t.t18 = load %v3.bld, %v3.bld* %t.t17
  store %v3.bld %t.t18, %v3.bld* %fn9_tmp.8
  ; fn9_tmp#9 = result(fn9_tmp#8)
  %t.t19 = load %v3.bld, %v3.bld* %fn9_tmp.8
  %t.t20 = call %v3 @v3.bld.result(%v3.bld %t.t19)
  store %v3 %t.t20, %v3* %fn9_tmp.9
  ; fn9_tmp#10 = vectors.$1
  %t.t21 = getelementptr inbounds %s6, %s6* %vectors, i32 0, i32 1
  %t.t22 = load %v6.bld, %v6.bld* %t.t21
  store %v6.bld %t.t22, %v6.bld* %fn9_tmp.10
  ; fn9_tmp#11 = result(fn9_tmp#10)
  %t.t23 = load %v6.bld, %v6.bld* %fn9_tmp.10
  %t.t24 = call %v6 @v6.bld.result(%v6.bld %t.t23)
  store %v6 %t.t24, %v6* %fn9_tmp.11
  ; fn9_tmp#12 = vectors.$2
  %t.t25 = getelementptr inbounds %s6, %s6* %vectors, i32 0, i32 2
  %t.t26 = load %v3.bld, %v3.bld* %t.t25
  store %v3.bld %t.t26, %v3.bld* %fn9_tmp.12
  ; fn9_tmp#13 = result(fn9_tmp#12)
  %t.t27 = load %v3.bld, %v3.bld* %fn9_tmp.12
  %t.t28 = call %v3 @v3.bld.result(%v3.bld %t.t27)
  store %v3 %t.t28, %v3* %fn9_tmp.13
  ; fn9_tmp#14 = vectors.$3
  %t.t29 = getelementptr inbounds %s6, %s6* %vectors, i32 0, i32 3
  %t.t30 = load %v5.bld, %v5.bld* %t.t29
  store %v5.bld %t.t30, %v5.bld* %fn9_tmp.14
  ; fn9_tmp#15 = result(fn9_tmp#14)
  %t.t31 = load %v5.bld, %v5.bld* %fn9_tmp.14
  %t.t32 = call %v5 @v5.bld.result(%v5.bld %t.t31)
  store %v5 %t.t32, %v5* %fn9_tmp.15
  ; fn9_tmp#16 = {fn9_tmp#1,fn9_tmp#3,fn9_tmp#5,fn9_tmp#7,fn9_tmp#9,fn9_tmp#11,fn9_tmp#13,fn9_tmp#15}
  %t.t33 = load %v3, %v3* %fn9_tmp.1
  %t.t34 = getelementptr inbounds %s7, %s7* %fn9_tmp.16, i32 0, i32 0
  store %v3 %t.t33, %v3* %t.t34
  %t.t35 = load %v4, %v4* %fn9_tmp.3
  %t.t36 = getelementptr inbounds %s7, %s7* %fn9_tmp.16, i32 0, i32 1
  store %v4 %t.t35, %v4* %t.t36
  %t.t37 = load %v4, %v4* %fn9_tmp.5
  %t.t38 = getelementptr inbounds %s7, %s7* %fn9_tmp.16, i32 0, i32 2
  store %v4 %t.t37, %v4* %t.t38
  %t.t39 = load %v0, %v0* %fn9_tmp.7
  %t.t40 = getelementptr inbounds %s7, %s7* %fn9_tmp.16, i32 0, i32 3
  store %v0 %t.t39, %v0* %t.t40
  %t.t41 = load %v3, %v3* %fn9_tmp.9
  %t.t42 = getelementptr inbounds %s7, %s7* %fn9_tmp.16, i32 0, i32 4
  store %v3 %t.t41, %v3* %t.t42
  %t.t43 = load %v6, %v6* %fn9_tmp.11
  %t.t44 = getelementptr inbounds %s7, %s7* %fn9_tmp.16, i32 0, i32 5
  store %v6 %t.t43, %v6* %t.t44
  %t.t45 = load %v3, %v3* %fn9_tmp.13
  %t.t46 = getelementptr inbounds %s7, %s7* %fn9_tmp.16, i32 0, i32 6
  store %v3 %t.t45, %v3* %t.t46
  %t.t47 = load %v5, %v5* %fn9_tmp.15
  %t.t48 = getelementptr inbounds %s7, %s7* %fn9_tmp.16, i32 0, i32 7
  store %v5 %t.t47, %v5* %t.t48
  ; return fn9_tmp#16
  %t.t49 = load %s7, %s7* %fn9_tmp.16
  %t.t50 = getelementptr %s7, %s7* null, i32 1
  %t.t51 = ptrtoint %s7* %t.t50 to i64
  %t.t54 = call i64 @weld_rt_get_run_id()
  %t.t52 = call i8* @weld_run_malloc(i64 %t.t54, i64 %t.t51)
  %t.t53 = bitcast i8* %t.t52 to %s7*
  store %s7 %t.t49, %s7* %t.t53
  call void @weld_rt_set_result(i8* %t.t52)
  br label %body.end
body.end:
  ret void
}

define void @f8(%s6 %fn7_tmp.5.in, %v2 %project.in, %work_t* %cur.work, i64 %lower.idx, i64 %upper.idx) {
fn.entry:
  %fn7_tmp.5 = alloca %s6
  %project = alloca %v2
  %fn8_tmp.16 = alloca %s6
  %isNull2 = alloca i1
  %fn8_tmp.5 = alloca i1
  %bs1 = alloca %s6
  %ns.1 = alloca %s3
  %fn8_tmp.6 = alloca double
  %i2 = alloca i64
  %fn8_tmp.2 = alloca %v3.bld
  %fn8_tmp.14 = alloca i64
  %fn8_tmp.12 = alloca i1
  %fn8_tmp.7 = alloca double
  %fn8_tmp.9 = alloca %v3.bld
  %fn8_tmp.8 = alloca double
  %isNull1 = alloca i1
  %fn8_tmp.13 = alloca i64
  %fn8_tmp.3 = alloca %v6.bld
  %fn8_tmp = alloca i1
  %fn8_tmp.11 = alloca i1
  %fn8_tmp.15 = alloca i64
  %fn8_tmp.10 = alloca %v5.bld
  %fn8_tmp.4 = alloca i1
  %fn8_tmp.1 = alloca i1
  %cur.idx = alloca i64
  store %s6 %fn7_tmp.5.in, %s6* %fn7_tmp.5
  store %v2 %project.in, %v2* %project
  %cur.tid = call i32 @weld_rt_thread_id()
  store %s6 %fn7_tmp.5.in, %s6* %bs1
  store i64 %lower.idx, i64* %cur.idx
  br label %loop.start
loop.start:
  %t.t0 = load i64, i64* %cur.idx
  %t.t1 = icmp ult i64 %t.t0, %upper.idx
  br i1 %t.t1, label %loop.body, label %loop.end
loop.body:
  %t.t2 = load %v2, %v2* %project
  %t.t3 = call %s3* @v2.at(%v2 %t.t2, i64 %t.t0)
  %t.t4 = load %s3, %s3* %t.t3
  store %s3 %t.t4, %s3* %ns.1
  store i64 %t.t0, i64* %i2
  br label %b.b0
b.b0:
  ; fn8_tmp = ns#1.$4
  %t.t5 = getelementptr inbounds %s3, %s3* %ns.1, i32 0, i32 4
  %t.t6 = load i1, i1* %t.t5
  store i1 %t.t6, i1* %fn8_tmp
  ; isNull2 = fn8_tmp
  %t.t7 = load i1, i1* %fn8_tmp
  store i1 %t.t7, i1* %isNull2
  ; fn8_tmp#1 = ns#1.$2
  %t.t8 = getelementptr inbounds %s3, %s3* %ns.1, i32 0, i32 2
  %t.t9 = load i1, i1* %t.t8
  store i1 %t.t9, i1* %fn8_tmp.1
  ; isNull1 = fn8_tmp#1
  %t.t10 = load i1, i1* %fn8_tmp.1
  store i1 %t.t10, i1* %isNull1
  ; fn8_tmp#2 = bs1.$0
  %t.t11 = getelementptr inbounds %s6, %s6* %bs1, i32 0, i32 0
  %t.t12 = load %v3.bld, %v3.bld* %t.t11
  store %v3.bld %t.t12, %v3.bld* %fn8_tmp.2
  ; merge(fn8_tmp#2, isNull1)
  %t.t13 = load %v3.bld, %v3.bld* %fn8_tmp.2
  %t.t14 = load i1, i1* %isNull1
  call %v3.bld @v3.bld.merge(%v3.bld %t.t13, i1 %t.t14, i32 %cur.tid)
  ; fn8_tmp#3 = bs1.$1
  %t.t15 = getelementptr inbounds %s6, %s6* %bs1, i32 0, i32 1
  %t.t16 = load %v6.bld, %v6.bld* %t.t15
  store %v6.bld %t.t16, %v6.bld* %fn8_tmp.3
  ; fn8_tmp#4 = false
  store i1 0, i1* %fn8_tmp.4
  ; fn8_tmp#5 = == isNull1 fn8_tmp#4
  %t.t17 = load i1, i1* %isNull1
  %t.t18 = load i1, i1* %fn8_tmp.4
  %t.t19 = icmp eq i1 %t.t17, %t.t18
  store i1 %t.t19, i1* %fn8_tmp.5
  ; branch fn8_tmp#5 B1 B2
  %t.t20 = load i1, i1* %fn8_tmp.5
  br i1 %t.t20, label %b.b1, label %b.b2
b.b1:
  ; fn8_tmp#6 = ns#1.$3
  %t.t21 = getelementptr inbounds %s3, %s3* %ns.1, i32 0, i32 3
  %t.t22 = load double, double* %t.t21
  store double %t.t22, double* %fn8_tmp.6
  ; fn8_tmp#8 = fn8_tmp#6
  %t.t23 = load double, double* %fn8_tmp.6
  store double %t.t23, double* %fn8_tmp.8
  ; jump B3
  br label %b.b3
b.b2:
  ; fn8_tmp#7 = 0.0
  store double 0.000000000000000000000000000000e0, double* %fn8_tmp.7
  ; fn8_tmp#8 = fn8_tmp#7
  %t.t24 = load double, double* %fn8_tmp.7
  store double %t.t24, double* %fn8_tmp.8
  ; jump B3
  br label %b.b3
b.b3:
  ; merge(fn8_tmp#3, fn8_tmp#8)
  %t.t25 = load %v6.bld, %v6.bld* %fn8_tmp.3
  %t.t26 = load double, double* %fn8_tmp.8
  call %v6.bld @v6.bld.merge(%v6.bld %t.t25, double %t.t26, i32 %cur.tid)
  ; fn8_tmp#9 = bs1.$2
  %t.t27 = getelementptr inbounds %s6, %s6* %bs1, i32 0, i32 2
  %t.t28 = load %v3.bld, %v3.bld* %t.t27
  store %v3.bld %t.t28, %v3.bld* %fn8_tmp.9
  ; merge(fn8_tmp#9, isNull2)
  %t.t29 = load %v3.bld, %v3.bld* %fn8_tmp.9
  %t.t30 = load i1, i1* %isNull2
  call %v3.bld @v3.bld.merge(%v3.bld %t.t29, i1 %t.t30, i32 %cur.tid)
  ; fn8_tmp#10 = bs1.$3
  %t.t31 = getelementptr inbounds %s6, %s6* %bs1, i32 0, i32 3
  %t.t32 = load %v5.bld, %v5.bld* %t.t31
  store %v5.bld %t.t32, %v5.bld* %fn8_tmp.10
  ; fn8_tmp#11 = false
  store i1 0, i1* %fn8_tmp.11
  ; fn8_tmp#12 = == isNull2 fn8_tmp#11
  %t.t33 = load i1, i1* %isNull2
  %t.t34 = load i1, i1* %fn8_tmp.11
  %t.t35 = icmp eq i1 %t.t33, %t.t34
  store i1 %t.t35, i1* %fn8_tmp.12
  ; branch fn8_tmp#12 B4 B5
  %t.t36 = load i1, i1* %fn8_tmp.12
  br i1 %t.t36, label %b.b4, label %b.b5
b.b4:
  ; fn8_tmp#13 = ns#1.$5
  %t.t37 = getelementptr inbounds %s3, %s3* %ns.1, i32 0, i32 5
  %t.t38 = load i64, i64* %t.t37
  store i64 %t.t38, i64* %fn8_tmp.13
  ; fn8_tmp#15 = fn8_tmp#13
  %t.t39 = load i64, i64* %fn8_tmp.13
  store i64 %t.t39, i64* %fn8_tmp.15
  ; jump B6
  br label %b.b6
b.b5:
  ; fn8_tmp#14 = 0L
  store i64 0, i64* %fn8_tmp.14
  ; fn8_tmp#15 = fn8_tmp#14
  %t.t40 = load i64, i64* %fn8_tmp.14
  store i64 %t.t40, i64* %fn8_tmp.15
  ; jump B6
  br label %b.b6
b.b6:
  ; merge(fn8_tmp#10, fn8_tmp#15)
  %t.t41 = load %v5.bld, %v5.bld* %fn8_tmp.10
  %t.t42 = load i64, i64* %fn8_tmp.15
  call %v5.bld @v5.bld.merge(%v5.bld %t.t41, i64 %t.t42, i32 %cur.tid)
  ; fn8_tmp#16 = {fn8_tmp#2,fn8_tmp#3,fn8_tmp#9,fn8_tmp#10}
  %t.t43 = load %v3.bld, %v3.bld* %fn8_tmp.2
  %t.t44 = getelementptr inbounds %s6, %s6* %fn8_tmp.16, i32 0, i32 0
  store %v3.bld %t.t43, %v3.bld* %t.t44
  %t.t45 = load %v6.bld, %v6.bld* %fn8_tmp.3
  %t.t46 = getelementptr inbounds %s6, %s6* %fn8_tmp.16, i32 0, i32 1
  store %v6.bld %t.t45, %v6.bld* %t.t46
  %t.t47 = load %v3.bld, %v3.bld* %fn8_tmp.9
  %t.t48 = getelementptr inbounds %s6, %s6* %fn8_tmp.16, i32 0, i32 2
  store %v3.bld %t.t47, %v3.bld* %t.t48
  %t.t49 = load %v5.bld, %v5.bld* %fn8_tmp.10
  %t.t50 = getelementptr inbounds %s6, %s6* %fn8_tmp.16, i32 0, i32 3
  store %v5.bld %t.t49, %v5.bld* %t.t50
  ; end
  br label %body.end
body.end:
  br label %loop.terminator
loop.terminator:
  %t.t51 = load i64, i64* %cur.idx
  %t.t52 = add i64 %t.t51, 1
  store i64 %t.t52, i64* %cur.idx
  br label %loop.start
loop.end:
  ret void
}

define void @f8_wrapper(%s4 %buffer, %s6 %fn7_tmp.5, %v2 %project, %work_t* %cur.work) {
fn.entry:
  %t.t0 = call i64 @v2.size(%v2 %project)
  %t.t1 = call i64 @v2.size(%v2 %project)
  %t.t2 = sub i64 %t.t0, 1
  %t.t3 = mul i64 1, %t.t2
  %t.t4 = add i64 %t.t3, 0
  %t.t5 = icmp slt i64 %t.t4, %t.t1
  br i1 %t.t5, label %t.t6, label %fn.boundcheckfailed
t.t6:
  br label %fn.boundcheckpassed
fn.boundcheckfailed:
  %t.t7 = call i64 @weld_rt_get_run_id()
  call void @weld_run_set_errno(i64 %t.t7, i64 5)
  call void @weld_rt_abort_thread()
  ; Unreachable!
  br label %fn.end
fn.boundcheckpassed:
  %t.t8 = icmp ule i64 %t.t0, 16384
  br i1 %t.t8, label %for.ser, label %for.par
for.ser:
  call void @f8(%s6 %fn7_tmp.5, %v2 %project, %work_t* %cur.work, i64 0, i64 %t.t0)
  call void @f9(%s4 %buffer, %s6 %fn7_tmp.5, %work_t* %cur.work)
  br label %fn.end
for.par:
  %t.t9 = insertvalue %s8 undef, %s6 %fn7_tmp.5, 0
  %t.t10 = insertvalue %s8 %t.t9, %v2 %project, 1
  %t.t11 = getelementptr %s8, %s8* null, i32 1
  %t.t12 = ptrtoint %s8* %t.t11 to i64
  %t.t13 = call i8* @malloc(i64 %t.t12)
  %t.t14 = bitcast i8* %t.t13 to %s8*
  store %s8 %t.t10, %s8* %t.t14
  %t.t15 = insertvalue %s9 undef, %s4 %buffer, 0
  %t.t16 = insertvalue %s9 %t.t15, %s6 %fn7_tmp.5, 1
  %t.t17 = getelementptr %s9, %s9* null, i32 1
  %t.t18 = ptrtoint %s9* %t.t17 to i64
  %t.t19 = call i8* @malloc(i64 %t.t18)
  %t.t20 = bitcast i8* %t.t19 to %s9*
  store %s9 %t.t16, %s9* %t.t20
  call void @weld_rt_start_loop(%work_t* %cur.work, i8* %t.t13, i8* %t.t19, void (%work_t*)* @f8_par, void (%work_t*)* @f9_par, i64 0, i64 %t.t0, i32 16384)
  br label %fn.end
fn.end:
  ret void
}

define void @f8_par(%work_t* %cur.work) {
entry:
  %t.t2 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 0
  %t.t3 = load i8*, i8** %t.t2
  %t.t0 = bitcast i8* %t.t3 to %s8*
  %t.t1 = load %s8, %s8* %t.t0
  %fn7_tmp.5 = extractvalue %s8 %t.t1, 0
  %project = extractvalue %s8 %t.t1, 1
  %t.t4 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 1
  %t.t5 = load i64, i64* %t.t4
  %t.t6 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 2
  %t.t7 = load i64, i64* %t.t6
  %t.t8 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 4
  %t.t9 = load i32, i32* %t.t8
  %t.t10 = trunc i32 %t.t9 to i1
  br i1 %t.t10, label %new_pieces, label %fn_call
new_pieces:
  br label %fn_call
fn_call:
  call void @f8(%s6 %fn7_tmp.5, %v2 %project, %work_t* %cur.work, i64 %t.t5, i64 %t.t7)
  ret void
}

define void @f9_par(%work_t* %cur.work) {
entry:
  %t.t2 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 0
  %t.t3 = load i8*, i8** %t.t2
  %t.t0 = bitcast i8* %t.t3 to %s9*
  %t.t1 = load %s9, %s9* %t.t0
  %buffer = extractvalue %s9 %t.t1, 0
  %fn7_tmp.5 = extractvalue %s9 %t.t1, 1
  %t.t4 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 4
  %t.t5 = load i32, i32* %t.t4
  %t.t6 = trunc i32 %t.t5 to i1
  br i1 %t.t6, label %new_pieces, label %fn_call
new_pieces:
  br label %fn_call
fn_call:
  call void @f9(%s4 %buffer, %s6 %fn7_tmp.5, %work_t* %cur.work)
  ret void
}

define void @f7(%s4 %buffer1.in, %s5 %fn6_tmp.3.in, %v2 %project.in, %work_t* %cur.work) {
fn.entry:
  %buffer1 = alloca %s4
  %fn6_tmp.3 = alloca %s5
  %project = alloca %v2
  %fn7_tmp.4 = alloca %v5.bld
  %fn7_tmp.5 = alloca %s6
  %fn7_tmp = alloca i1
  %fn7_tmp.1 = alloca %v3.bld
  %fn7_tmp.3 = alloca %v3.bld
  %buffer = alloca %s4
  %fn7_tmp.2 = alloca %v6.bld
  store %s4 %buffer1.in, %s4* %buffer1
  store %s5 %fn6_tmp.3.in, %s5* %fn6_tmp.3
  store %v2 %project.in, %v2* %project
  %cur.tid = call i32 @weld_rt_thread_id()
  br label %b.b0
b.b0:
  ; buffer1 = fn6_tmp#3.$0
  %t.t0 = getelementptr inbounds %s5, %s5* %fn6_tmp.3, i32 0, i32 0
  %t.t1 = load %s4, %s4* %t.t0
  store %s4 %t.t1, %s4* %buffer1
  ; fn7_tmp = fn6_tmp#3.$1
  %t.t2 = getelementptr inbounds %s5, %s5* %fn6_tmp.3, i32 0, i32 1
  %t.t3 = load i1, i1* %t.t2
  store i1 %t.t3, i1* %fn7_tmp
  ; branch fn7_tmp B1 B2
  %t.t4 = load i1, i1* %fn7_tmp
  br i1 %t.t4, label %b.b1, label %b.b2
b.b1:
  ; jump F3
  %t.t5 = load %s4, %s4* %buffer1
  %t.t6 = load %v2, %v2* %project
  call void @f3(%s4 %t.t5, %v2 %t.t6, %work_t* %cur.work)
  br label %body.end
b.b2:
  ; buffer = buffer1
  %t.t7 = load %s4, %s4* %buffer1
  store %s4 %t.t7, %s4* %buffer
  ; fn7_tmp#1 = new appender[bool]()
  %t.t8 = call %v3.bld @v3.bld.new(i64 16, %work_t* %cur.work, i32 0)
  store %v3.bld %t.t8, %v3.bld* %fn7_tmp.1
  ; fn7_tmp#2 = new appender[f64]()
  %t.t9 = call %v6.bld @v6.bld.new(i64 16, %work_t* %cur.work, i32 0)
  store %v6.bld %t.t9, %v6.bld* %fn7_tmp.2
  ; fn7_tmp#3 = new appender[bool]()
  %t.t10 = call %v3.bld @v3.bld.new(i64 16, %work_t* %cur.work, i32 0)
  store %v3.bld %t.t10, %v3.bld* %fn7_tmp.3
  ; fn7_tmp#4 = new appender[i64]()
  %t.t11 = call %v5.bld @v5.bld.new(i64 16, %work_t* %cur.work, i32 0)
  store %v5.bld %t.t11, %v5.bld* %fn7_tmp.4
  ; fn7_tmp#5 = {fn7_tmp#1,fn7_tmp#2,fn7_tmp#3,fn7_tmp#4}
  %t.t12 = load %v3.bld, %v3.bld* %fn7_tmp.1
  %t.t13 = getelementptr inbounds %s6, %s6* %fn7_tmp.5, i32 0, i32 0
  store %v3.bld %t.t12, %v3.bld* %t.t13
  %t.t14 = load %v6.bld, %v6.bld* %fn7_tmp.2
  %t.t15 = getelementptr inbounds %s6, %s6* %fn7_tmp.5, i32 0, i32 1
  store %v6.bld %t.t14, %v6.bld* %t.t15
  %t.t16 = load %v3.bld, %v3.bld* %fn7_tmp.3
  %t.t17 = getelementptr inbounds %s6, %s6* %fn7_tmp.5, i32 0, i32 2
  store %v3.bld %t.t16, %v3.bld* %t.t17
  %t.t18 = load %v5.bld, %v5.bld* %fn7_tmp.4
  %t.t19 = getelementptr inbounds %s6, %s6* %fn7_tmp.5, i32 0, i32 3
  store %v5.bld %t.t18, %v5.bld* %t.t19
  ; for [project, ] fn7_tmp#5 bs1 i2 ns#1 F8 F9 true
  %t.t20 = load %s4, %s4* %buffer
  %t.t21 = load %s6, %s6* %fn7_tmp.5
  %t.t22 = load %v2, %v2* %project
  call void @f8_wrapper(%s4 %t.t20, %s6 %t.t21, %v2 %t.t22, %work_t* %cur.work)
  br label %body.end
body.end:
  ret void
}

define void @f6(%s4 %buffer1.in, i64 %fn3_tmp.16.in, %v3.bld %fn3_tmp.17.in, i32 %fn3_tmp.19.in, %v4.bld %fn3_tmp.20.in, %v4.bld %fn3_tmp.22.in, %v0.bld %fn5_tmp.in, %v2 %project.in, %work_t* %cur.work) {
fn.entry:
  %fn3_tmp.16 = alloca i64
  %fn3_tmp.19 = alloca i32
  %project = alloca %v2
  %fn3_tmp.17 = alloca %v3.bld
  %fn5_tmp = alloca %v0.bld
  %fn3_tmp.20 = alloca %v4.bld
  %buffer1 = alloca %s4
  %fn3_tmp.22 = alloca %v4.bld
  %fn6_tmp.2 = alloca %s5
  %fn6_tmp.3 = alloca %s5
  %fn6_tmp.1 = alloca i1
  %fn6_tmp = alloca %s4
  store i64 %fn3_tmp.16.in, i64* %fn3_tmp.16
  store i32 %fn3_tmp.19.in, i32* %fn3_tmp.19
  store %v2 %project.in, %v2* %project
  store %v3.bld %fn3_tmp.17.in, %v3.bld* %fn3_tmp.17
  store %v0.bld %fn5_tmp.in, %v0.bld* %fn5_tmp
  store %v4.bld %fn3_tmp.20.in, %v4.bld* %fn3_tmp.20
  store %s4 %buffer1.in, %s4* %buffer1
  store %v4.bld %fn3_tmp.22.in, %v4.bld* %fn3_tmp.22
  %cur.tid = call i32 @weld_rt_thread_id()
  br label %b.b0
b.b0:
  ; fn6_tmp = {fn3_tmp#16,fn3_tmp#17,fn3_tmp#19,fn3_tmp#20,fn3_tmp#22,fn5_tmp}
  %t.t0 = load i64, i64* %fn3_tmp.16
  %t.t1 = getelementptr inbounds %s4, %s4* %fn6_tmp, i32 0, i32 0
  store i64 %t.t0, i64* %t.t1
  %t.t2 = load %v3.bld, %v3.bld* %fn3_tmp.17
  %t.t3 = getelementptr inbounds %s4, %s4* %fn6_tmp, i32 0, i32 1
  store %v3.bld %t.t2, %v3.bld* %t.t3
  %t.t4 = load i32, i32* %fn3_tmp.19
  %t.t5 = getelementptr inbounds %s4, %s4* %fn6_tmp, i32 0, i32 2
  store i32 %t.t4, i32* %t.t5
  %t.t6 = load %v4.bld, %v4.bld* %fn3_tmp.20
  %t.t7 = getelementptr inbounds %s4, %s4* %fn6_tmp, i32 0, i32 3
  store %v4.bld %t.t6, %v4.bld* %t.t7
  %t.t8 = load %v4.bld, %v4.bld* %fn3_tmp.22
  %t.t9 = getelementptr inbounds %s4, %s4* %fn6_tmp, i32 0, i32 4
  store %v4.bld %t.t8, %v4.bld* %t.t9
  %t.t10 = load %v0.bld, %v0.bld* %fn5_tmp
  %t.t11 = getelementptr inbounds %s4, %s4* %fn6_tmp, i32 0, i32 5
  store %v0.bld %t.t10, %v0.bld* %t.t11
  ; fn6_tmp#1 = true
  store i1 1, i1* %fn6_tmp.1
  ; fn6_tmp#2 = {fn6_tmp,fn6_tmp#1}
  %t.t12 = load %s4, %s4* %fn6_tmp
  %t.t13 = getelementptr inbounds %s5, %s5* %fn6_tmp.2, i32 0, i32 0
  store %s4 %t.t12, %s4* %t.t13
  %t.t14 = load i1, i1* %fn6_tmp.1
  %t.t15 = getelementptr inbounds %s5, %s5* %fn6_tmp.2, i32 0, i32 1
  store i1 %t.t14, i1* %t.t15
  ; fn6_tmp#3 = fn6_tmp#2
  %t.t16 = load %s5, %s5* %fn6_tmp.2
  store %s5 %t.t16, %s5* %fn6_tmp.3
  ; jump F7
  %t.t17 = load %s4, %s4* %buffer1
  %t.t18 = load %s5, %s5* %fn6_tmp.3
  %t.t19 = load %v2, %v2* %project
  call void @f7(%s4 %t.t17, %s5 %t.t18, %v2 %t.t19, %work_t* %cur.work)
  br label %body.end
body.end:
  ret void
}

define void @f5(%s4 %buffer1.in, i64 %fn3_tmp.16.in, %v3.bld %fn3_tmp.17.in, i32 %fn3_tmp.19.in, %v4.bld %fn3_tmp.20.in, %v4.bld %fn3_tmp.22.in, %v0.bld %fn3_tmp.25.in, %v2 %project.in, %work_t* %cur.work) {
fn.entry:
  %project = alloca %v2
  %fn3_tmp.19 = alloca i32
  %fn3_tmp.20 = alloca %v4.bld
  %fn3_tmp.16 = alloca i64
  %fn3_tmp.22 = alloca %v4.bld
  %buffer1 = alloca %s4
  %fn3_tmp.25 = alloca %v0.bld
  %fn3_tmp.17 = alloca %v3.bld
  %fn5_tmp = alloca %v0.bld
  store %v2 %project.in, %v2* %project
  store i32 %fn3_tmp.19.in, i32* %fn3_tmp.19
  store %v4.bld %fn3_tmp.20.in, %v4.bld* %fn3_tmp.20
  store i64 %fn3_tmp.16.in, i64* %fn3_tmp.16
  store %v4.bld %fn3_tmp.22.in, %v4.bld* %fn3_tmp.22
  store %s4 %buffer1.in, %s4* %buffer1
  store %v0.bld %fn3_tmp.25.in, %v0.bld* %fn3_tmp.25
  store %v3.bld %fn3_tmp.17.in, %v3.bld* %fn3_tmp.17
  %cur.tid = call i32 @weld_rt_thread_id()
  br label %b.b0
b.b0:
  ; fn5_tmp = fn3_tmp#25
  %t.t0 = load %v0.bld, %v0.bld* %fn3_tmp.25
  store %v0.bld %t.t0, %v0.bld* %fn5_tmp
  ; jump F6
  %t.t1 = load %s4, %s4* %buffer1
  %t.t2 = load i64, i64* %fn3_tmp.16
  %t.t3 = load %v3.bld, %v3.bld* %fn3_tmp.17
  %t.t4 = load i32, i32* %fn3_tmp.19
  %t.t5 = load %v4.bld, %v4.bld* %fn3_tmp.20
  %t.t6 = load %v4.bld, %v4.bld* %fn3_tmp.22
  %t.t7 = load %v0.bld, %v0.bld* %fn5_tmp
  %t.t8 = load %v2, %v2* %project
  call void @f6(%s4 %t.t1, i64 %t.t2, %v3.bld %t.t3, i32 %t.t4, %v4.bld %t.t5, %v4.bld %t.t6, %v0.bld %t.t7, %v2 %t.t8, %work_t* %cur.work)
  br label %body.end
body.end:
  ret void
}

define void @f4(%v0.bld %fn3_tmp.25.in, %v0 %fn3_tmp.26.in, %work_t* %cur.work, i64 %lower.idx, i64 %upper.idx) {
fn.entry:
  %fn3_tmp.25 = alloca %v0.bld
  %fn3_tmp.26 = alloca %v0
  %n = alloca %u8
  %i1 = alloca i64
  %b = alloca %v0.bld
  %cur.idx = alloca i64
  store %v0.bld %fn3_tmp.25.in, %v0.bld* %fn3_tmp.25
  store %v0 %fn3_tmp.26.in, %v0* %fn3_tmp.26
  %cur.tid = call i32 @weld_rt_thread_id()
  store %v0.bld %fn3_tmp.25.in, %v0.bld* %b
  store i64 %lower.idx, i64* %cur.idx
  br label %loop.start
loop.start:
  %t.t0 = load i64, i64* %cur.idx
  %t.t1 = icmp ult i64 %t.t0, %upper.idx
  br i1 %t.t1, label %loop.body, label %loop.end
loop.body:
  %t.t2 = load %v0, %v0* %fn3_tmp.26
  %t.t3 = call %u8* @v0.at(%v0 %t.t2, i64 %t.t0)
  %t.t4 = load %u8, %u8* %t.t3
  store %u8 %t.t4, %u8* %n
  store i64 %t.t0, i64* %i1
  br label %b.b0
b.b0:
  ; merge(b, n)
  %t.t5 = load %v0.bld, %v0.bld* %b
  %t.t6 = load %u8, %u8* %n
  call %v0.bld @v0.bld.merge(%v0.bld %t.t5, %u8 %t.t6, i32 %cur.tid)
  ; end
  br label %body.end
body.end:
  br label %loop.terminator
loop.terminator:
  %t.t7 = load i64, i64* %cur.idx
  %t.t8 = add i64 %t.t7, 1
  store i64 %t.t8, i64* %cur.idx
  br label %loop.start
loop.end:
  ret void
}

define void @f4_wrapper(%s4 %buffer1, i64 %fn3_tmp.16, %v3.bld %fn3_tmp.17, i32 %fn3_tmp.19, %v4.bld %fn3_tmp.20, %v4.bld %fn3_tmp.22, %v0.bld %fn3_tmp.25, %v0 %fn3_tmp.26, %v2 %project, %work_t* %cur.work) {
fn.entry:
  %t.t0 = call i64 @v0.size(%v0 %fn3_tmp.26)
  %t.t1 = call i64 @v0.size(%v0 %fn3_tmp.26)
  %t.t2 = sub i64 %t.t0, 1
  %t.t3 = mul i64 1, %t.t2
  %t.t4 = add i64 %t.t3, 0
  %t.t5 = icmp slt i64 %t.t4, %t.t1
  br i1 %t.t5, label %t.t6, label %fn.boundcheckfailed
t.t6:
  br label %fn.boundcheckpassed
fn.boundcheckfailed:
  %t.t7 = call i64 @weld_rt_get_run_id()
  call void @weld_run_set_errno(i64 %t.t7, i64 5)
  call void @weld_rt_abort_thread()
  ; Unreachable!
  br label %fn.end
fn.boundcheckpassed:
  %t.t8 = icmp ule i64 %t.t0, 16384
  br i1 %t.t8, label %for.ser, label %for.par
for.ser:
  call void @f4(%v0.bld %fn3_tmp.25, %v0 %fn3_tmp.26, %work_t* %cur.work, i64 0, i64 %t.t0)
  call void @f5(%s4 %buffer1, i64 %fn3_tmp.16, %v3.bld %fn3_tmp.17, i32 %fn3_tmp.19, %v4.bld %fn3_tmp.20, %v4.bld %fn3_tmp.22, %v0.bld %fn3_tmp.25, %v2 %project, %work_t* %cur.work)
  br label %fn.end
for.par:
  %t.t9 = insertvalue %s10 undef, %v0.bld %fn3_tmp.25, 0
  %t.t10 = insertvalue %s10 %t.t9, %v0 %fn3_tmp.26, 1
  %t.t11 = getelementptr %s10, %s10* null, i32 1
  %t.t12 = ptrtoint %s10* %t.t11 to i64
  %t.t13 = call i8* @malloc(i64 %t.t12)
  %t.t14 = bitcast i8* %t.t13 to %s10*
  store %s10 %t.t10, %s10* %t.t14
  %t.t15 = insertvalue %s11 undef, %s4 %buffer1, 0
  %t.t16 = insertvalue %s11 %t.t15, i64 %fn3_tmp.16, 1
  %t.t17 = insertvalue %s11 %t.t16, %v3.bld %fn3_tmp.17, 2
  %t.t18 = insertvalue %s11 %t.t17, i32 %fn3_tmp.19, 3
  %t.t19 = insertvalue %s11 %t.t18, %v4.bld %fn3_tmp.20, 4
  %t.t20 = insertvalue %s11 %t.t19, %v4.bld %fn3_tmp.22, 5
  %t.t21 = insertvalue %s11 %t.t20, %v0.bld %fn3_tmp.25, 6
  %t.t22 = insertvalue %s11 %t.t21, %v2 %project, 7
  %t.t23 = getelementptr %s11, %s11* null, i32 1
  %t.t24 = ptrtoint %s11* %t.t23 to i64
  %t.t25 = call i8* @malloc(i64 %t.t24)
  %t.t26 = bitcast i8* %t.t25 to %s11*
  store %s11 %t.t22, %s11* %t.t26
  call void @weld_rt_start_loop(%work_t* %cur.work, i8* %t.t13, i8* %t.t25, void (%work_t*)* @f4_par, void (%work_t*)* @f5_par, i64 0, i64 %t.t0, i32 16384)
  br label %fn.end
fn.end:
  ret void
}

define void @f4_par(%work_t* %cur.work) {
entry:
  %t.t2 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 0
  %t.t3 = load i8*, i8** %t.t2
  %t.t0 = bitcast i8* %t.t3 to %s10*
  %t.t1 = load %s10, %s10* %t.t0
  %fn3_tmp.25 = extractvalue %s10 %t.t1, 0
  %fn3_tmp.26 = extractvalue %s10 %t.t1, 1
  %t.t4 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 1
  %t.t5 = load i64, i64* %t.t4
  %t.t6 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 2
  %t.t7 = load i64, i64* %t.t6
  %t.t8 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 4
  %t.t9 = load i32, i32* %t.t8
  %t.t10 = trunc i32 %t.t9 to i1
  br i1 %t.t10, label %new_pieces, label %fn_call
new_pieces:
  call void @v0.bld.newPiece(%v0.bld %fn3_tmp.25, %work_t* %cur.work)
  br label %fn_call
fn_call:
  call void @f4(%v0.bld %fn3_tmp.25, %v0 %fn3_tmp.26, %work_t* %cur.work, i64 %t.t5, i64 %t.t7)
  ret void
}

define void @f5_par(%work_t* %cur.work) {
entry:
  %t.t2 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 0
  %t.t3 = load i8*, i8** %t.t2
  %t.t0 = bitcast i8* %t.t3 to %s11*
  %t.t1 = load %s11, %s11* %t.t0
  %buffer1 = extractvalue %s11 %t.t1, 0
  %fn3_tmp.16 = extractvalue %s11 %t.t1, 1
  %fn3_tmp.17 = extractvalue %s11 %t.t1, 2
  %fn3_tmp.19 = extractvalue %s11 %t.t1, 3
  %fn3_tmp.20 = extractvalue %s11 %t.t1, 4
  %fn3_tmp.22 = extractvalue %s11 %t.t1, 5
  %fn3_tmp.25 = extractvalue %s11 %t.t1, 6
  %project = extractvalue %s11 %t.t1, 7
  %t.t4 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 4
  %t.t5 = load i32, i32* %t.t4
  %t.t6 = trunc i32 %t.t5 to i1
  br i1 %t.t6, label %new_pieces, label %fn_call
new_pieces:
  call void @v3.bld.newPiece(%v3.bld %fn3_tmp.17, %work_t* %cur.work)
  call void @v4.bld.newPiece(%v4.bld %fn3_tmp.20, %work_t* %cur.work)
  call void @v4.bld.newPiece(%v4.bld %fn3_tmp.22, %work_t* %cur.work)
  call void @v0.bld.newPiece(%v0.bld %fn3_tmp.25, %work_t* %cur.work)
  br label %fn_call
fn_call:
  call void @f5(%s4 %buffer1, i64 %fn3_tmp.16, %v3.bld %fn3_tmp.17, i32 %fn3_tmp.19, %v4.bld %fn3_tmp.20, %v4.bld %fn3_tmp.22, %v0.bld %fn3_tmp.25, %v2 %project, %work_t* %cur.work)
  ret void
}

define void @f3(%s4 %buffer1.in, %v2 %project.in, %work_t* %cur.work) {
fn.entry:
  %project = alloca %v2
  %buffer1 = alloca %s4
  %fn3_tmp.26 = alloca %v0
  %fn3_tmp.22 = alloca %v4.bld
  %fn5_tmp = alloca %v0.bld
  %fn3_tmp.10 = alloca i64
  %index = alloca i64
  %fn3_tmp.24 = alloca i1
  %fn3_tmp.18 = alloca i32
  %fn6_tmp.3 = alloca %s5
  %ns = alloca %s3
  %fn3_tmp.20 = alloca %v4.bld
  %fn3_tmp.7 = alloca i32
  %fn3_tmp.29 = alloca %s5
  %fn3_tmp.1 = alloca %s3
  %fn3_tmp.8 = alloca i32
  %fn3_tmp.19 = alloca i32
  %fn3_tmp.27 = alloca %v0.bld
  %fn3_tmp.9 = alloca i32
  %fn3_tmp.25 = alloca %v0.bld
  %fn3_tmp.11 = alloca i1
  %fn3_tmp.13 = alloca i1
  %fn3_tmp.23 = alloca i1
  %length = alloca i32
  %fn3_tmp.12 = alloca i64
  %fn3_tmp.21 = alloca i32
  %fn3_tmp.2 = alloca i1
  %fn3_tmp.4 = alloca i1
  %fn3_tmp.5 = alloca %v0
  %fn3_tmp.6 = alloca i64
  %fn3_tmp.3 = alloca i1
  %fn3_tmp = alloca i64
  %fn3_tmp.15 = alloca i64
  %fn3_tmp.17 = alloca %v3.bld
  %fn3_tmp.16 = alloca i64
  %fn3_tmp.14 = alloca i1
  %fn3_tmp.28 = alloca i1
  %isNull = alloca i1
  store %v2 %project.in, %v2* %project
  store %s4 %buffer1.in, %s4* %buffer1
  %cur.tid = call i32 @weld_rt_thread_id()
  br label %b.b0
b.b0:
  ; fn3_tmp = buffer1.$0
  %t.t0 = getelementptr inbounds %s4, %s4* %buffer1, i32 0, i32 0
  %t.t1 = load i64, i64* %t.t0
  store i64 %t.t1, i64* %fn3_tmp
  ; index = fn3_tmp
  %t.t2 = load i64, i64* %fn3_tmp
  store i64 %t.t2, i64* %index
  ; fn3_tmp#1 = lookup(project, index)
  %t.t3 = load %v2, %v2* %project
  %t.t4 = load i64, i64* %index
  %t.t5 = call %s3* @v2.at(%v2 %t.t3, i64 %t.t4)
  %t.t6 = load %s3, %s3* %t.t5
  store %s3 %t.t6, %s3* %fn3_tmp.1
  ; ns = fn3_tmp#1
  %t.t7 = load %s3, %s3* %fn3_tmp.1
  store %s3 %t.t7, %s3* %ns
  ; fn3_tmp#2 = ns.$0
  %t.t8 = getelementptr inbounds %s3, %s3* %ns, i32 0, i32 0
  %t.t9 = load i1, i1* %t.t8
  store i1 %t.t9, i1* %fn3_tmp.2
  ; isNull = fn3_tmp#2
  %t.t10 = load i1, i1* %fn3_tmp.2
  store i1 %t.t10, i1* %isNull
  ; fn3_tmp#3 = false
  store i1 0, i1* %fn3_tmp.3
  ; fn3_tmp#4 = == isNull fn3_tmp#3
  %t.t11 = load i1, i1* %isNull
  %t.t12 = load i1, i1* %fn3_tmp.3
  %t.t13 = icmp eq i1 %t.t11, %t.t12
  store i1 %t.t13, i1* %fn3_tmp.4
  ; branch fn3_tmp#4 B1 B2
  %t.t14 = load i1, i1* %fn3_tmp.4
  br i1 %t.t14, label %b.b1, label %b.b2
b.b1:
  ; fn3_tmp#5 = ns.$1
  %t.t15 = getelementptr inbounds %s3, %s3* %ns, i32 0, i32 1
  %t.t16 = load %v0, %v0* %t.t15
  store %v0 %t.t16, %v0* %fn3_tmp.5
  ; fn3_tmp#6 = len(fn3_tmp#5)
  %t.t17 = load %v0, %v0* %fn3_tmp.5
  %t.t18 = call i64 @v0.size(%v0 %t.t17)
  store i64 %t.t18, i64* %fn3_tmp.6
  ; fn3_tmp#7 = cast(fn3_tmp#6)
  %t.t19 = load i64, i64* %fn3_tmp.6
  %t.t20 = trunc i64 %t.t19 to i32
  store i32 %t.t20, i32* %fn3_tmp.7
  ; fn3_tmp#9 = fn3_tmp#7
  %t.t21 = load i32, i32* %fn3_tmp.7
  store i32 %t.t21, i32* %fn3_tmp.9
  ; jump B3
  br label %b.b3
b.b2:
  ; fn3_tmp#8 = 0
  store i32 0, i32* %fn3_tmp.8
  ; fn3_tmp#9 = fn3_tmp#8
  %t.t22 = load i32, i32* %fn3_tmp.8
  store i32 %t.t22, i32* %fn3_tmp.9
  ; jump B3
  br label %b.b3
b.b3:
  ; length = fn3_tmp#9
  %t.t23 = load i32, i32* %fn3_tmp.9
  store i32 %t.t23, i32* %length
  ; fn3_tmp#10 = len(project)
  %t.t24 = load %v2, %v2* %project
  %t.t25 = call i64 @v2.size(%v2 %t.t24)
  store i64 %t.t25, i64* %fn3_tmp.10
  ; fn3_tmp#11 = < index fn3_tmp#10
  %t.t26 = load i64, i64* %index
  %t.t27 = load i64, i64* %fn3_tmp.10
  %t.t28 = icmp slt i64 %t.t26, %t.t27
  store i1 %t.t28, i1* %fn3_tmp.11
  ; fn3_tmp#12 = 100L
  store i64 100, i64* %fn3_tmp.12
  ; fn3_tmp#13 = >= index fn3_tmp#12
  %t.t29 = load i64, i64* %index
  %t.t30 = load i64, i64* %fn3_tmp.12
  %t.t31 = icmp sle i64 %t.t29, %t.t30
  store i1 %t.t31, i1* %fn3_tmp.13
  ; fn3_tmp#14 = || fn3_tmp#11 fn3_tmp#13
  %t.t32 = load i1, i1* %fn3_tmp.11
  %t.t33 = load i1, i1* %fn3_tmp.13
  %t.t34 = and i1 %t.t32, %t.t33
  store i1 %t.t34, i1* %fn3_tmp.14
  ; branch fn3_tmp#14 B4 B5
  %t.t35 = load i1, i1* %fn3_tmp.14
  br i1 %t.t35, label %b.b4, label %b.b5
b.b4:
  ; fn3_tmp#15 = 1L
  store i64 1, i64* %fn3_tmp.15
  ; fn3_tmp#16 = + index fn3_tmp#15
  %t.t36 = load i64, i64* %index
  %t.t37 = load i64, i64* %fn3_tmp.15
  %t.t38 = add i64 %t.t36, %t.t37
  store i64 %t.t38, i64* %fn3_tmp.16
  ; fn3_tmp#17 = buffer1.$1
  %t.t39 = getelementptr inbounds %s4, %s4* %buffer1, i32 0, i32 1
  %t.t40 = load %v3.bld, %v3.bld* %t.t39
  store %v3.bld %t.t40, %v3.bld* %fn3_tmp.17
  ; merge(fn3_tmp#17, isNull)
  %t.t41 = load %v3.bld, %v3.bld* %fn3_tmp.17
  %t.t42 = load i1, i1* %isNull
  call %v3.bld @v3.bld.merge(%v3.bld %t.t41, i1 %t.t42, i32 %cur.tid)
  ; fn3_tmp#18 = buffer1.$2
  %t.t43 = getelementptr inbounds %s4, %s4* %buffer1, i32 0, i32 2
  %t.t44 = load i32, i32* %t.t43
  store i32 %t.t44, i32* %fn3_tmp.18
  ; fn3_tmp#19 = + fn3_tmp#18 length
  %t.t45 = load i32, i32* %fn3_tmp.18
  %t.t46 = load i32, i32* %length
  %t.t47 = add i32 %t.t45, %t.t46
  store i32 %t.t47, i32* %fn3_tmp.19
  ; fn3_tmp#20 = buffer1.$3
  %t.t48 = getelementptr inbounds %s4, %s4* %buffer1, i32 0, i32 3
  %t.t49 = load %v4.bld, %v4.bld* %t.t48
  store %v4.bld %t.t49, %v4.bld* %fn3_tmp.20
  ; fn3_tmp#21 = buffer1.$2
  %t.t50 = getelementptr inbounds %s4, %s4* %buffer1, i32 0, i32 2
  %t.t51 = load i32, i32* %t.t50
  store i32 %t.t51, i32* %fn3_tmp.21
  ; merge(fn3_tmp#20, fn3_tmp#21)
  %t.t52 = load %v4.bld, %v4.bld* %fn3_tmp.20
  %t.t53 = load i32, i32* %fn3_tmp.21
  call %v4.bld @v4.bld.merge(%v4.bld %t.t52, i32 %t.t53, i32 %cur.tid)
  ; fn3_tmp#22 = buffer1.$4
  %t.t54 = getelementptr inbounds %s4, %s4* %buffer1, i32 0, i32 4
  %t.t55 = load %v4.bld, %v4.bld* %t.t54
  store %v4.bld %t.t55, %v4.bld* %fn3_tmp.22
  ; merge(fn3_tmp#22, length)
  %t.t56 = load %v4.bld, %v4.bld* %fn3_tmp.22
  %t.t57 = load i32, i32* %length
  call %v4.bld @v4.bld.merge(%v4.bld %t.t56, i32 %t.t57, i32 %cur.tid)
  ; fn3_tmp#23 = false
  store i1 0, i1* %fn3_tmp.23
  ; fn3_tmp#24 = == isNull fn3_tmp#23
  %t.t58 = load i1, i1* %isNull
  %t.t59 = load i1, i1* %fn3_tmp.23
  %t.t60 = icmp eq i1 %t.t58, %t.t59
  store i1 %t.t60, i1* %fn3_tmp.24
  ; branch fn3_tmp#24 B6 B7
  %t.t61 = load i1, i1* %fn3_tmp.24
  br i1 %t.t61, label %b.b6, label %b.b7
b.b5:
  ; fn3_tmp#28 = false
  store i1 0, i1* %fn3_tmp.28
  ; fn3_tmp#29 = {buffer1,fn3_tmp#28}
  %t.t62 = load %s4, %s4* %buffer1
  %t.t63 = getelementptr inbounds %s5, %s5* %fn3_tmp.29, i32 0, i32 0
  store %s4 %t.t62, %s4* %t.t63
  %t.t64 = load i1, i1* %fn3_tmp.28
  %t.t65 = getelementptr inbounds %s5, %s5* %fn3_tmp.29, i32 0, i32 1
  store i1 %t.t64, i1* %t.t65
  ; fn6_tmp#3 = fn3_tmp#29
  %t.t66 = load %s5, %s5* %fn3_tmp.29
  store %s5 %t.t66, %s5* %fn6_tmp.3
  ; jump F7
  %t.t67 = load %s4, %s4* %buffer1
  %t.t68 = load %s5, %s5* %fn6_tmp.3
  %t.t69 = load %v2, %v2* %project
  call void @f7(%s4 %t.t67, %s5 %t.t68, %v2 %t.t69, %work_t* %cur.work)
  br label %body.end
b.b6:
  ; fn3_tmp#25 = buffer1.$5
  %t.t70 = getelementptr inbounds %s4, %s4* %buffer1, i32 0, i32 5
  %t.t71 = load %v0.bld, %v0.bld* %t.t70
  store %v0.bld %t.t71, %v0.bld* %fn3_tmp.25
  ; fn3_tmp#26 = ns.$1
  %t.t72 = getelementptr inbounds %s3, %s3* %ns, i32 0, i32 1
  %t.t73 = load %v0, %v0* %t.t72
  store %v0 %t.t73, %v0* %fn3_tmp.26
  ; for [fn3_tmp#26, ] fn3_tmp#25 b i1 n F4 F5 true
  %t.t74 = load %s4, %s4* %buffer1
  %t.t75 = load i64, i64* %fn3_tmp.16
  %t.t76 = load %v3.bld, %v3.bld* %fn3_tmp.17
  %t.t77 = load i32, i32* %fn3_tmp.19
  %t.t78 = load %v4.bld, %v4.bld* %fn3_tmp.20
  %t.t79 = load %v4.bld, %v4.bld* %fn3_tmp.22
  %t.t80 = load %v0.bld, %v0.bld* %fn3_tmp.25
  %t.t81 = load %v0, %v0* %fn3_tmp.26
  %t.t82 = load %v2, %v2* %project
  call void @f4_wrapper(%s4 %t.t74, i64 %t.t75, %v3.bld %t.t76, i32 %t.t77, %v4.bld %t.t78, %v4.bld %t.t79, %v0.bld %t.t80, %v0 %t.t81, %v2 %t.t82, %work_t* %cur.work)
  br label %body.end
b.b7:
  ; fn3_tmp#27 = buffer1.$5
  %t.t83 = getelementptr inbounds %s4, %s4* %buffer1, i32 0, i32 5
  %t.t84 = load %v0.bld, %v0.bld* %t.t83
  store %v0.bld %t.t84, %v0.bld* %fn3_tmp.27
  ; fn5_tmp = fn3_tmp#27
  %t.t85 = load %v0.bld, %v0.bld* %fn3_tmp.27
  store %v0.bld %t.t85, %v0.bld* %fn5_tmp
  ; jump F6
  %t.t86 = load %s4, %s4* %buffer1
  %t.t87 = load i64, i64* %fn3_tmp.16
  %t.t88 = load %v3.bld, %v3.bld* %fn3_tmp.17
  %t.t89 = load i32, i32* %fn3_tmp.19
  %t.t90 = load %v4.bld, %v4.bld* %fn3_tmp.20
  %t.t91 = load %v4.bld, %v4.bld* %fn3_tmp.22
  %t.t92 = load %v0.bld, %v0.bld* %fn5_tmp
  %t.t93 = load %v2, %v2* %project
  call void @f6(%s4 %t.t86, i64 %t.t87, %v3.bld %t.t88, i32 %t.t89, %v4.bld %t.t90, %v4.bld %t.t91, %v0.bld %t.t92, %v2 %t.t93, %work_t* %cur.work)
  br label %body.end
body.end:
  ret void
}

define void @f2(%v2.bld %fn0_tmp.1.in, %work_t* %cur.work) {
fn.entry:
  %fn0_tmp.1 = alloca %v2.bld
  %fn2_tmp.2 = alloca %v3.bld
  %fn2_tmp = alloca %v2
  %fn2_tmp.3 = alloca i32
  %fn2_tmp.6 = alloca %v0.bld
  %buffer1 = alloca %s4
  %project = alloca %v2
  %fn2_tmp.4 = alloca %v4.bld
  %fn2_tmp.7 = alloca %s4
  %fn2_tmp.5 = alloca %v4.bld
  %fn2_tmp.1 = alloca i64
  store %v2.bld %fn0_tmp.1.in, %v2.bld* %fn0_tmp.1
  %cur.tid = call i32 @weld_rt_thread_id()
  br label %b.b0
b.b0:
  ; fn2_tmp = result(fn0_tmp#1)
  %t.t0 = load %v2.bld, %v2.bld* %fn0_tmp.1
  %t.t1 = call %v2 @v2.bld.result(%v2.bld %t.t0)
  store %v2 %t.t1, %v2* %fn2_tmp
  ; project = fn2_tmp
  %t.t2 = load %v2, %v2* %fn2_tmp
  store %v2 %t.t2, %v2* %project
  ; fn2_tmp#1 = 0L
  store i64 0, i64* %fn2_tmp.1
  ; fn2_tmp#2 = new appender[bool]()
  %t.t3 = call %v3.bld @v3.bld.new(i64 16, %work_t* %cur.work, i32 0)
  store %v3.bld %t.t3, %v3.bld* %fn2_tmp.2
  ; fn2_tmp#3 = 0
  store i32 0, i32* %fn2_tmp.3
  ; fn2_tmp#4 = new appender[i32]()
  %t.t4 = call %v4.bld @v4.bld.new(i64 16, %work_t* %cur.work, i32 0)
  store %v4.bld %t.t4, %v4.bld* %fn2_tmp.4
  ; fn2_tmp#5 = new appender[i32]()
  %t.t5 = call %v4.bld @v4.bld.new(i64 16, %work_t* %cur.work, i32 0)
  store %v4.bld %t.t5, %v4.bld* %fn2_tmp.5
  ; fn2_tmp#6 = new appender[u8]()
  %t.t6 = call %v0.bld @v0.bld.new(i64 16, %work_t* %cur.work, i32 0)
  store %v0.bld %t.t6, %v0.bld* %fn2_tmp.6
  ; fn2_tmp#7 = {fn2_tmp#1,fn2_tmp#2,fn2_tmp#3,fn2_tmp#4,fn2_tmp#5,fn2_tmp#6}
  %t.t7 = load i64, i64* %fn2_tmp.1
  %t.t8 = getelementptr inbounds %s4, %s4* %fn2_tmp.7, i32 0, i32 0
  store i64 %t.t7, i64* %t.t8
  %t.t9 = load %v3.bld, %v3.bld* %fn2_tmp.2
  %t.t10 = getelementptr inbounds %s4, %s4* %fn2_tmp.7, i32 0, i32 1
  store %v3.bld %t.t9, %v3.bld* %t.t10
  %t.t11 = load i32, i32* %fn2_tmp.3
  %t.t12 = getelementptr inbounds %s4, %s4* %fn2_tmp.7, i32 0, i32 2
  store i32 %t.t11, i32* %t.t12
  %t.t13 = load %v4.bld, %v4.bld* %fn2_tmp.4
  %t.t14 = getelementptr inbounds %s4, %s4* %fn2_tmp.7, i32 0, i32 3
  store %v4.bld %t.t13, %v4.bld* %t.t14
  %t.t15 = load %v4.bld, %v4.bld* %fn2_tmp.5
  %t.t16 = getelementptr inbounds %s4, %s4* %fn2_tmp.7, i32 0, i32 4
  store %v4.bld %t.t15, %v4.bld* %t.t16
  %t.t17 = load %v0.bld, %v0.bld* %fn2_tmp.6
  %t.t18 = getelementptr inbounds %s4, %s4* %fn2_tmp.7, i32 0, i32 5
  store %v0.bld %t.t17, %v0.bld* %t.t18
  ; buffer1 = fn2_tmp#7
  %t.t19 = load %s4, %s4* %fn2_tmp.7
  store %s4 %t.t19, %s4* %buffer1
  ; jump F3
  %t.t20 = load %s4, %s4* %buffer1
  %t.t21 = load %v2, %v2* %project
  call void @f3(%s4 %t.t20, %v2 %t.t21, %work_t* %cur.work)
  br label %body.end
body.end:
  ret void
}

define void @f1(%v2.bld %fn0_tmp.1.in, %v1 %kvs.in, %work_t* %cur.work, i64 %lower.idx, i64 %upper.idx) {
fn.entry:
  %kvs = alloca %v1
  %fn0_tmp.1 = alloca %v2.bld
  %fn1_tmp = alloca %s1
  %fn1_tmp.5 = alloca %s2
  %fn1_tmp.8 = alloca %s2
  %fn1_tmp.2 = alloca %s1
  %fn1_tmp.1 = alloca i1
  %fn1_tmp.6 = alloca double
  %fn1_tmp.10 = alloca %s3
  %fn1_tmp.7 = alloca i1
  %fn1_tmp.9 = alloca i64
  %kv = alloca %s0
  %bs = alloca %v2.bld
  %i = alloca i64
  %fn1_tmp.3 = alloca %v0
  %fn1_tmp.4 = alloca i1
  %cur.idx = alloca i64
  store %v1 %kvs.in, %v1* %kvs
  store %v2.bld %fn0_tmp.1.in, %v2.bld* %fn0_tmp.1
  %cur.tid = call i32 @weld_rt_thread_id()
  store %v2.bld %fn0_tmp.1.in, %v2.bld* %bs
  store i64 %lower.idx, i64* %cur.idx
  br label %loop.start
loop.start:
  %t.t0 = load i64, i64* %cur.idx
  %t.t1 = icmp ult i64 %t.t0, %upper.idx
  br i1 %t.t1, label %loop.body, label %loop.end
loop.body:
  %t.t2 = load %v1, %v1* %kvs
  %t.t3 = call %s0* @v1.at(%v1 %t.t2, i64 %t.t0)
  %t.t4 = load %s0, %s0* %t.t3
  store %s0 %t.t4, %s0* %kv
  store i64 %t.t0, i64* %i
  br label %b.b0
b.b0:
  ; fn1_tmp = kv.$0
  %t.t5 = getelementptr inbounds %s0, %s0* %kv, i32 0, i32 0
  %t.t6 = load %s1, %s1* %t.t5
  store %s1 %t.t6, %s1* %fn1_tmp
  ; fn1_tmp#1 = fn1_tmp.$0
  %t.t7 = getelementptr inbounds %s1, %s1* %fn1_tmp, i32 0, i32 0
  %t.t8 = load i1, i1* %t.t7
  store i1 %t.t8, i1* %fn1_tmp.1
  ; fn1_tmp#2 = kv.$0
  %t.t9 = getelementptr inbounds %s0, %s0* %kv, i32 0, i32 0
  %t.t10 = load %s1, %s1* %t.t9
  store %s1 %t.t10, %s1* %fn1_tmp.2
  ; fn1_tmp#3 = fn1_tmp#2.$1
  %t.t11 = getelementptr inbounds %s1, %s1* %fn1_tmp.2, i32 0, i32 1
  %t.t12 = load %v0, %v0* %t.t11
  store %v0 %t.t12, %v0* %fn1_tmp.3
  ; fn1_tmp#4 = false
  store i1 0, i1* %fn1_tmp.4
  ; fn1_tmp#5 = kv.$1
  %t.t13 = getelementptr inbounds %s0, %s0* %kv, i32 0, i32 1
  %t.t14 = load %s2, %s2* %t.t13
  store %s2 %t.t14, %s2* %fn1_tmp.5
  ; fn1_tmp#6 = fn1_tmp#5.$0
  %t.t15 = getelementptr inbounds %s2, %s2* %fn1_tmp.5, i32 0, i32 0
  %t.t16 = load double, double* %t.t15
  store double %t.t16, double* %fn1_tmp.6
  ; fn1_tmp#7 = false
  store i1 0, i1* %fn1_tmp.7
  ; fn1_tmp#8 = kv.$1
  %t.t17 = getelementptr inbounds %s0, %s0* %kv, i32 0, i32 1
  %t.t18 = load %s2, %s2* %t.t17
  store %s2 %t.t18, %s2* %fn1_tmp.8
  ; fn1_tmp#9 = fn1_tmp#8.$1
  %t.t19 = getelementptr inbounds %s2, %s2* %fn1_tmp.8, i32 0, i32 1
  %t.t20 = load i64, i64* %t.t19
  store i64 %t.t20, i64* %fn1_tmp.9
  ; fn1_tmp#10 = {fn1_tmp#1,fn1_tmp#3,fn1_tmp#4,fn1_tmp#6,fn1_tmp#7,fn1_tmp#9}
  %t.t21 = load i1, i1* %fn1_tmp.1
  %t.t22 = getelementptr inbounds %s3, %s3* %fn1_tmp.10, i32 0, i32 0
  store i1 %t.t21, i1* %t.t22
  %t.t23 = load %v0, %v0* %fn1_tmp.3
  %t.t24 = getelementptr inbounds %s3, %s3* %fn1_tmp.10, i32 0, i32 1
  store %v0 %t.t23, %v0* %t.t24
  %t.t25 = load i1, i1* %fn1_tmp.4
  %t.t26 = getelementptr inbounds %s3, %s3* %fn1_tmp.10, i32 0, i32 2
  store i1 %t.t25, i1* %t.t26
  %t.t27 = load double, double* %fn1_tmp.6
  %t.t28 = getelementptr inbounds %s3, %s3* %fn1_tmp.10, i32 0, i32 3
  store double %t.t27, double* %t.t28
  %t.t29 = load i1, i1* %fn1_tmp.7
  %t.t30 = getelementptr inbounds %s3, %s3* %fn1_tmp.10, i32 0, i32 4
  store i1 %t.t29, i1* %t.t30
  %t.t31 = load i64, i64* %fn1_tmp.9
  %t.t32 = getelementptr inbounds %s3, %s3* %fn1_tmp.10, i32 0, i32 5
  store i64 %t.t31, i64* %t.t32
  ; merge(bs, fn1_tmp#10)
  %t.t33 = load %v2.bld, %v2.bld* %bs
  %t.t34 = load %s3, %s3* %fn1_tmp.10
  call %v2.bld @v2.bld.merge(%v2.bld %t.t33, %s3 %t.t34, i32 %cur.tid)
  ; end
  br label %body.end
body.end:
  br label %loop.terminator
loop.terminator:
  %t.t35 = load i64, i64* %cur.idx
  %t.t36 = add i64 %t.t35, 1
  store i64 %t.t36, i64* %cur.idx
  br label %loop.start
loop.end:
  ret void
}

define void @f1_wrapper(%v2.bld %fn0_tmp.1, %v1 %kvs, %work_t* %cur.work) {
fn.entry:
  %t.t0 = call i64 @v1.size(%v1 %kvs)
  %t.t1 = call i64 @v1.size(%v1 %kvs)
  %t.t2 = sub i64 %t.t0, 1
  %t.t3 = mul i64 1, %t.t2
  %t.t4 = add i64 %t.t3, 0
  %t.t5 = icmp slt i64 %t.t4, %t.t1
  br i1 %t.t5, label %t.t6, label %fn.boundcheckfailed
t.t6:
  br label %fn.boundcheckpassed
fn.boundcheckfailed:
  %t.t7 = call i64 @weld_rt_get_run_id()
  call void @weld_run_set_errno(i64 %t.t7, i64 5)
  call void @weld_rt_abort_thread()
  ; Unreachable!
  br label %fn.end
fn.boundcheckpassed:
  %t.t8 = icmp ule i64 %t.t0, 16384
  br i1 %t.t8, label %for.ser, label %for.par
for.ser:
  call void @f1(%v2.bld %fn0_tmp.1, %v1 %kvs, %work_t* %cur.work, i64 0, i64 %t.t0)
  call void @f2(%v2.bld %fn0_tmp.1, %work_t* %cur.work)
  br label %fn.end
for.par:
  %t.t9 = insertvalue %s12 undef, %v2.bld %fn0_tmp.1, 0
  %t.t10 = insertvalue %s12 %t.t9, %v1 %kvs, 1
  %t.t11 = getelementptr %s12, %s12* null, i32 1
  %t.t12 = ptrtoint %s12* %t.t11 to i64
  %t.t13 = call i8* @malloc(i64 %t.t12)
  %t.t14 = bitcast i8* %t.t13 to %s12*
  store %s12 %t.t10, %s12* %t.t14
  %t.t15 = insertvalue %s13 undef, %v2.bld %fn0_tmp.1, 0
  %t.t16 = getelementptr %s13, %s13* null, i32 1
  %t.t17 = ptrtoint %s13* %t.t16 to i64
  %t.t18 = call i8* @malloc(i64 %t.t17)
  %t.t19 = bitcast i8* %t.t18 to %s13*
  store %s13 %t.t15, %s13* %t.t19
  call void @weld_rt_start_loop(%work_t* %cur.work, i8* %t.t13, i8* %t.t18, void (%work_t*)* @f1_par, void (%work_t*)* @f2_par, i64 0, i64 %t.t0, i32 16384)
  br label %fn.end
fn.end:
  ret void
}

define void @f1_par(%work_t* %cur.work) {
entry:
  %t.t2 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 0
  %t.t3 = load i8*, i8** %t.t2
  %t.t0 = bitcast i8* %t.t3 to %s12*
  %t.t1 = load %s12, %s12* %t.t0
  %fn0_tmp.1 = extractvalue %s12 %t.t1, 0
  %kvs = extractvalue %s12 %t.t1, 1
  %t.t4 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 1
  %t.t5 = load i64, i64* %t.t4
  %t.t6 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 2
  %t.t7 = load i64, i64* %t.t6
  %t.t8 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 4
  %t.t9 = load i32, i32* %t.t8
  %t.t10 = trunc i32 %t.t9 to i1
  br i1 %t.t10, label %new_pieces, label %fn_call
new_pieces:
  call void @v2.bld.newPiece(%v2.bld %fn0_tmp.1, %work_t* %cur.work)
  br label %fn_call
fn_call:
  call void @f1(%v2.bld %fn0_tmp.1, %v1 %kvs, %work_t* %cur.work, i64 %t.t5, i64 %t.t7)
  ret void
}

define void @f2_par(%work_t* %cur.work) {
entry:
  %t.t2 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 0
  %t.t3 = load i8*, i8** %t.t2
  %t.t0 = bitcast i8* %t.t3 to %s13*
  %t.t1 = load %s13, %s13* %t.t0
  %fn0_tmp.1 = extractvalue %s13 %t.t1, 0
  %t.t4 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 4
  %t.t5 = load i32, i32* %t.t4
  %t.t6 = trunc i32 %t.t5 to i1
  br i1 %t.t6, label %new_pieces, label %fn_call
new_pieces:
  call void @v2.bld.newPiece(%v2.bld %fn0_tmp.1, %work_t* %cur.work)
  br label %fn_call
fn_call:
  call void @f2(%v2.bld %fn0_tmp.1, %work_t* %cur.work)
  ret void
}

define void @f0(%v1 %kvs.in, i32 %partitionIndex.in, %work_t* %cur.work) {
fn.entry:
  %partitionIndex = alloca i32
  %kvs = alloca %v1
  %fn0_tmp.1 = alloca %v2.bld
  %fn0_tmp = alloca i64
  store i32 %partitionIndex.in, i32* %partitionIndex
  store %v1 %kvs.in, %v1* %kvs
  %cur.tid = call i32 @weld_rt_thread_id()
  br label %b.b0
b.b0:
  ; fn0_tmp = len(kvs)
  %t.t0 = load %v1, %v1* %kvs
  %t.t1 = call i64 @v1.size(%v1 %t.t0)
  store i64 %t.t1, i64* %fn0_tmp
  ; fn0_tmp#1 = new appender[{bool,vec[u8],bool,f64,bool,i64}](fn0_tmp)
  %t.t3 = load i64, i64* %fn0_tmp
  %t.t2 = call %v2.bld @v2.bld.new(i64 %t.t3, %work_t* %cur.work, i32 1)
  store %v2.bld %t.t2, %v2.bld* %fn0_tmp.1
  ; for [kvs, ] fn0_tmp#1 bs i kv F1 F2 true
  %t.t4 = load %v2.bld, %v2.bld* %fn0_tmp.1
  %t.t5 = load %v1, %v1* %kvs
  call void @f1_wrapper(%v2.bld %t.t4, %v1 %t.t5, %work_t* %cur.work)
  br label %body.end
body.end:
  ret void
}

define void @f0_par(%work_t* %cur.work) {
  %t.t2 = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 0
  %t.t3 = load i8*, i8** %t.t2
  %t.t0 = bitcast i8* %t.t3 to %s14*
  %t.t1 = load %s14, %s14* %t.t0
  %kvs = extractvalue %s14 %t.t1, 0
  %partitionIndex = extractvalue %s14 %t.t1, 1
  call void @f0(%v1 %kvs, i32 %partitionIndex, %work_t* %cur.work)
  ret void
}

define i64 @run(i64 %r.input) {
  %r.inp_typed = inttoptr i64 %r.input to %input_arg_t*
  %r.inp_val = load %input_arg_t, %input_arg_t* %r.inp_typed
  %r.args = extractvalue %input_arg_t %r.inp_val, 0
  %r.nworkers = extractvalue %input_arg_t %r.inp_val, 1
  %r.memlimit = extractvalue %input_arg_t %r.inp_val, 2
  %r.args_typed = inttoptr i64 %r.args to %s15*
  %r.args_val = load %s15, %s15* %r.args_typed
  %partitionIndex = extractvalue %s15 %r.args_val, 0
  %kvs = extractvalue %s15 %r.args_val, 1
  %t.t0 = insertvalue %s14 undef, %v1 %kvs, 0
  %t.t1 = insertvalue %s14 %t.t0, i32 %partitionIndex, 1
  %t.t2 = getelementptr %s14, %s14* null, i32 1
  %t.t3 = ptrtoint %s14* %t.t2 to i64
  %t.t4 = call i8* @malloc(i64 %t.t3)
  %t.t5 = bitcast i8* %t.t4 to %s14*
  store %s14 %t.t1, %s14* %t.t5
  %t.t6 = call i64 @weld_run_begin(void (%work_t*)* @f0_par, i8* %t.t4, i64 %r.memlimit, i32 %r.nworkers)
  %res_ptr = call i8* @weld_run_get_result(i64 %t.t6)
  %res_address = ptrtoint i8* %res_ptr to i64
  %t.t7 = call i64 @weld_run_get_errno(i64 %t.t6)
  %t.t8 = insertvalue %output_arg_t undef, i64 %res_address, 0
  %t.t9 = insertvalue %output_arg_t %t.t8, i64 %t.t6, 1
  %t.t10 = insertvalue %output_arg_t %t.t9, i64 %t.t7, 2
  %t.t11 = getelementptr %output_arg_t, %output_arg_t* null, i32 1
  %t.t12 = ptrtoint %output_arg_t* %t.t11 to i64
  %t.t13 = call i8* @malloc(i64 %t.t12)
  %t.t14 = bitcast i8* %t.t13 to %output_arg_t*
  store %output_arg_t %t.t10, %output_arg_t* %t.t14
  %t.t15 = ptrtoint %output_arg_t* %t.t14 to i64
  ret i64 %t.t15
}
