; Common prelude to add at the start of generated LLVM modules.

; LLVM intrinsic functions
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)
declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i32, i1)
declare float @llvm.exp.f32(float)
declare double @llvm.exp.f64(double)
declare i64 @llvm.ctlz.i64(i64, i1)

; std library functions
declare i8* @malloc(i64)
declare void @qsort(i8*, i64, i64, i32 (i8*, i8*)*)

; Weld library functions
declare void    @weld_rt_init(i64)
declare i8*     @weld_rt_malloc(i64, i64)
declare i8*     @weld_rt_realloc(i64, i8*, i64)
declare void    @weld_rt_free(i64, i8*)
declare i32     @my_id_public()
declare void    @weld_rt_set_errno(i64, i64)
declare i64     @weld_rt_get_errno(i64)
declare void    @weld_abort_thread()

; Parallel runtime structures
; documentation in parlib.h
%work_t = type { i8*, i64, i64, i64, i32, i64*, i64*, i32, i64, void (%work_t*)*, %work_t*, i32, i32, i32 }
; documentation in vb.cpp
%vb.vp = type { i8*, i64, i64, i64*, i64*, i32 }
%vb.out = type { i8*, i64 }

; Input argument (input data pointer, nworkers, mem_limit)
%input_arg_t = type { i64, i32, i64 }
; Return type (output data pointer, run ID, errno)
%output_arg_t = type { i64, i64, i64 }

; documenation in parlib.cpp
declare void @set_result(i8*)
declare i8* @get_result()
declare void @pl_start_loop(%work_t*, i8*, i8*, void (%work_t*)*, void (%work_t*)*, i64, i64, i32)
declare void @execute(void (%work_t*)*, i8*)
; documentation in vb.cpp
declare i8* @new_vb(i64, i64)
declare void @new_piece(i8*, %work_t*)
declare %vb.vp* @cur_piece(i8*, i32)
declare %vb.out @result_vb(i8*)
; merger.cpp
declare i8* @new_merger(i64, i32)
declare i8* @get_merger_at_index(i8*, i64, i32)
declare void @free_merger(i8*)

; Number of workers
declare void @set_nworkers(i32)
declare i32 @get_nworkers()

; Run IDs
declare void @set_runid(i64)
declare i64 @get_runid()

; Hash functions

; Same as Boost's hash_combine; obtained by compiling that with clang
define i64 @hash_combine(i64 %seed, i64 %value) {
  ; return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
  %1 = add i64 %value, 2654435769   ; TODO: should this be 64-bit?
  %2 = shl i64 %seed, 6
  %3 = add i64 %1, %2
  %4 = lshr i64 %seed, 2
  %5 = add i64 %3, %4
  %6 = xor i64 %5, %seed
  ret i64 %6
}

define i64 @i64.hash(i64 %arg) {
  ret i64 %arg
}

define i64 @i32.hash(i32 %arg) {
  %1 = zext i32 %arg to i64
  ret i64 %1
}

define i64 @i8.hash(i8 %arg) {
  %1 = zext i8 %arg to i64
  ret i64 %1
}

define i64 @i1.hash(i1 %arg) {
  %1 = zext i1 %arg to i64
  ret i64 %1
}

define i64 @float.hash(float %arg) {
  %1 = bitcast float %arg to i32
  %2 = zext i32 %1 to i64
  ret i64 %2
}

define i64 @double.hash(double %arg) {
  %1 = bitcast double %arg to i64
  ret i64 %1
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
