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
declare void    @weld_runtime_init()

declare i64     @weld_run_begin(void (%work_t*)*, i8*, i64, i32)
declare i8*     @weld_run_get_result(i64)
declare void    @weld_run_dispose(i64)

declare i8*     @weld_run_malloc(i64, i64)
declare i8*     @weld_run_realloc(i64, i8*, i64)
declare void    @weld_run_free(i64, i8*)
declare i64     @weld_run_memory_usage(i64)

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

define i64 @run_dispose(i64 %run_id) {
    call void @weld_run_dispose(i64 %run_id)
    ret i64 0
}

define i64 @runtime_init(i64 %unused) {
    call void @weld_runtime_init()
    ret i64 0
}

define i64 @run_memory_usage(i64 %run_id) {
    %res = call i64 @weld_run_memory_usage(i64 %run_id)
    ret i64 %res
}

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
