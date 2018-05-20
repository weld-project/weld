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

declare <4 x double> @llvm.fabs.v4f64(<4 x double>)

declare float @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)

declare double @llvm.powi.f64(double, i32)
declare float @llvm.powi.f32(float, i32)

declare float     @llvm.sin.f32(float  %Val)
declare double    @llvm.sin.f64(double %Val)
declare float     @llvm.cos.f32(float  %Val)
declare double    @llvm.cos.f64(double %Val)

declare i64 @llvm.ctlz.i64(i64, i1)

declare float @llvm.maxnum.f32(float, float)
declare double @llvm.maxnum.f64(double, double)
declare float @llvm.minnum.f32(float, float)
declare double @llvm.minnum.f64(double, double)

; std library functions
declare i8* @malloc(i64)
declare void @qsort(i8*, i64, i64, i32 (i8*, i8*)*)
declare i32 @memcmp(i8*, i8*, i64)
declare float @erff(float)
declare double @erf(double)

; Trigonometry functions without LLVM intrinsics.
declare float @tanf(float)
declare double @tan(double)

declare float @asinf(float)
declare double @asin(double)
declare float @acosf(float)
declare double @acos(double)
declare float @atanf(float)
declare double @atan(double)

declare float @sinhf(float)
declare double @sinh(double)
declare float @coshf(float)
declare double @cosh(double)
declare float @tanhf(float)
declare double @tanh(double)

; Power
declare float @llvm.pow.f32(float, float)
declare double @llvm.pow.f64(double, double)

declare i32 @puts(i8* nocapture) nounwind

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
declare void    @weld_rt_new_vb_piece(i8*, %work_t*, i32)
declare %vb.vp* @weld_rt_cur_vb_piece(i8*, i32)
declare %vb.out @weld_rt_result_vb(i8*)

declare i8*     @weld_rt_new_merger(i64, i32)
declare i8*     @weld_rt_get_merger_at_index(i8*, i64, i32)
declare void    @weld_rt_free_merger(i8*)

declare i8*     @weld_rt_dict_new(i32, i32 (i8*, i8*)*, i32, i32, i64, i64)
declare i8*     @weld_rt_dict_lookup(i8*, i32, i8*)
declare void    @weld_rt_dict_put(i8*, i8*)
declare i8*     @weld_rt_dict_finalize_next_local_slot(i8*)
declare i8*     @weld_rt_dict_finalize_global_slot_for_local(i8*, i8*)
declare i8*     @weld_rt_dict_to_array(i8*, i32, i32)
declare i64     @weld_rt_dict_get_size(i8*)
declare void    @weld_rt_dict_free(i8*)

declare i8*     @weld_rt_gb_new(i32, i32 (i8*, i8*)*, i32, i64, i64)
declare void    @weld_rt_gb_merge(i8*, i8*, i32, i8*)
declare i8*     @weld_rt_gb_result(i8*)
declare void    @weld_rt_gb_free(i8*)

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

; ERF implementation for <4 x double> based on https://www.johndcook.com/blog/cpp_erf/
define <4 x double> @erf256_pd(<4 x double> %iinp) {
  ; sign for each value in iinp
  ;%zeros = <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>
  %j0 = fcmp oge <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00,double -0.000000e+00>, %iinp
  %j1 = icmp eq <4 x i1> <i1 0, i1 0, i1 0, i1 0>, %j0

  ; now j0 - j1 should give us the sign of each value in iinp, if only these were not bits.
  %j2 = sext <4 x i1> %j0 to <4 x i32>
  %j3 = sext <4 x i1> %j1 to <4 x i32>
  %j4 = sub <4 x i32> %j2, %j3
  %j5 = sext <4 x i32> %j4 to <4 x i64>
  ;%sign = bitcast <4 x i64> %j5 to <4 x double>
  %sign = sitofp <4 x i64> %j5 to <4 x double>

  %i4 = call <4 x double> @llvm.fabs.v4f64(<4 x double> %iinp)
  %i5 = fmul <4 x double> %i4, <double 3.275911e-01, double 3.275911e-01, double 3.275911e-01, double 3.275911e-01>
  %i6 = fadd <4 x double> %i5, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %i7 = fdiv <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %i6
  %i8 = fmul <4 x double> %i7, <double 0x3FF0FB844255A12D, double 0x3FF0FB844255A12D, double 0x3FF0FB844255A12D, double 0x3FF0FB844255A12D>
  %i9 = fadd <4 x double> %i8, <double 0xBFF7401C57014C39, double 0xBFF7401C57014C39, double 0xBFF7401C57014C39, double 0xBFF7401C57014C39>
  %i10 = fmul <4 x double> %i7, %i9
  %i11 = fadd <4 x double> %i10, <double 0x3FF6BE1C55BAE157, double 0x3FF6BE1C55BAE157, double 0x3FF6BE1C55BAE157, double 0x3FF6BE1C55BAE157>
  %i12 = fmul <4 x double> %i7, %i11
  %i13 = fadd <4 x double> %i12, <double 0xBFD23531CC3C1469, double 0xBFD23531CC3C1469, double 0xBFD23531CC3C1469, double 0xBFD23531CC3C1469>
  %i14 = fmul <4 x double> %i7, %i13
  %i15 = fadd <4 x double> %i14, <double 0x3FD04F20C6EC5A7E, double 0x3FD04F20C6EC5A7E, double 0x3FD04F20C6EC5A7E, double 0x3FD04F20C6EC5A7E>
  %i16 = fmul <4 x double> %i7, %i15
  %i17 = fmul <4 x double> %iinp, %iinp
  %i18 = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %i17
  %i19 = call <4 x double> @llvm.exp.v4f64(<4 x double> %i18)
  %i20 = fmul <4 x double> %i19, %i16
  %i21 = fsub <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %i20

  %i22 = fmul <4 x double> %i21, %sign
  ret <4 x double> %i22
}
