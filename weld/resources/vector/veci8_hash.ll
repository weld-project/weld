; Compute the hash code of a vec[i8].
; Generated from the following C code using `clang -emit-llvm -O0 -S -c <code>.c`
;
; #include "stdint.h"
;
; int32_t i8_hash(int8_t); // need to change this to i8.hash in the generated LLVM
; int32_t i64_hash(int64_t); // need to change this to i64.hash in the generated LLVM
; int32_t hash_combine(int32_t, int32_t);
;
; int32_t hash_veci8(int8_t *elements, int64_t size) {{
;   int32_t h = 0;
;   for (int64_t i = 0; i < size/8; i++) {{
;     h = hash_combine(h, i64_hash(((int64_t *)elements)[i]));
;   }}
;   for (int64_t i = size/8*8; i < size; i++) {{
;     h = hash_combine(h, i8_hash(elements[i]));
;   }}
;   return h;
; }}

define i32 @hash_veci8(i8*, i64) #0 {{
  %3 = alloca i8*, align 8
  %4 = alloca i64, align 8
  %5 = alloca i32, align 4
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  store i8* %0, i8** %3, align 8
  store i64 %1, i64* %4, align 8
  store i32 0, i32* %5, align 4
  store i64 0, i64* %6, align 8
  br label %8

; <label>:8:                                      ; preds = %22, %2
  %9 = load i64, i64* %6, align 8
  %10 = load i64, i64* %4, align 8
  %11 = sdiv i64 %10, 8
  %12 = icmp slt i64 %9, %11
  br i1 %12, label %13, label %25

; <label>:13:                                     ; preds = %8
  %14 = load i32, i32* %5, align 4
  %15 = load i64, i64* %6, align 8
  %16 = load i8*, i8** %3, align 8
  %17 = bitcast i8* %16 to i64*
  %18 = getelementptr inbounds i64, i64* %17, i64 %15
  %19 = load i64, i64* %18, align 8
  %20 = call i32 @i64.hash(i64 %19)
  %21 = call i32 @hash_combine(i32 %14, i32 %20)
  store i32 %21, i32* %5, align 4
  br label %22

; <label>:22:                                     ; preds = %13
  %23 = load i64, i64* %6, align 8
  %24 = add nsw i64 %23, 1
  store i64 %24, i64* %6, align 8
  br label %8

; <label>:25:                                     ; preds = %8
  %26 = load i64, i64* %4, align 8
  %27 = sdiv i64 %26, 8
  %28 = mul nsw i64 %27, 8
  store i64 %28, i64* %7, align 8
  br label %29

; <label>:29:                                     ; preds = %41, %25
  %30 = load i64, i64* %7, align 8
  %31 = load i64, i64* %4, align 8
  %32 = icmp slt i64 %30, %31
  br i1 %32, label %33, label %44

; <label>:33:                                     ; preds = %29
  %34 = load i32, i32* %5, align 4
  %35 = load i64, i64* %7, align 8
  %36 = load i8*, i8** %3, align 8
  %37 = getelementptr inbounds i8, i8* %36, i64 %35
  %38 = load i8, i8* %37, align 1
  %39 = call i32 @i8.hash(i8 signext %38)
  %40 = call i32 @hash_combine(i32 %34, i32 %39)
  store i32 %40, i32* %5, align 4
  br label %41

; <label>:41:                                     ; preds = %33
  %42 = load i64, i64* %7, align 8
  %43 = add nsw i64 %42, 1
  store i64 %43, i64* %7, align 8
  br label %29

; <label>:44:                                     ; preds = %29
  %45 = load i32, i32* %5, align 4
  ret i32 %45
}}

define i32 @{NAME}.hash(%{NAME} %vec) {{
  %elementsRaw = extractvalue %{NAME} %vec, 0
  %elementsI8 = bitcast {ELEM}* %elementsRaw to i8*
  %size = extractvalue %{NAME} %vec, 1
  %res = call i32 @hash_veci8(i8* %elementsI8, i64 %size)
  ret i32 %res
}}
