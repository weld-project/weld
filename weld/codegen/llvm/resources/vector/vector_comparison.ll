; Vector extension to generate comparison functions. We use a different file,
; vector_comparisons_i8.ll, for comparisons on arrays of bytes, which are optimized.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element (e.g. i32 or %MyStruct)
; - ELEM_PREFIX: prefix for helper functions on ELEM (e.g. @i32 or @MyStruct)



; Compare two vectors lexicographically.
define i32 @{NAME}.cmp(%{NAME} %a, %{NAME} %b) {{
entry:
  %elemsA = extractvalue %{NAME} %a, 0
  %elemsB = extractvalue %{NAME} %b, 0
  %sizeA = extractvalue %{NAME} %a, 1
  %sizeB = extractvalue %{NAME} %b, 1
  %cond1 = icmp ult i64 %sizeA, %sizeB
  %minSize = select i1 %cond1, i64 %sizeA, i64 %sizeB
  %cond = icmp ult i64 0, %minSize
  br i1 %cond, label %body, label %done

body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body2 ]
  %ptrA = getelementptr {ELEM}, {ELEM}* %elemsA, i64 %i
  %ptrB = getelementptr {ELEM}, {ELEM}* %elemsB, i64 %i
  %elemA = load {ELEM}, {ELEM}* %ptrA
  %elemB = load {ELEM}, {ELEM}* %ptrB
  %cmp = call i32 {ELEM_PREFIX}.cmp({ELEM} %elemA, {ELEM} %elemB)
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
}}

; Compare two vectors for equality.
define i1 @{NAME}.eq(%{NAME} %a, %{NAME} %b) {{
  %sizeA = extractvalue %{NAME} %a, 1
  %sizeB = extractvalue %{NAME} %b, 1
  %cond = icmp eq i64 %sizeA, %sizeB
  br i1 %cond, label %same_size, label %different_size
same_size:
  %cmp = call i32 @{NAME}.cmp(%{NAME} %a, %{NAME} %b)
  %res = icmp eq i32 %cmp, 0
  ret i1 %res
different_size:
  ret i1 0
}}
