; Vector extension to generate comparison functions using memcmp.
; This is safe for unsigned integer types and booleans.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix

; Compare two vectors of characters lexicographically.
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
  %cmp = call i32 @memcmp(i8* %elemsA, i8* %elemsB, i64 %minSize)
  %ne = icmp ne i32 %cmp, 0
  br i1 %ne, label %return, label %done

return:
  ret i32 %cmp

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
  %elemsA = extractvalue %{NAME} %a, 0
  %elemsB = extractvalue %{NAME} %b, 0
  %cmp = call i32 @memcmp(i8* %elemsA, i8* %elemsB, i64 %sizeA)
  %res = icmp eq i32 %cmp, 0
  ret i1 %res
different_size:
  ret i1 0
}}
