; Compute the hash code of a vector.
; TODO: We should hash more bytes at a time if elements are non-pointer types.
define i32 @{NAME}.hash(%{NAME} %vec) {{
entry:
  %elements = extractvalue %{NAME} %vec, 0
  %size = extractvalue %{NAME} %vec, 1
  %cond = icmp ult i64 0, %size
  br i1 %cond, label %body, label %done

body:
  %i = phi i64 [ 0, %entry ], [ %i2, %body ]
  %prevHash = phi i32 [ 0, %entry ], [ %newHash, %body ]
  %ptr = getelementptr {ELEM}, {ELEM}* %elements, i64 %i
  %elem = load {ELEM}, {ELEM}* %ptr
  %elemHash = call i32 {ELEM_PREFIX}.hash({ELEM} %elem)
  %newHash = call i32 @hash_combine(i32 %prevHash, i32 %elemHash)
  %i2 = add i64 %i, 1
  %cond2 = icmp ult i64 %i2, %size
  br i1 %cond2, label %body, label %done

done:
  %res = phi i32 [ 0, %entry ], [ %newHash, %body ]
  ret i32 %res
}}
