define i32 {TYPE_PREFIX}.eq_on_pointers(i8* %a, i8* %b) {{
  %aTypedPtr = bitcast i8* %a to {TYPE}*
  %bTypedPtr = bitcast i8* %b to {TYPE}*
  %aTyped = load {TYPE}, {TYPE}* %aTypedPtr
  %bTyped = load {TYPE}, {TYPE}* %bTypedPtr
  %resultBool = call i1 {TYPE_PREFIX}.eq({TYPE} %aTyped, {TYPE} %bTyped)
  %result = zext i1 %resultBool to i32
  ret i32 %result
}}