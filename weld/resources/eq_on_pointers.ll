define i32 {KEY_PREFIX}.eq_on_pointers(i8* %a, i8* %b) {{
  %aKeyPtr = bitcast i8* %a to {KEY}*
  %bKeyPtr = bitcast i8* %b to {KEY}*
  %aKey = load {KEY}, {KEY}* %aKeyPtr
  %bKey = load {KEY}, {KEY}* %bKeyPtr
  %resultBool = call i1 {KEY_PREFIX}.eq({KEY} %aKey, {KEY} %bKey)
  %result = zext i1 %resultBool to i32
  ret i32 %result
}}