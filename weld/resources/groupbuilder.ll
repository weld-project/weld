; Templated function for the finalization step of GroupBuilder. We implement GroupBuilder using
; a VecBuilder of {key, value} structs, and we need to turn it into a dict[key, vec[value]].
; We do this by sorting the elements by key and then inserting them into a dictionary.
;
; Parameters:
; - NAME: name of generated function
; - KEY: LLVM type of key (e.g. i32 or %MyStruct)
; - VALUE: LLVM type of value (e.g. i32 or %MyStruct)
; - KEY_PREFIX: prefix for helper functions of key (e.g. @i32 or @MyStruct)
; - KV_STRUCT: name of struct holding {KEY, VALUE} (should be generated outside)
; - KV_VEC: name of vector of KV_STRUCTs (should be generated outside)
; - KV_VEC_PREFIX: prefix for helper functions of KV_VEC
; - VALUE_VEC: name of vector of VALUEs
; - VALUE_VEC_PREFIX: prefix for helper functions of VALUE_VEC
; - DICT: name of generated dictionary (of (key, vec[value]))
; - DICT_PREFIX: prefix for helper functions of DICT

define $DICT @$NAME($KV_VEC.bld %pairs) {
entry:
  %size = call i64 $KV_VEC_PREFIX.bld.size($KV_VEC.bld %pairs, i32 0)
  %elements = call %$KV_STRUCT* $KV_VEC_PREFIX.bld.at($KV_VEC.bld %pairs, i64 0, i32 0)
  %elementsRaw = bitcast %$KV_STRUCT* %elements to i8*
  %elemPtr = getelementptr %$KV_STRUCT, %$KV_STRUCT* null, i64 1
  %elemSize = ptrtoint %$KV_STRUCT* %elemPtr to i64
  call void @qsort(i8* %elementsRaw, i64 %size, i64 %elemSize, i32 (i8*, i8*)* @$NAME.helper)
  %dict = call $DICT $DICT_PREFIX.new(i64 16)
  br label %outerLoop

  ; We have two loops: for each element, we check how many of the ones after have the same key,
  ; and turn those into a vector that we then place into the dictionary
outerLoop:
  %startPos = phi i64 [ 0, %entry ], [ %endPos2, %copyLoopDone ]
  %dict2 = phi $DICT [ %dict, %entry ], [ %dict3, %copyLoopDone ]
  %cond = icmp uge i64 %startPos, %size
  br i1 %cond, label %done, label %outerLoop2

outerLoop2:
  %keyPtr = getelementptr %$KV_STRUCT, %$KV_STRUCT* %elements, i64 %startPos, i32 0
  %key = load $KEY, $KEY* %keyPtr
  %endPos = add i64 %startPos, 1
  br label %innerLoop

innerLoop:
  %endPos2 = phi i64 [ %endPos, %outerLoop2 ], [ %endPos3, %innerLoop2 ]
  %cond2 = icmp uge i64 %endPos2, %size
  br i1 %cond2, label %innerLoopDone, label %innerLoop2

innerLoop2:
  %keyPtr2 = getelementptr %$KV_STRUCT, %$KV_STRUCT* %elements, i64 %endPos2, i32 0
  %key2 = load $KEY, $KEY* %keyPtr2
  %cmp = call i32 $KEY_PREFIX.cmp($KEY %key, $KEY %key2)
  %ne = icmp ne i32 %cmp, 0
  %endPos3 = add i64 %endPos2, 1
  br i1 %ne, label %innerLoopDone, label %innerLoop

innerLoopDone:
  %groupSize = sub i64 %endPos2, %startPos
  %startPtr = getelementptr %$KV_STRUCT, %$KV_STRUCT* %elements, i64 %startPos
  %newVec = call $VALUE_VEC $VALUE_VEC_PREFIX.new(i64 %groupSize)
  br label %copyLoop

copyLoop:
  %j = phi i64 [ 0, %innerLoopDone ], [ %j2, %copyLoop2 ]
  %cond3 = icmp uge i64 %j, %groupSize
  br i1 %cond3, label %copyLoopDone, label %copyLoop2

copyLoop2:
  %pos = add i64 %startPos, %j
  %valuePtr = getelementptr %$KV_STRUCT, %$KV_STRUCT* %elements, i64 %pos, i32 1
  %value = load $VALUE, $VALUE* %valuePtr
  %destPtr = call $VALUE* $VALUE_VEC_PREFIX.at($VALUE_VEC %newVec, i64 %j)
  store $VALUE %value, $VALUE* %destPtr
  %j2 = add i64 %j, 1
  br label %copyLoop

copyLoopDone:
  %slot = call $DICT.slot $DICT_PREFIX.lookup($DICT %dict2, $KEY %key)
  %dict3 = call $DICT $DICT_PREFIX.put($DICT %dict2, $DICT.slot %slot, $KEY %key, $VALUE_VEC %newVec)
  br label %outerLoop

done:
  ret $DICT %dict2
}

; Helper function that compares two $KV_STRUCT* by key (but takes i8* for use with qsort).
define i32 @$NAME.helper(i8* %p1, i8* %p2) {
  %kv1 = bitcast i8* %p1 to %$KV_STRUCT*
  %kv2 = bitcast i8* %p2 to %$KV_STRUCT*
  %kPtr1 = getelementptr %$KV_STRUCT, %$KV_STRUCT* %kv1, i64 0, i32 0
  %kPtr2 = getelementptr %$KV_STRUCT, %$KV_STRUCT* %kv2, i64 0, i32 0
  %k1 = load $KEY, $KEY* %kPtr1
  %k2 = load $KEY, $KEY* %kPtr2
  %res = call i32 $KEY_PREFIX.cmp($KEY %k1, $KEY %k2)
  ret i32 %res
}
