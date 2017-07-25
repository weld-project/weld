; Template for a dictmerger and its helper functions.
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - KEY: LLVM type of key (e.g. i32 or %MyStruct)
; - VALUE: LLVM type of value (e.g. i32 or %MyStruct)
; - KV_STRUCT: name of struct holding {{KEY, VALUE}} (should be generated outside)
; - KV_VEC: name of vector of KV_STRUCTs (should be generated outside)
; - KV_VEC_PREFIX: prefix for helper functions of KV_VEC
;
; In addition, the function {NAME}.bld.merge_op({VALUE}, {VALUE}) is expected to be
; defined, implementing the operation needed to merge two values.

%{NAME}.bld = type i8* ; the dictmerger is a pointer to the corresponding dictionary

; Initialize and return a new dictionary with the given initial capacity.
; The capacity must be a power of 2.
define %{NAME}.bld @{NAME}.bld.new(i64 %capacity) {{
  %nworkers = call i32 @weld_rt_get_nworkers()
  %structSizePtr = getelementptr %{NAME}, %{NAME}* null, i32 1
  %structSize = ptrtoint %{NAME}* %structSizePtr to i64
  %bldPtr = call i8* @weld_rt_new_merger(i64 %structSize, i32 %nworkers)
  br label %entry

entry:
  %cond = icmp ult i32 0, %nworkers
  br i1 %cond, label %body, label %done

body:
  %i = phi i32 [ 0, %entry ], [ %i2, %body ]
  %dict = call %{NAME}* @{NAME}.bld.getptrIndexed(%{NAME}.bld %bldPtr, i32 %i)
  %newDict = call %{NAME} @{NAME}.new(i64 %capacity)
  store %{NAME} %newDict, %{NAME}* %dict, align 1
  %i2 = add i32 %i, 1
  %cond2 = icmp ult i32 %i2, %nworkers
  br i1 %cond2, label %body, label %done

done:
  ret %{NAME}.bld %bldPtr
}}

define %{NAME}* @{NAME}.bld.getptrIndexed(%{NAME}.bld %bldPtr, i32 %i) alwaysinline {{
  %dictPtr = getelementptr %{NAME}, %{NAME}* null, i32 1
  %dictSize = ptrtoint %{NAME}* %dictPtr to i64

  %rawPtr = call i8* @weld_rt_get_merger_at_index(%{NAME}.bld %bldPtr, i64 %dictSize, i32 %i)
  %ptr = bitcast i8* %rawPtr to %{NAME}*
  ret %{NAME}* %ptr
}}

; Append a value into a builder, growing its space if needed.
define %{NAME}.bld @{NAME}.bld.merge(%{NAME}.bld %bldPtr, %{KV_STRUCT} %keyValue, i32 %workerId) {{
entry:
  %bldPtrLocal = call %{NAME}* @{NAME}.bld.getptrIndexed(%{NAME}.bld %bldPtr, i32 %workerId)
  %bld = load %{NAME}, %{NAME}* %bldPtrLocal
  %key = extractvalue %{KV_STRUCT} %keyValue, 0
  %value = extractvalue %{KV_STRUCT} %keyValue, 1
  %slot = call %{NAME}.slot @{NAME}.lookup(%{NAME} %bld, {KEY} %key)
  %filled = call i1 @{NAME}.slot.filled(%{NAME}.slot %slot)
  br i1 %filled, label %onFilled, label %onEmpty

onFilled:
  %oldValue = call {VALUE} @{NAME}.slot.value(%{NAME}.slot %slot)
  %newValue = call {VALUE} @{NAME}.bld.merge_op({VALUE} %oldValue, {VALUE} %value)
  %res1 = call %{NAME} @{NAME}.put(%{NAME} %bld, %{NAME}.slot %slot, {KEY} %key, {VALUE} %newValue)
  br label %done

onEmpty:
  %res2 = call %{NAME} @{NAME}.put(%{NAME} %bld, %{NAME}.slot %slot, {KEY} %key, {VALUE} %value)
  br label %done

done:
  %res = phi %{NAME} [ %res1, %onFilled ], [ %res2, %onEmpty ]
  store %{NAME} %res, %{NAME}* %bldPtrLocal
  ret %{NAME}.bld %bldPtr
}}

; Complete building a vector, trimming any extra space left while growing it.
define %{NAME} @{NAME}.bld.result(%{NAME}.bld %bldPtr) {{
  %finalDictPtr = alloca %{NAME}
  %emptyDict = call %{NAME} @{NAME}.new(i64 16)
  store %{NAME} %emptyDict, %{NAME}* %finalDictPtr
  br label %entryLabel

entryLabel:
  %nworkers = call i32 @weld_rt_get_nworkers()
  br label %bodyLabel

bodyLabel:
  %i = phi i32 [ 0, %entryLabel ], [ %i2, %bodyEndLabel ]
  %dictPtr = call %{NAME}* @{NAME}.bld.getptrIndexed(%{NAME}.bld %bldPtr, i32 %i)
  %dict = load %{NAME}, %{NAME}* %dictPtr
  %kvVec = call {KV_VEC} @{NAME}.tovec(%{NAME} %dict)
  %kvVecSize = call i64 {KV_VEC_PREFIX}.size({KV_VEC} %kvVec)
  %emptyKVVec = icmp ult i64 0, %kvVecSize
  br i1 %emptyKVVec, label %innerBodyLabel, label %bodyEndLabel

innerBodyLabel:
  %j = phi i64 [ 0, %bodyLabel ], [ %j2, %innerBodyEndLabel ]
  %elemVarPtr = call %{KV_STRUCT}* {KV_VEC_PREFIX}.at({KV_VEC} %kvVec, i64 %j)
  %elemVar = load %{KV_STRUCT}, %{KV_STRUCT}* %elemVarPtr
  %key = extractvalue %{KV_STRUCT} %elemVar, 0
  %value = extractvalue %{KV_STRUCT} %elemVar, 1
  %finalDict = load %{NAME}, %{NAME}* %finalDictPtr
  %slot = call %{NAME}.slot @{NAME}.lookup(%{NAME} %finalDict, {KEY} %key)
  %filled = call i1 @{NAME}.slot.filled(%{NAME}.slot %slot)
  br i1 %filled, label %onFilled, label %onEmpty

onFilled:
  %finalDict2 = load %{NAME}, %{NAME}* %finalDictPtr
  %oldValue = call {VALUE} @{NAME}.slot.value(%{NAME}.slot %slot)
  %newValue = call {VALUE} @{NAME}.bld.merge_op({VALUE} %oldValue, {VALUE} %value)
  %res1 = call %{NAME} @{NAME}.put(%{NAME} %finalDict2, %{NAME}.slot %slot, {KEY} %key, {VALUE} %newValue)
  br label %done

onEmpty:
  %finalDict3 = load %{NAME}, %{NAME}* %finalDictPtr
  %res2 = call %{NAME} @{NAME}.put(%{NAME} %finalDict3, %{NAME}.slot %slot, {KEY} %key, {VALUE} %value)
  br label %done

done:
  %res = phi %{NAME} [ %res1, %onFilled ], [ %res2, %onEmpty ]
  store %{NAME} %res, %{NAME}* %finalDictPtr
  br label %innerBodyEndLabel

innerBodyEndLabel:
  %j2 = add i64 %j, 1
  %cond2 = icmp ult i64 %j2, %kvVecSize
  br i1 %cond2, label %innerBodyLabel, label %bodyEndLabel

bodyEndLabel:
  %i2 = add i32 %i, 1
  %cond1 = icmp ult i32 %i2, %nworkers
  br i1 %cond1, label %bodyLabel, label %freeLabel

freeLabel:
  %freeCond = icmp ult i32 0, %nworkers
  br i1 %freeCond, label %freeBody, label %endLabel

freeBody:
  %k = phi i32 [ 0, %freeLabel ], [ %k2, %freeBody ]
  %dictPtr2 = call %{NAME}* @{NAME}.bld.getptrIndexed(%{NAME}.bld %bldPtr, i32 %k)
  %dict2 = load %{NAME}, %{NAME}* %dictPtr2
  call void @{NAME}.free(%{NAME} %dict2)
  %k2 = add i32 %k, 1
  %freeCond2 = icmp ult i32 %k2, %nworkers
  br i1 %freeCond2, label %freeBody, label %endLabel

endLabel:
  call void @weld_rt_free_merger(i8* %bldPtr)
  %finalRes = load %{NAME}, %{NAME}* %finalDictPtr
  ret %{NAME} %finalRes
}}

; Dummy hash function; this is needed for structs that use these dictmergers as fields.
define i32 @{NAME}.bld.hash(%{NAME}.bld %bld) {{
  ret i32 0
}}

; Dummy comparison function; this is needed for structs that use these dictmergers as fields.
define i32 @{NAME}.bld.cmp(%{NAME}.bld %bld1, %{NAME}.bld %bld2) {{
  ret i32 -1
}}

; Dummy equality function
define i1 @{NAME}.bld.eq(%{NAME}.bld %bld1, %{NAME}.bld %bld2) {{
  ret i1 0
}}
