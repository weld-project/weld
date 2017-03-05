; Insert gen_merge(mergePtr, mergeValue, ...) above.
  {j2} = add i64 {j}, 1
  {copyCond2} = icmp ult i64 {j2}, {size}
  br i1 {copyCond2}, label %{copyBodyLabel}, label %{copyDoneLabel}
{copyDoneLabel}:
  {i2} = add i32 {i}, 1
  {cond2} = icmp ult i32 {i2}, {nworkers}
  br i1 {cond2}, label %{bodyLabel}, label %{doneLabel}
{doneLabel}:
  store {resType} {retValue}, {resType}* {output}
  {rawPtr} = bitcast {bldType} {buildPtr} to i8*
  call void @free_merger(i8* {rawPtr})
; End vecmerger merge.
