; Begin vecmerger merge
  {nworkers} = call i32 @get_nworkers()
  {t0} = call {bldType} {bldPrefix}.getPtrIndexed({bldType} {buildPtr}, i32 0)
  {typedPtr} = bitcast {bldType} {t0} to {resType}*
  {firstVec} = load {resType}* {typedPtr}
  {size} = call i64 {resPrefix}.size({resType} {firstVec})
  {retValue} = call {resType} {resPrefix}.clone({resType} {firstVec})
  br label %{entry}
{entry}:
  {cond} = icmp ult i32 1, {nworkers}
  br i1 {cond}, label %{bodyLabel}, label %{doneLabel}
{bodyLabel}:
  {i} = phi i32 [ 1, %{entry} ], [ {i2}, %{copyDoneLabel} ]
  {vecPtr} = call {resType}* {bldPrefix}.getPtrIndexed({bldType} {typedPtr}, i32 {i})
  {curVec} = load {resType}* {vecPtr}
  br label %{copyEntryLabel}
{copyEntryLabel}:
  {copyCond} = icmp ult i64 0, {size}
  br i1 {copyCond}, label %{copyBodyLabel}, label %{copyDoneLabel}
{copyBodyLabel}:
  {j} = phi i64 [ 0, %{copyEntryLabel} ], [ {j2}, %{copyBodyLabel} ]
  {elemPtr} = call {elemType}* {resPrefix}.at({resType} {curVec}, i64 {j})
  {mergeValue} = load {elemType}* {elemPtr}
  {mergePtr} = call {elemType}* {resPrefix}.at({resType} {retValue}, i64 {j})
; Insert gen_merge(mergePtr, mergeValue, ...) below.

  
