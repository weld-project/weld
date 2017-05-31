; Begin merger merge.
  {t0} = call {elem_ty_str}* {bld_prefix}.getPtrIndexed({bld_ty_str} {bld_tmp}, i32 0)
  {first} = load {elem_ty_str}, {elem_ty_str}* {t0}
  {nworkers} = call i32 @get_nworkers()
  br label %{entry}
{entry}:
  {cond} = icmp ult i32 1, {nworkers}
  br i1 {cond}, label %{body}, label %{done}
{body}:
  {i} = phi i32 [ 1, %{entry} ], [ {i2}, %{body} ]
  {bld_ptr} = call {elem_ty_str}* {bld_prefix}.getPtrIndexed({bld_ty_str} {bld_tmp}, i32 {i})
  {val} = load {elem_ty_str}, {elem_ty_str}* {bld_ptr}
; begin generated merge
