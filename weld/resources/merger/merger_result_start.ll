; Begin merger merge.
  {t0} = call {bld_ty_str} {bld_prefix}.getPtrIndexed({bld_ty_str} {bld_tmp}, i32 0)
  {scalar_ptr} = call {elem_ty_str}* {bld_prefix}.scalarMergePtr({bld_ty_str} {t0})
  {vector_ptr} = call {elem_vec_ty_str}* {bld_prefix}.vectorMergePtr({bld_ty_str} {t0})
  {first_scalar} = load {elem_ty_str}, {elem_ty_str}* {scalar_ptr}
  {first_vector} = load {elem_vec_ty_str}, {elem_vec_ty_str}* {vector_ptr}
  {nworkers} = call i32 @weld_rt_get_nworkers()
  br label %{entry}
{entry}:
  {cond} = icmp ult i32 1, {nworkers}
  br i1 {cond}, label %{body}, label %{done}
{body}:
  {i} = phi i32 [ 1, %{entry} ], [ {i2}, %{body} ]
  {bld_ptr} = call {bld_ty_str} {bld_prefix}.getPtrIndexed({bld_ty_str} {bld_tmp}, i32 {i})
  {val_scalar_ptr} = call {elem_ty_str}* {bld_prefix}.scalarMergePtr({bld_ty_str} {bld_ptr})
  {val_vector_ptr} = call {elem_vec_ty_str}* {bld_prefix}.vectorMergePtr({bld_ty_str} {bld_ptr})
  {val_scalar} = load {elem_ty_str}, {elem_ty_str}* {val_scalar_ptr}
  {val_vector} = load {elem_vec_ty_str}, {elem_vec_ty_str}* {val_vector_ptr}
; begin generated merge
