; End generated merge.
  {i2} = add i32 {i}, 1
  {cond2} = icmp ult i32 {i2}, {nworkers}
  br i1 {cond2}, label %{body}, label %{done}
{done}:
  {scalar_val_2} = load {res_ty_str}, {res_ty_str}* {scalar_ptr}
  store {res_ty_str} {scalar_val_2}, {res_ty_str}*  {output}
  {final_val_vec} = load {elem_vec_ty_str}, {elem_vec_ty_str}* {vector_ptr}
  br label %{entry_v}
{entry_v}:
  {cond_v} = icmp ult i32 0, {vector_width}
  br i1 {cond_v}, label %{body_v}, label %{done_v}
{body_v}:
  {i_v} = phi i32 [ 0, %{entry_v} ], [ {i2_v}, %{body_v} ]
  {val_v} = extractelement {elem_vec_ty_str} {final_val_vec}, i32 {i_v}
; begin generate vector collapse
