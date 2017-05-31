; End generated merge.
  {i2} = add i32 {i}, 1
  {cond2} = icmp ult i32 {i2}, {nworkers}
  br i1 {cond2}, label %{body}, label %{done}
{done}:
  {final_val_vec} = load {res_ty_str}, {res_ty_str}* {t0}
; Collapse the value into a single scalar.
  {first_v} = extractelement {elem_ty_str} {final_val_vec}, i32 0
  store {res_ty_str} {first_v}, {res_ty_str}*  %{output}
  br label %{entry_v}
{entry_v}:
  {cond_v} = icmp ult i32 1, {vector_width}
  br i1 {cond_v}, label %{body_v}, label %{done_v}
{body_v}:
  {i_v} = phi i32 [ 1, %{entry_v} ], [ {i2_v}, %{body_v} ]
  {val_v} = extractelement {elem_ty_str} {final_val_vec}, i32 {i_v}
; begin generate vector collapse



