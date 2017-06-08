; End generated merge.
  {i2} = add i32 {i}, 1
  {cond2} = icmp ult i32 {i2}, {nworkers}
  br i1 {cond2}, label %{body}, label %{done}
{done}:
  {final_val} = load {res_ty_str}, {res_ty_str}* {t0}
  {as_ptr} = call i8* {bld_prefix}.getRawPtr({bld_ty_str} {bld_tmp})
  ;call void @free_merger(i8* {as_ptr})
  store {res_ty_str} {final_val}, {res_ty_str}*  %{output}
