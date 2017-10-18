; end generate vector collapse
  {i2_v} = add i32 {i_v}, 1
  {cond2_v} = icmp ult i32 {i2_v}, {vector_width}
  br i1 {cond2_v}, label %{body_v}, label %{done_v}
{done_v}:
  {as_ptr} = call i8* {bld_prefix}.getGlobal({bld_ty_str}* {bld_sym})
  ;call void @weld_rt_free_merger(i8* {as_ptr})
