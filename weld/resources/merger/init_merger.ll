  ; Copy the initial value into the first merger
  {nworkers} = call i32 @weld_rt_get_nworkers()
  {first_raw} = call {bld_ty_str} {bld_prefix}.getPtrIndexed({bld_ty_str} {bld_inp}, i32 0)
  call void {bld_prefix}.clearPiece({bld_ty_str} {first_raw}, {elem_type} {iden_elem})
  {first} = call {elem_type}* {bld_prefix}.scalarMergePtr({bld_ty_str} {first_raw})
  store {elem_type} {init_elem}, {elem_type}* {first}
  br label %{entry}

{entry}:
  {cond} = icmp ult i32 1, {nworkers}
  br i1 {cond}, label %{body}, label %{done}

{body}:
  {i} = phi i32 [ 1, %{entry} ], [ {i2}, %{body} ]
  {cur_bld_ptr} = call {bld_ty_str} {bld_prefix}.getPtrIndexed({bld_ty_str} {bld_inp}, i32 {i})
  call void {bld_prefix}.clearPiece({bld_ty_str} {cur_bld_ptr}, {elem_type} {iden_elem})
  {i2} = add i32 {i}, 1
  {cond2} = icmp ult i32 {i2}, {nworkers}
  br i1 {cond2}, label %{body}, label %{done}

{done}:
