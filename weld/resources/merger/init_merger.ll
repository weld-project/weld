  ; Copy the initial value into the first merger
  {nworkers} = call i32 @get_nworkers()
  {first_raw} = call {bld_ty_str} {bld_prefix}.getPtrIndexed({bld_ty_str} {bld_inp}, i32 0)
  {first} = call {elem_type}* {bld_prefix}.scalarMergePtr({bld_ty_str} {first_raw})
  call void {bld_prefix}.clearVector({bld_ty_str} {first_raw}, {elem_type} {iden_elem})
  store {elem_type} {init_elem}, {elem_type}* {first}
  br label %{entry}

{entry}:
  {cond} = icmp ult i32 1, {nworkers}
  br i1 {cond}, label %{body}, label %{done}

{body}:
  {i} = phi i32 [ 1, %{entry} ], [ {i2}, %{body} ]
  {cur_bld_ptr} = call {bld_ty_str} {bld_prefix}.getPtrIndexed({bld_ty_str} {bld_inp}, i32 {i})
  {cur_ptr} = call {elem_type}* {bld_prefix}.scalarMergePtr({bld_ty_str} {cur_bld_ptr})
  call void {bld_prefix}.clearVector({bld_ty_str} {cur_bld_ptr}, {elem_type} {iden_elem})
  store {elem_type} {iden_elem}, {elem_type}* {cur_ptr}
  {i2} = add i32 {i}, 1
  {cond2} = icmp ult i32 {i2}, {nworkers}
  br i1 {cond2}, label %{body}, label %{done}

{done}:
