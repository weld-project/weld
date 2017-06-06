  ; Copy the initial value into the first merger
  {nworkers} = call i32 @get_nworkers()
  {first} = call {bld_ty_str} {bld_prefix}.getPtrIndexed({bld_ty_str} {bld_inp}, i32 0)
  store {elem_type} {init_elem}, {bld_ty_str} {first}
  br label %{entry}

{entry}:
  {cond} = icmp ult i32 1, {nworkers}
  br i1 {cond}, label %{body}, label %{done}

{body}:
  {i} = phi i32 [ 1, %{entry} ], [ {i2}, %{body} ]
  {cur_ptr} = call {bld_ty_str} {bld_prefix}.getPtrIndexed({bld_ty_str} {bld_inp}, i32 {i})
  store {elem_type} {iden_elem}, {bld_ty_str} {cur_ptr}
  {i2} = add i32 {i}, 1
  {cond2} = icmp ult i32 {i2}, {nworkers}
  br i1 {cond2}, label %{body}, label %{done}

{done}:
