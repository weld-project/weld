; Begin merger merge.
  {is_global} = call i1 {bld_prefix}.isGlobal({bld_ty_str}* {bld_sym})
  {scalar_ptr} = call {elem_ty_str}* {bld_prefix}.scalarMergePtrForStackPiece({bld_ty_str}* {bld_sym})
  {vector_ptr} = call {elem_vec_ty_str}* {bld_prefix}.vectorMergePtrForStackPiece({bld_ty_str}* {bld_sym})
  br i1 {is_global}, label %{entry}, label %{done}
{entry}:
  {nworkers} = call i32 @weld_rt_get_nworkers()
  br label %{body}
{body}:
  {i} = phi i32 [ 0, %{entry} ], [ {i2}, %{body} ]
  {bld_ptr} = call {bld_ty_str}.piecePtr {bld_prefix}.getPtrIndexed({bld_ty_str}* {bld_sym}, i32 {i})
  {val_scalar_ptr} = call {elem_ty_str}* {bld_prefix}.scalarMergePtrForPiece({bld_ty_str}.piecePtr {bld_ptr})
  {val_vector_ptr} = call {elem_vec_ty_str}* {bld_prefix}.vectorMergePtrForPiece({bld_ty_str}.piecePtr {bld_ptr})
  {val_scalar} = load {elem_ty_str}, {elem_ty_str}* {val_scalar_ptr}
  {val_vector} = load {elem_vec_ty_str}, {elem_vec_ty_str}* {val_vector_ptr}
; begin generated merge
