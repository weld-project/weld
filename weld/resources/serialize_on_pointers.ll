
define void {TYPE_PREFIX}.serialize_on_pointers(i8* %a, i8* %b) {{
  %aTypedPtr = bitcast i8* %a to %{BUFNAME}.growable*
  %bTypedPtr = bitcast i8* %b to {TYPE}*
  %aTyped = load %{BUFNAME}.growable, %{BUFNAME}.growable* %aTypedPtr
  %bTyped = load {TYPE}, {TYPE}* %bTypedPtr
  %result = call %{BUFNAME}.growable {SERIALIZE}(%{BUFNAME}.growable %aTyped, {TYPE} %bTyped)
  store %{BUFNAME}.growable %result, %{BUFNAME}.growable* %aTypedPtr
  ret void
}}
