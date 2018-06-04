; Template for dictionary serialization routines.
;
; Parameters:
; - NAME: name of generated dictionary type, without % or @ prefix.
; - BUFNAME: name of vec[i8] buffer type, without % or @ prefix.
; - HAS_POINTER: designates whether the dictionary key or value has a pointer. Should be either '0' or '1'.
; - KEY_SERIALIZE_ON_PTR: key serialize function, if the key has a pointer.
; - VAL_SERIALIZE_ON_PTR: value serialize function, if the value has a pointer.

; Serialize the dictionary by calling its serialization routine.
define %{BUFNAME}.growable @{NAME}.serialize(%{BUFNAME}.growable %buf, %{NAME} %dict) {{
  %bufPtr = alloca %{BUFNAME}.growable
  store %{BUFNAME}.growable %buf, %{BUFNAME}.growable* %bufPtr
  %bufPtrRaw = bitcast %{BUFNAME}.growable* %bufPtr to i8*
  call void @weld_rt_dict_serialize(i8* %dict, i8* %bufPtrRaw, i32 {HAS_POINTER}, void (i8*, i8*)* {KEY_SERIALIZE_ON_PTR}, void (i8*, i8*)* {VAL_SERIALIZE_ON_PTR})
  %newBuf = load %{BUFNAME}.growable, %{BUFNAME}.growable* %bufPtr
  ret %{BUFNAME}.growable %newBuf
}}
