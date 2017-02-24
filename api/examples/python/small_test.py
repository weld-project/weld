import os
import sys

home = os.environ.get("WELD_HOME")
if home is None:
    home = "../../python/"
if home[-1] != "/":
    home += "/"

libpath = home + "api/python"
sys.path.append(libpath)

import weld
import ctypes

arg = ctypes.c_int64(0) 
ptr = ctypes.cast(ctypes.byref(arg), ctypes.c_void_p)

# Weld code begins here.
arg_obj = weld.WeldValue(ptr)
mod = weld.WeldModule("||1", weld.WeldConf(), weld.WeldError())
value = mod.run(weld.WeldConf(), arg_obj, weld.WeldError())

print ctypes.cast(value.data(), ctypes.POINTER(ctypes.c_int32)).contents.value
