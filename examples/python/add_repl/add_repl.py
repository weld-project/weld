

import ctypes

from weld.bindings import *

# Setup the Weld code we want to run.
code = "|x:i64| x + 5L"
module = WeldModule(code, WeldConf(), WeldError())

while True:
    inp = raw_input(">>> ")
    try:
        if inp == "quit":
            break
        inp = int(inp)
    except ValueError:
        print "nope, try again."
        continue

    arg = ctypes.c_int64(inp)
    res_obj = module.run(WeldConf(), WeldValue(ctypes.byref(arg)), WeldError())
    data = res_obj.data()
    res_value = ctypes.cast(data, ctypes.POINTER(
        ctypes.c_int64)).contents.value
    print res_value

    # Free the object and its underlying data. If we want python to track the
    # data, we should first copy it's value out.
    res_obj.free()
