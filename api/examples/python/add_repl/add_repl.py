
# Python wrappers for the C API
from weld import *
from ctypes import *

# Setup the Weld code we want to run.
code = c_char_p("|x:i64| x + 5L")
conf = c_char_p("configuration")
err = c_void_p()
module = weld_module_compile(code, conf, byref(err))

while True:
    inp = raw_input(">>> ")
    try:
        if inp == "quit":
            break
        inp = int(inp)
    except ValueError:
        print "nope, try again."
        continue

    arg = c_int64(inp) 
    arg_obj = weld_object_new(byref(arg))

    # A new error handle
    err2 = c_void_p()
    res_obj = weld_module_run(module, arg_obj, byref(err2))

    data = weld_object_data(res_obj)

    res_value = cast(data, POINTER(c_int64)).contents.value

    # Free everything.
    weld_error_free(err2)
    weld_object_free(arg_obj)
    weld_object_free(res_obj)

    print res_value

# Free the module before quiting.
weld_error_free(err)
weld_module_free(module)


