
from ctypes import *

# Load the Weld Dynamic Library.
weld = CDLL("../../../../target/debug/libweld.dylib")

# Set up the function pointers for the other functions we'll call.
# TODO(shoumik) Could move this into another file or class.

weld_module_compile = weld.weld_module_compile
weld_module_compile.argtypes = [c_char_p, c_char_p, POINTER(c_void_p)]
weld_module_compile.restype = c_void_p

weld_module_run = weld.weld_module_run
weld_module_run.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
weld_module_run.restype = c_void_p

weld_object_new = weld.weld_object_new
weld_object_new.argtypes = [c_void_p]
weld_object_new.restype = c_void_p

weld_object_data = weld.weld_object_data
weld_object_data.argtypes = [c_void_p]
weld_object_data.restype = c_void_p

weld_error_free = weld.weld_error_free
weld_error_free.argtypes = [c_void_p]
weld_error_free.restype = None

weld_object_free = weld.weld_object_free
weld_object_free.argtypes = [c_void_p]
weld_object_free.restype = None

weld_module_free = weld.weld_module_free
weld_module_free.argtypes = [c_void_p]
weld_module_free.restype = None

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


