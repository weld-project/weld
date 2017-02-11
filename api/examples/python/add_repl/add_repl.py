
# Python wrappers for the C API
import ctypes
import weld

# Setup the Weld code we want to run.
code = ctypes.c_char_p("|x:i64| x + 5L")
conf = ctypes.c_char_p("configuration")
err = ctypes.c_void_p()
module = weld.weld_module_compile(code, conf, ctypes.byref(err))
print weld.weld_error_message(err)

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
    arg_obj = weld.weld_value_new(ctypes.byref(arg))

    # A new error handle
    err2 = ctypes.c_void_p()
    res_obj = weld.weld_module_run(module, arg_obj, ctypes.byref(err2))

    data = weld.weld_value_data(res_obj)

    res_value = ctypes.cast(data, ctypes.POINTER(ctypes.c_int64)).contents.value

    # Free everything.
    weld.weld_error_free(err2)
    weld.weld_value_free(arg_obj)
    weld.weld_value_free(res_obj)

    print res_value

# Free the module before quiting.
weld.weld_error_free(err)
weld.weld_module_free(module)


